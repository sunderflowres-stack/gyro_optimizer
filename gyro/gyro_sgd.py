import torch
from torch.optim import Optimizer


class GYROSGD(Optimizer):
    """
    GYROSGD: SGD augmented with momentum-aware gradient projection.

    Detects gradient oscillations by comparing the current gradient against
    the exponential moving average buffer. When the cosine similarity drops
    below -theta_base, the oscillating component is partially or fully removed
    and the gradient is rescaled to preserve its original norm.

    Related to Cautious Optimizers (Liang et al., 2024), which apply a
    sign-based mask at the final update stage. GYRO instead modifies the
    gradient before it enters the momentum buffers.

    Args:
        params:       iterable of parameters to optimize
        lr:           learning rate (default: 1e-3)
        momentum:     EMA decay for gradient buffer (default: 0.9)
        eps:          numerical stability term (default: 1e-8)
        theta_base:   oscillation threshold in [0, 1). (default: 0.0)
        proj_factor:  soft projection strength in [0, 1]. (default: 1.0)
        warmup_steps: steps before projection activates. Defaults to
                      round(1 / (1 - momentum)) if None. (default: None)
        telemetry:    if True, stores diagnostic info via get_telemetry(). (default: False)
    """
    def __init__(self, params, lr=1e-3, momentum=0.9, theta_base=0.0,
                 proj_factor=1.0, warmup_steps=None, eps=1e-8, telemetry=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not 0.0 <= theta_base < 1.0:
            raise ValueError(f"theta_base must be in [0, 1), got: {theta_base}")
        if not 0.0 <= proj_factor <= 1.0:
            raise ValueError(f"proj_factor must be in [0, 1], got: {proj_factor}")

        if warmup_steps is None:
            warmup_steps = round(1.0 / (1.0 - momentum))

        defaults = dict(lr=lr, momentum=momentum, theta_base=theta_base,
                        proj_factor=proj_factor, warmup_steps=warmup_steps,
                        eps=eps, telemetry=telemetry)
        super(GYROSGD, self).__init__(params, defaults)

        self._tel_cos_sum     = 0.0
        self._tel_proj_count  = 0
        self._tel_total_count = 0
        self._tel_norm_g_sum  = 0.0
        self._tel_norm_m_sum  = 0.0
        self._tel_step        = 0

    def get_telemetry(self):
        n = max(self._tel_total_count, 1)
        return {
            'step':            self._tel_step,
            'mean_cos':        self._tel_cos_sum / n,
            'projection_rate': self._tel_proj_count / n,
            'mean_norm_g':     self._tel_norm_g_sum / n,
            'mean_norm_m':     self._tel_norm_m_sum / n,
        }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._tel_cos_sum     = 0.0
        self._tel_proj_count  = 0
        self._tel_total_count = 0
        self._tel_norm_g_sum  = 0.0
        self._tel_norm_m_sum  = 0.0

        for group in self.param_groups:
            lr          = group['lr']
            momentum    = group['momentum']
            eps         = group['eps']
            theta_base  = group['theta_base']
            proj_factor = group['proj_factor']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step']    = 0
                    state['exp_avg'] = torch.zeros_like(grad, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                state['step'] += 1
                self._tel_step = state['step']

                grad_projected = grad.clone()
                grad_f32       = grad.float()
                exp_avg_f32    = exp_avg.float()
                norm_g         = torch.norm(grad_f32)
                norm_ea        = torch.norm(exp_avg_f32)

                if group['telemetry']:
                    self._tel_total_count += 1
                    self._tel_norm_g_sum  += norm_g.item()
                    self._tel_norm_m_sum  += norm_ea.item()

                warmed_up = state['step'] > group['warmup_steps']

                if warmed_up and norm_g > eps and norm_ea > eps:
                    dot       = torch.sum(grad_f32 * exp_avg_f32)
                    cos_alpha = dot / (norm_g * norm_ea)

                    if group['telemetry']:
                        self._tel_cos_sum += cos_alpha.item()

                    if cos_alpha < -theta_base:
                        proj           = (dot / (norm_ea ** 2)) * exp_avg_f32
                        grad_proj      = grad_f32 - proj_factor * proj
                        norm_grad_proj = torch.norm(grad_proj)

                        if norm_grad_proj > eps:
                            grad_projected = (grad_proj * (norm_g / norm_grad_proj)).to(grad.dtype)
                            if group['telemetry']:
                                self._tel_proj_count += 1

                elif group['telemetry'] and norm_g > eps and norm_ea > eps:
                    dot       = torch.sum(grad_f32 * exp_avg_f32)
                    cos_alpha = dot / (norm_g * norm_ea)
                    self._tel_cos_sum += cos_alpha.item()

                exp_avg.mul_(momentum).add_(grad_projected, alpha=1 - momentum)
                p.add_(grad_projected, alpha=-lr)

        return loss
