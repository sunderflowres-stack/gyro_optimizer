import math
import torch
from torch.optim import Optimizer


class GYROAdam(Optimizer):
    """
    GYROAdam: Adam optimizer augmented with momentum-aware gradient projection.

    Detects gradient oscillations by comparing the current gradient against
    the exponential moving average buffer. When the cosine similarity drops
    below -theta_base, the oscillating component is partially or fully removed
    and the gradient is rescaled to preserve its original norm before updating
    Adam's EMA buffers.

    Args:
        params:       iterable of parameters to optimize
        lr:           learning rate (default: 1e-3)
        betas:        EMA coefficients for gradient and squared gradient (default: (0.9, 0.999))
        eps:          numerical stability term (default: 1e-8)
        weight_decay: decoupled weight decay coefficient (default: 0.0)
        theta_base:   oscillation threshold in [0, 1). Projection triggers when
                      cos(g_t, m_t) < -theta_base. At 0.0 any opposing direction
                      triggers correction. (default: 0.0)
        proj_factor:  soft projection strength in [0, 1]. At 1.0 the oscillating
                      component is fully removed. At 0.5 only half is removed. (default: 1.0)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, theta_base=0.0, proj_factor=1.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= theta_base < 1.0:
            raise ValueError(f"theta_base must be in [0, 1), got: {theta_base}")
        if not 0.0 <= proj_factor <= 1.0:
            raise ValueError(f"proj_factor must be in [0, 1], got: {proj_factor}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, theta_base=theta_base,
                        proj_factor=proj_factor)
        super(GYROAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('GYROAdam does not support sparse gradients.')

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg']    = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                eps          = group['eps']
                state['step'] += 1

                grad_rotated = grad.clone()
                grad_f32     = grad.float()
                exp_avg_f32  = exp_avg.float()
                norm_g       = torch.norm(grad_f32)
                norm_ea      = torch.norm(exp_avg_f32)

                if norm_g > eps and norm_ea > eps:
                    dot       = torch.sum(grad_f32 * exp_avg_f32)
                    cos_alpha = dot / (norm_g * norm_ea)

                    if cos_alpha < -group['theta_base']:
                        proj           = (dot / (norm_ea ** 2)) * exp_avg_f32
                        grad_proj      = grad_f32 - group['proj_factor'] * proj
                        norm_grad_proj = torch.norm(grad_proj)

                        if norm_grad_proj > eps:
                            grad_rotated = (grad_proj * (norm_g / norm_grad_proj)).to(grad.dtype)

                exp_avg.mul_(beta1).add_(grad_rotated, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_rotated, grad_rotated, value=1 - beta2)

                bias_correction1      = 1 - beta1 ** state['step']
                bias_correction2_sqrt = math.sqrt(1 - beta2 ** state['step'])
                step_size = group['lr'] * bias_correction2_sqrt / bias_correction1
                denom     = exp_avg_sq.sqrt().add_(eps * bias_correction2_sqrt)

                if group['weight_decay'] != 0:
                    p.mul_(1.0 - group['lr'] * group['weight_decay'])

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
