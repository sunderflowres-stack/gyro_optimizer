import math
import torch
from torch.optim import Optimizer


class GYROAdam(Optimizer):
    """
    GYROAdam: Adam optimizer augmented with geometric gradient projection.

    Detects gradient oscillations via negative cosine similarity between
    consecutive steps and removes the component of the current gradient
    pointing along the previous gradient direction, then rescales to
    preserve the original gradient norm before updating Adam's EMA buffers.

    The projection operates per-parameter-tensor, not globally across the
    entire model, so norm computations are local and never require
    cross-layer synchronization.

    Args:
        params:       iterable of parameters to optimize
        lr:           learning rate (default: 1e-3)
        betas:        coefficients for EMA of gradient and squared gradient (default: (0.9, 0.999))
        eps:          numerical stability term (default: 1e-8)
        weight_decay: decoupled weight decay coefficient (default: 0.0)
        theta_base:   retained for backward compatibility, currently unused (default: 0.3)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, theta_base=0.3):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, theta_base=theta_base)
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
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['prev_grad'] = None

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                eps = group['eps']
                state['step'] += 1

                grad_rotated = grad.clone()
                prev_grad = state['prev_grad']

                if prev_grad is not None:
                    dot = torch.sum(grad * prev_grad)
                    norm_g = torch.norm(grad)
                    norm_pg = torch.norm(prev_grad)

                    if norm_g > eps and norm_pg > eps:
                        cos_alpha = dot / (norm_g * norm_pg)
                        if cos_alpha < 0:
                            proj = (dot / (norm_pg ** 2)) * prev_grad
                            grad_proj = grad - proj
                            norm_grad_proj = torch.norm(grad_proj)
                            if norm_grad_proj > eps:
                                grad_rotated = grad_proj * (norm_g / norm_grad_proj)

                if state['prev_grad'] is None:
                    state['prev_grad'] = torch.zeros_like(grad)
                state['prev_grad'].copy_(p.grad)

                exp_avg.mul_(beta1).add_(grad_rotated, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_rotated, grad_rotated, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                if group['weight_decay'] != 0:
                    p.data.mul_(1.0 - group['lr'] * group['weight_decay'])

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
