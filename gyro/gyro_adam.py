import math
import torch
from torch.optim import Optimizer


class GYROAdam(Optimizer):
    """
    GYROAdam: Adam optimizer augmented with geometric gradient rotation.

    Detects gradient oscillations via negative cosine similarity between
    consecutive steps and applies a norm-preserving 2D planar rotation
    before updating Adam's exponential moving average buffers.

    The rotation operates per-parameter-tensor, not globally across the
    entire model, so norm computations are local and never require
    cross-layer synchronization.
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
                    # prev_grad allocated lazily on first oscillation detection
                    state['prev_grad'] = None

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                eps = group['eps']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                grad_rotated = grad.clone()
                prev_grad = state['prev_grad']

                if prev_grad is not None:
                    dot = torch.sum(grad * prev_grad)
                    # Per-tensor norms — local, no cross-layer sync
                    norm_g = torch.norm(grad)
                    norm_pg = torch.norm(prev_grad)

                    if norm_g > eps and norm_pg > eps:
                        cos_alpha = dot / (norm_g * norm_pg)
                        if cos_alpha < 0:
                            proj = (dot / (norm_g ** 2)) * grad
                            ortho = prev_grad - proj
                            norm_ortho = torch.norm(ortho)
                            if norm_ortho > eps:
                                ortho_normalized = ortho / norm_ortho
                                dynamic_scale = torch.tanh(norm_g)
                                theta = torch.tensor(math.pi / 2) * group['theta_base'] * dynamic_scale
                                grad_rotated = (grad * torch.cos(theta) +
                                                ortho_normalized * norm_g * torch.sin(theta))

                # Allocate prev_grad only after first step
                if state['prev_grad'] is None:
                    state['prev_grad'] = torch.zeros_like(grad)
                state['prev_grad'].copy_(p.grad if group['weight_decay'] == 0 else grad)

                exp_avg.mul_(beta1).add_(grad_rotated, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_rotated, grad_rotated, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
