import torch
from torch.optim import Optimizer


class GYROSGD(Optimizer):
    """
    GYROSGD: SGD augmented with geometric gradient projection.

    Detects gradient oscillations via negative cosine similarity between
    consecutive steps and removes the component of the current gradient
    pointing along the previous gradient direction, then rescales to
    preserve the original gradient norm.

    The projection operates per-parameter-tensor, not globally across the
    entire model, so norm computations are local and never require
    cross-layer synchronization.

    Args:
        params:     iterable of parameters to optimize
        lr:         learning rate (default: 1e-3)
        eps:        numerical stability term (default: 1e-8)
        theta_base: retained for backward compatibility, currently unused (default: 0.3)
    """
    def __init__(self, params, lr=1e-3, theta_base=0.3, eps=1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, theta_base=theta_base, eps=eps)
        super(GYROSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.clone()
                state = self.state[p]
                if len(state) == 0:
                    state['prev_grad'] = None

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
                                grad = grad_proj * (norm_g / norm_grad_proj)

                if state['prev_grad'] is None:
                    state['prev_grad'] = torch.zeros_like(grad)
                state['prev_grad'].copy_(grad)

                p.add_(grad, alpha=-lr)

        return loss
