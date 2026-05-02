import torch
from torch.optim import Optimizer


class GYROSGD(Optimizer):
    """
    GYROSGD: SGD augmented with geometric gradient projection.

    Detects gradient oscillations by comparing the current gradient against
    the exponential moving average buffer. Removes the oscillating component
    and rescales to preserve the original gradient norm.

    The projection operates per-parameter-tensor, not globally across the
    entire model, so norm computations are local and never require
    cross-layer synchronization.

    Args:
        params:     iterable of parameters to optimize
        lr:         learning rate (default: 1e-3)
        momentum:   EMA decay for gradient buffer (default: 0.9)
        eps:        numerical stability term (default: 1e-8)
        theta_base: retained for backward compatibility, currently unused (default: 0.3)
    """
    def __init__(self, params, lr=1e-3, momentum=0.9, theta_base=0.3, eps=1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        defaults = dict(lr=lr, momentum=momentum, theta_base=theta_base, eps=eps)
        super(GYROSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.clone()
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(grad)

                exp_avg = state['exp_avg']
                grad_projected = grad.clone()

                norm_g = torch.norm(grad)
                norm_ea = torch.norm(exp_avg)

                if norm_g > eps and norm_ea > eps:
                    dot = torch.sum(grad * exp_avg)
                    cos_alpha = dot / (norm_g * norm_ea)
                    if cos_alpha < 0:
                        proj = (dot / (norm_ea ** 2)) * exp_avg
                        grad_proj = grad - proj
                        norm_grad_proj = torch.norm(grad_proj)
                        if norm_grad_proj > eps:
                            grad_projected = grad_proj * (norm_g / norm_grad_proj)

                exp_avg.mul_(momentum).add_(grad_projected, alpha=1 - momentum)
                p.add_(grad_projected, alpha=-lr)

        return loss
