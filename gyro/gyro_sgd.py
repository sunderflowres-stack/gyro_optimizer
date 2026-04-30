import math
import torch
from torch.optim import Optimizer


class GYROSGD(Optimizer):
    """
    GYROSGD: SGD augmented with geometric gradient rotation.

    Detects gradient oscillations via negative cosine similarity between
    consecutive steps and applies a norm-preserving 2D planar rotation
    to bypass saddle points and narrow ravines.
    """
    def __init__(self, params, lr=1e-3, theta_base=0.3, eps=1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if theta_base < 0.0:
            raise ValueError(f"Invalid theta_base: {theta_base}")
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
            theta_base = group['theta_base']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['prev_grad'] = torch.zeros_like(grad)

                prev_grad = state['prev_grad']
                dot = torch.sum(grad * prev_grad)
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
                            theta = torch.tensor(math.pi / 2) * theta_base * dynamic_scale
                            grad_rotated = (grad * torch.cos(theta) +
                                            ortho_normalized * norm_g * torch.sin(theta))
                            grad = grad_rotated

                p.add_(grad, alpha=-lr)
                state['prev_grad'].copy_(grad)

        return loss
