import math
import torch
from torch.optim import Optimizer


class CliffordAdam(Optimizer):
    """
    CliffordAdam: Adam optimizer augmented with Clifford geometric rotation.
    
    Combines the adaptive learning rate mechanics of Adam with the topological
    saddle-point bypassing of Clifford algebra. Rotates the gradient vector 
    before feeding it into the exponential moving average momentum buffers.
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
        super(CliffordAdam, self).__init__(params, defaults)

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
                    raise RuntimeError('CliffordAdam does not support sparse gradients.')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['prev_grad'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                prev_grad = state['prev_grad']
                beta1, beta2 = group['betas']
                eps = group['eps']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                dot = torch.sum(grad * prev_grad)
                norm_g = torch.norm(grad)
                norm_pg = torch.norm(prev_grad)
                
                grad_rotated = grad.clone()

                if norm_g > eps and norm_pg > eps:
                    cos_alpha = dot / (norm_g * norm_pg)

                    # Detect oscillation and apply geometric rotation
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

                # Standard Adam mechanics with the rotated gradient
                exp_avg.mul_(beta1).add_(grad_rotated, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_rotated, grad_rotated, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                p.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Store the true gradient for the next step's bivector calculation
                state['prev_grad'].copy_(grad)

        return loss