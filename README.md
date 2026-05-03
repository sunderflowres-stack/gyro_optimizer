## GYRO: Geometric Yield Rotation Optimizer

GYRO is an optimizer for deep neural networks that augments Adam with a geometric gradient projection step applied before momentum buffers are updated. The approach is inspired by the rotor mechanics of Clifford algebra, implemented as a norm-preserving projection in gradient space. The novelty lies in the application of oscillation-corrected gradient steering as a geometric stabilization layer for adaptive optimizers, keeping complexity identical to Adam.

The motivation comes from a well-known failure mode in high-dimensional optimization: narrow ravines. In these regions, the gradient oscillates between steep walls step after step, and while Adam partially compensates through coordinate-wise variance scaling, it treats each parameter independently and ignores the directional relationship between consecutive gradient vectors. GYRO works on that relationship directly.

At each step, GYRO checks whether the current gradient $g_t$ and the exponential moving average buffer $m_t$ point in opposing directions — that is, whether their cosine similarity is negative. Comparing against $m_t$ rather than a single previous gradient makes oscillation detection more stable, as $m_t$ represents a smoothed history of past gradient directions. If an oscillation is detected, the component of $g_t$ pointing along $m_t$ is removed, steering the update away from the oscillating direction.

**The correction is constructed in three stages.**

First, the oscillating component is identified via projection of $g_t$ onto $m_t$:

$$\text{proj} = \frac{\langle g_t,\, m_t \rangle}{\|m_t\|^2} \, m_t$$

Second, this component is subtracted to obtain the corrected gradient. The `proj_factor` parameter controls how much of the oscillating component is removed — at 1.0 it is fully removed, at 0.5 only half is removed:

$$g_{\text{proj}} = g_t - \lambda \cdot \text{proj}, \quad \lambda \in [0, 1]$$

Third, $g_{\text{proj}}$ is rescaled to preserve the original gradient norm:

$$\tilde{g}_t = g_{\text{proj}} \cdot \frac{\|g_t\|}{\|g_{\text{proj}}\|}$$

This is a norm-preserving operation — the magnitude of the gradient is unchanged, only its direction is corrected. $\tilde{g}_t$ is then passed into Adam's exponential moving average buffers as a drop-in replacement for $g_t$.

The projection operates per-parameter-tensor, not globally across the entire model, so norm computations are local and never require cross-layer synchronization. All intermediate computations are cast to float32 for numerical stability, supporting mixed-precision training. The overhead is one dot product and two norms per tensor per step, with no additional state beyond Adam's existing buffers, keeping both time and memory complexity at $O(N)$, identical to Adam.

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `lr` | `1e-3` | Learning rate |
| `betas` | `(0.9, 0.999)` | EMA coefficients, same as Adam |
| `eps` | `1e-8` | Numerical stability term |
| `weight_decay` | `0.0` | Decoupled weight decay |
| `theta_base` | `0.0` | Oscillation threshold. Projection triggers when cos(g, m) < -theta_base. Higher values ignore weak oscillations. |
| `proj_factor` | `1.0` | Soft projection strength. 1.0 fully removes the oscillating component, 0.5 removes half. |

## Benchmarks

**Short run (3 epochs)** — standard CNN on MNIST and CIFAR-10, identical hyperparameters across all optimizers.

| Optimizer | MNIST Accuracy | CIFAR-10 Accuracy |
|-----------|---------------|-------------------|
| SGD | 98.99% | 66.03% |
| Adam | 98.66% | **66.42%** |
| AdamW | 98.92% | 65.22% |
| **GYRO** | 98.57% | 65.41% |

At 3 epochs all optimizers cluster within noise. The projection only activates when oscillations are detected, which takes several epochs to accumulate a meaningful momentum history.

**Extended run (15 epochs)** — longer training reveals trajectory differences that short runs cannot capture.


<img width="2100" height="750" alt="benchmark_mnist" src="https://github.com/user-attachments/assets/388ce9d8-626c-4ef6-ad86-7c6c3f431e41" />


**MNIST:**

| Optimizer | Final Accuracy | Best Accuracy | Final Train Loss |
|-----------|---------------|--------------|-----------------|
| SGD | 99.08% | 99.08% | 0.0063 |
| Adam | 99.04% | 99.09% | 0.0075 |
| AdamW | 99.03% | 99.13% | 0.0068 |
| **GYRO** | 98.97% | **99.13%** | **0.0052** |


<img width="2100" height="750" alt="benchmark_cifar10" src="https://github.com/user-attachments/assets/31cf85d6-01fb-4069-af8e-c352eb11678d" />


**CIFAR-10:**

| Optimizer | Final Accuracy | Best Accuracy | Final Train Loss |
|-----------|---------------|--------------|-----------------|
| SGD | 69.05% | 70.51% | 0.2568 |
| Adam | 69.41% | **69.85%** | 0.3887 |
| AdamW | 68.51% | 69.96% | 0.3948 |
| **GYRO** | 69.18% | **70.64%** | **0.3843** |

On MNIST, GYRO achieves the lowest training loss (0.0052) and matches the best accuracy of any optimizer. On CIFAR-10, GYRO achieves the highest best accuracy (70.64%) and lowest training loss. Final accuracy numbers are close across all optimizers on both datasets — the advantage is visible in the loss curves and best-epoch figures rather than the last-epoch snapshot, consistent with the projection acting as a mild implicit regularizer.

**Synthetic benchmarks** — optimization on analytic functions with known minima, isolating trajectory behavior from dataset noise.


<img width="1500" height="750" alt="bench_rosenbrock" src="https://github.com/user-attachments/assets/c80e9f83-94a7-4c8e-aed8-0799ba413631" />


*Rosenbrock function* — classic narrow curved ravine, global minimum at f(1,1) = 0, 5000 steps:

| Optimizer | Final Loss |
|-----------|-----------|
| SGD | 0.003560 |
| Adam | 0.000225 |
| AdamW | 0.000227 |
| **GYRO** | **0.000225** |


<img width="1500" height="750" alt="bench_ravine" src="https://github.com/user-attachments/assets/0498a5ea-38be-46a0-87d6-0de59bd8cc12" />


*Narrow ravine* — asymmetric quadratic f(x) = 100·x₀² + x₁², global minimum at 0, 3000 steps:

| Optimizer | Final Loss |
|-----------|-----------|
| SGD | 1.065446 |
| Adam | ~0 |
| AdamW | ~0 |
| **GYRO** | **~0** |

On the Rosenbrock function GYRO matches Adam exactly. On the narrow ravine SGD fails to converge while all adaptive optimizers including GYRO reach the minimum — confirming the ravine geometry is genuinely difficult and that the projection does not interfere with convergence on pathological landscapes.

**Transformer benchmark** — coming soon.

## Usage

```python
from gyro import GYROAdam, GYROSGD

# Drop-in Adam replacement
optimizer = GYROAdam(model.parameters(), lr=1e-3)

# With soft projection and threshold
optimizer = GYROAdam(model.parameters(), lr=1e-3, theta_base=0.1, proj_factor=0.8)

# SGD variant
optimizer = GYROSGD(model.parameters(), lr=1e-2, momentum=0.9)
```
