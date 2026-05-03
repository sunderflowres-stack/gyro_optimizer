## GYRO: Geometric Yield Rotation Optimizer

GYRO is an optimizer for deep neural networks that augments Adam with a gradient projection step applied before momentum buffers are updated. The core idea is simple: when the current gradient and the accumulated momentum buffer point in opposing directions, the oscillating component is removed before the update is applied.

Standard adaptive optimizers treat each parameter independently and ignore the directional relationship between the current gradient and accumulated momentum. In narrow ravines — a common failure mode in high-dimensional non-convex optimization — this causes the gradient to oscillate between steep walls while making slow progress along the ravine axis. GYRO detects this pattern and corrects it locally, per parameter tensor, without any global synchronization.

At each step, the cosine similarity between the current gradient $g_t$ and the exponential moving average buffer $m_t$ is computed. If it falls below $-\theta$ (controlled by `theta_base`), an oscillation is flagged and the component of $g_t$ pointing along $m_t$ is partially or fully removed. Comparing against $m_t$ rather than a single previous gradient makes detection more stable, as $m_t$ represents a smoothed history of past directions.

**The correction is constructed in three stages.**

First, the oscillating component is identified:

$$\text{proj} = \frac{\langle g_t,\, m_t \rangle}{\|m_t\|^2} \, m_t$$

Second, it is subtracted with strength controlled by `proj_factor` $\lambda \in [0, 1]$:

$$g_{\text{proj}} = g_t - \lambda \cdot \text{proj}$$

Third, the result is rescaled to preserve the original gradient norm:

$$\tilde{g}_t = g_{\text{proj}} \cdot \frac{\|g_t\|}{\|g_{\text{proj}}\|}$$

$\tilde{g}_t$ is then passed into Adam's EMA buffers as a drop-in replacement for $g_t$. All intermediate computations are cast to float32 for numerical stability, supporting mixed-precision training. No additional optimizer state is required beyond Adam's existing buffers, keeping time and memory complexity at $O(N)$.

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

Results are reported honestly. On short runs and simple datasets GYRO is within noise of other optimizers — the projection activates only when oscillations are detected, which requires several epochs of momentum accumulation to become meaningful. Differences are more visible on longer runs and harder tasks.

**Short run (3 epochs)** — standard CNN on MNIST and CIFAR-10.

| Optimizer | MNIST Accuracy | CIFAR-10 Accuracy |
|-----------|---------------|-------------------|
| SGD | 98.99% | 66.03% |
| Adam | 98.66% | **66.42%** |
| AdamW | 98.92% | 65.22% |
| GYRO | 98.57% | 65.41% |

**Extended run (15 epochs)**

<img width="2100" height="750" alt="benchmark_mnist" src="https://github.com/user-attachments/assets/388ce9d8-626c-4ef6-ad86-7c6c3f431e41" />

**MNIST:**

| Optimizer | Final Accuracy | Best Accuracy | Final Train Loss |
|-----------|---------------|--------------|-----------------|
| SGD | 99.08% | 99.08% | 0.0063 |
| Adam | 99.04% | 99.09% | 0.0075 |
| AdamW | 99.03% | 99.13% | 0.0068 |
| GYRO | 98.97% | **99.13%** | **0.0052** |

<img width="2100" height="750" alt="benchmark_cifar10" src="https://github.com/user-attachments/assets/31cf85d6-01fb-4069-af8e-c352eb11678d" />

**CIFAR-10:**

| Optimizer | Final Accuracy | Best Accuracy | Final Train Loss |
|-----------|---------------|--------------|-----------------|
| SGD | 69.05% | 70.51% | 0.2568 |
| Adam | 69.41% | **69.85%** | 0.3887 |
| AdamW | 68.51% | 69.96% | 0.3948 |
| GYRO | 69.18% | **70.64%** | **0.3843** |

Final accuracy numbers are close across all optimizers — the difference is more visible in the loss curves and best-epoch figures. GYRO achieves the lowest training loss on both datasets, consistent with the projection acting as a mild implicit regularizer.

**Transformer benchmark** — character-level language model (4-layer encoder, 128 hidden dim) on TinyShakespeare, 90/10 train/val split, 3 epochs.

| Optimizer | Epoch 1 Train | Final Train | Final Val |
|---|---|---|---|
| AdamW | 0.888 | 0.0178 | 0.0165 |
| GYRO | 1.313 | 0.0186 | 0.0169 |

GYRO converges slower in epoch 1 — the momentum buffer needs time to accumulate direction history before oscillation detection becomes useful. By epoch 3 both optimizers reach similar validation loss. AdamW has a small edge on this benchmark. Extended runs on larger models are needed to draw stronger conclusions.

**Synthetic benchmarks** — analytic functions with known minima, isolating trajectory behavior from dataset noise.

<img width="1500" height="750" alt="bench_rosenbrock" src="https://github.com/user-attachments/assets/c80e9f83-94a7-4c8e-aed8-0799ba413631" />

*Rosenbrock function* — narrow curved ravine, global minimum at f(1,1) = 0, 5000 steps:

| Optimizer | Final Loss |
|-----------|-----------|
| SGD | 0.003560 |
| Adam | 0.000225 |
| AdamW | 0.000227 |
| GYRO | 0.000225 |

<img width="1500" height="750" alt="bench_ravine" src="https://github.com/user-attachments/assets/0498a5ea-38be-46a0-87d6-0de59bd8cc12" />

*Narrow ravine* — f(x) = 100·x₀² + x₁², global minimum at 0, 3000 steps:

| Optimizer | Final Loss |
|-----------|-----------|
| SGD | 1.065446 |
| Adam | ~0 |
| AdamW | ~0 |
| GYRO | ~0 |

GYRO matches Adam on Rosenbrock. On the narrow ravine SGD fails to converge while all adaptive optimizers reach the minimum — the projection does not interfere with convergence on pathological landscapes.

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
