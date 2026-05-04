## GYRO: Geometric Yield Rotation Optimizer

GYRO is a lightweight PyTorch optimizer utility and side project. It is not an academic paper, but a practical code release aimed at experimenting with gradient steering. It augments Adam with a gradient projection step applied before momentum buffers are updated: when the current gradient and the accumulated momentum buffer point in opposing directions, the oscillating component is removed before the update is applied.

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

## Related Work

The mathematical operation of projecting conflicting vectors has been successfully used in multi-task learning — notably PCGrad (Yu et al., 2020) — to resolve interfering task gradients. GYRO applies a similar geometric projection conceptually, but acts temporally rather than spatially: it projects the current gradient against the momentum buffer to resolve trajectory oscillations within a single-task setting.

A closely related approach is the Cautious Optimizer (Liang et al., 2024), which also uses the dot product of the instant gradient with the momentum buffer to decide upon a correction. The key difference is where the correction is applied: Cautious modifies the final parameter update with a sign-based mask, leaving momentum buffers unchanged. GYRO applies the correction to the gradient itself before it enters the buffers, so the effect persists in accumulated state across subsequent steps. Whether this distinction matters in practice is an open question.

The idea of using gradient direction history to improve optimizer trajectory is also explored in Lookahead (Zhang et al., 2019) and Gradient Centralization (Yong et al., 2020), though the mechanisms differ.

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `lr` | `1e-3` | Learning rate |
| `betas` | `(0.9, 0.999)` | EMA coefficients, same as Adam |
| `eps` | `1e-8` | Numerical stability term |
| `weight_decay` | `0.0` | Decoupled weight decay |
| `theta_base` | `0.0` | Oscillation threshold. Projection triggers when cos(g, m) < -theta_base. Higher values ignore weak oscillations. |
| `proj_factor` | `1.0` | Soft projection strength. 1.0 fully removes the oscillating component, 0.5 removes half. |
| `warmup_steps` | auto | Steps before projection activates. Defaults to round(1/(1-beta2)) ≈ 1000. Allows momentum to stabilize first. |
| `telemetry` | `False` | If True, exposes per-step diagnostics via `get_telemetry()`. |

## Telemetry

```python
optimizer = GYROAdam(model.parameters(), lr=1e-3, telemetry=True)

# After each optimizer.step():
tel = optimizer.get_telemetry()
# tel['mean_cos']        — mean cosine similarity between g_t and m_t
# tel['projection_rate'] — fraction of tensors where projection triggered
# tel['mean_norm_g']     — mean gradient norm
# tel['mean_norm_m']     — mean momentum buffer norm
```

Run `examples/telemetry_example.py` to visualize these diagnostics over training.

## Benchmarks

Results are reported honestly. On short runs and simple datasets GYRO is within noise of other optimizers — the projection activates only when oscillations are detected, which requires several epochs of momentum accumulation to become meaningful. Differences are more visible on longer runs and harder tasks.

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

*Narrow ravine* — f(x) = 100·x₀² + x₁², global minimum at 0, 3000 steps. SGD requires careful learning rate tuning on ill-conditioned problems — with the default lr used here it stalls, while all adaptive optimizers converge cleanly.

| Optimizer | Final Loss |
|-----------|-----------|
| SGD | 1.065446 |
| Adam | ~0 |
| AdamW | ~0 |
| GYRO | ~0 |

**Extended CNN run (15 epochs, CIFAR-10)**


<img width="2100" height="750" alt="benchmark_cifar10" src="https://github.com/user-attachments/assets/d2f2d473-0623-49a7-9f03-32734d2b4055" />


| Optimizer | Final Accuracy | Best Accuracy | Final Train Loss |
|-----------|---------------|--------------|-----------------|
| SGD | 68.91% | 69.70% | 0.2625 |
| Adam | 69.48% | 69.48% | 0.4568 |
| AdamW | 69.63% | 70.32% | 0.4047 |
| GYRO | 69.12% | **70.41%** | **0.1912** |

GYRO achieves the lowest training loss by a significant margin (0.1912 vs 0.4047 for AdamW) and the highest best accuracy. The large gap in training loss relative to test accuracy is consistent with the projection acting as an implicit regularizer — the model is not overfitting the training trajectory as aggressively.

**Sanity check (MNIST)**

<img width="2100" height="750" alt="benchmark_mnist" src="https://github.com/user-attachments/assets/fdd8dff2-7c04-4eb8-b672-63d919e1f82a" />


| Optimizer | Final Accuracy | Best Accuracy | Final Train Loss |
|-----------|---------------|--------------|-----------------|
| SGD | 99.17% | 99.17% | 0.0057 |
| Adam | 98.98% | 99.16% | 0.0053 |
| AdamW | 98.92% | 99.09% | 0.0053 |
| GYRO | 98.94% | **99.17%** | **0.0044** |

All optimizers cluster within noise on MNIST as expected. GYRO matches the best accuracy of any optimizer and achieves the lowest training loss.

## Usage

```python
from gyro import GYROAdam, GYROSGD

# Drop-in Adam replacement
optimizer = GYROAdam(model.parameters(), lr=1e-3)

# With soft projection, threshold, and warmup
optimizer = GYROAdam(model.parameters(), lr=1e-3, theta_base=0.1, proj_factor=0.8, warmup_steps=500)

# With telemetry
optimizer = GYROAdam(model.parameters(), lr=1e-3, telemetry=True)

# SGD variant
optimizer = GYROSGD(model.parameters(), lr=1e-2, momentum=0.9)
```
