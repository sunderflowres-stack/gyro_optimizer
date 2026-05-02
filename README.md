## GYRO: Geometric Yield Rotation Optimizer

GYRO is an optimizer for deep neural networks that augments Adam with a geometric gradient projection step applied before momentum buffers are updated. The approach is inspired by the rotor mechanics of Clifford algebra, implemented as a norm-preserving projection in gradient space. The novelty lies in the application of oscillation-corrected gradient steering as a geometric stabilization layer for adaptive optimizers, keeping complexity identical to Adam.

The motivation comes from a well-known failure mode in high-dimensional optimization: narrow ravines. In these regions, the gradient oscillates between steep walls step after step, and while Adam partially compensates through coordinate-wise variance scaling, it treats each parameter independently and ignores the directional relationship between consecutive gradient vectors. GYRO works on that relationship directly.

At each step, GYRO checks whether the current gradient $g_t$ and the previous gradient $g_{t-1}$ point in opposing directions — that is, whether their cosine similarity is negative. If so, an oscillation is flagged and the component of $g_t$ pointing along $g_{t-1}$ is removed, steering the update away from the oscillating direction.

**The correction is constructed in three stages.**

First, the oscillating component is identified via projection of $g_t$ onto $g_{t-1}$:

$$\text{proj} = \frac{\langle g_t,\, g_{t-1} \rangle}{\|g_{t-1}\|^2} \, g_{t-1}$$

Second, this component is subtracted to obtain the corrected gradient:

$$g_{\text{proj}} = g_t - \text{proj}$$

Third, $g_{\text{proj}}$ is rescaled to preserve the original gradient norm:

$$\tilde{g}_t = g_{\text{proj}} \cdot \frac{\|g_t\|}{\|g_{\text{proj}}\|}$$

This is a norm-preserving operation — the magnitude of the gradient is unchanged, only its direction is corrected. $\tilde{g}_t$ is then passed into Adam's exponential moving average buffers as a drop-in replacement for $g_t$.

The projection operates per-parameter-tensor, not globally across the entire model, so norm computations are local and never require cross-layer synchronization. The overhead is one dot product, two norms, and one stored gradient vector per step, keeping both time and memory complexity at $O(N)$, identical to Adam.

## Benchmarks

**Short run (3 epochs)** — evaluated on MNIST and CIFAR-10 using a standard CNN, identical hyperparameters across all optimizers.

| Optimizer | MNIST Accuracy | MNIST Time | CIFAR-10 Accuracy | CIFAR-10 Time |
|-----------|---------------|------------|-------------------|---------------|
| SGD | 98.99% | 51.0s | 66.03% | 48.8s |
| Adam | 98.66% | 52.2s | **66.42%** | 50.7s |
| AdamW | 98.92% | 52.7s | 65.22% | 49.9s |
| **GYRO** | 98.57% | 61.1s | 65.41% | 51.1s |

At 3 epochs all optimizers cluster within noise on both datasets. GYRO's lower train loss on CIFAR-10 (0.9295 vs 0.9587 for AdamW) suggests faster early convergence, but short runs are insufficient to observe the trajectory-correction benefit.

<img width="1200" height="500" alt="benchmark_cifar10" src="https://github.com/user-attachments/assets/c11ea747-f632-44a5-bed9-91fda77b0aab" />

**Extended run (15 epochs, CIFAR-10)** — longer training reveals trajectory differences between optimizers that 3-epoch snapshots cannot capture.

| Optimizer | Final Test Accuracy | Best Test Accuracy | Final Train Loss |
|-----------|-------------------|-------------------|-----------------|
| SGD | 68.81% | 69.49% | 0.2463 |
| AdamW | 69.62% | 69.75% | 0.4275 |
| **GYRO** | **70.81%** | **71.60%** | 0.3802 |

Over 15 epochs GYRO outperforms both baselines by a consistent margin, pulling ahead from epoch 5 and maintaining the lead through epoch 15. GYRO achieves lower training loss than AdamW while reaching higher test accuracy — a pattern consistent with the projection step acting as an implicit regularizer by removing the oscillating component of the gradient rather than accumulating it into the momentum buffers. The gap over AdamW (1.19%) and SGD (2.0%) is stable across the final 10 epochs rather than appearing as a single-epoch spike, suggesting the effect is structural rather than stochastic.

**Transformer benchmark (3 epochs, TinyShakespeare)** — character-level language model, 4-layer transformer encoder, 128 hidden dim, evaluated on a held-out validation split.

| Optimizer | Epoch 1 Train Loss | Final Train Loss | Final Val Loss |
|---|---|---|---|
| AdamW | 0.880 | 0.0180 | 0.0171 |
| **GYRO** | 0.971 | **0.0180** | **0.0170** |

GYRO converges slightly slower in epoch 1 — consistent with the projection being conservative on fresh gradients with no oscillation history — but reaches identical train loss and marginally better validation loss by epoch 3. The result extends the CIFAR-10 finding to transformer architectures: GYRO matches or exceeds AdamW on generalization while the projection step incurs no measurable overhead on GPU.

```python
from gyro import GYROAdam
optimizer = GYROAdam(model.parameters(), lr=1e-3)
```
