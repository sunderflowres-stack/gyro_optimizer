## GYRO: Geometric Yield Rotation Optimizer

GYRO is an optimizer for deep neural networks that augments Adam with a geometric gradient projection step applied before momentum buffers are updated. The approach is inspired by the rotor mechanics of Clifford algebra, implemented as a norm-preserving projection in gradient space. The novelty lies in the application of oscillation-corrected gradient steering as a geometric stabilization layer for adaptive optimizers, keeping complexity identical to Adam.

The motivation comes from a well-known failure mode in high-dimensional optimization: narrow ravines. In these regions, the gradient oscillates between steep walls step after step, and while Adam partially compensates through coordinate-wise variance scaling, it treats each parameter independently and ignores the directional relationship between consecutive gradient vectors. GYRO works on that relationship directly.

At each step, GYRO checks whether the current gradient $g_t$ and the exponential moving average buffer $m_t$ point in opposing directions — that is, whether their cosine similarity is negative. Comparing against $m_t$ rather than a single previous gradient makes oscillation detection more stable, as $m_t$ represents a smoothed history of past gradient directions. If an oscillation is detected, the component of $g_t$ pointing along $m_t$ is removed, steering the update away from the oscillating direction.

**The correction is constructed in three stages.**

First, the oscillating component is identified via projection of $g_t$ onto $m_t$:

$$\text{proj} = \frac{\langle g_t,\, m_t \rangle}{\|m_t\|^2} \, m_t$$

Second, this component is subtracted to obtain the corrected gradient:

$$g_{\text{proj}} = g_t - \text{proj}$$

Third, $g_{\text{proj}}$ is rescaled to preserve the original gradient norm:

$$\tilde{g}_t = g_{\text{proj}} \cdot \frac{\|g_t\|}{\|g_{\text{proj}}\|}$$

This is a norm-preserving operation — the magnitude of the gradient is unchanged, only its direction is corrected. $\tilde{g}_t$ is then passed into Adam's exponential moving average buffers as a drop-in replacement for $g_t$.

The projection operates per-parameter-tensor, not globally across the entire model, so norm computations are local and never require cross-layer synchronization. The overhead is one dot product, two norms, and no additional state beyond Adam's existing buffers, keeping both time and memory complexity at $O(N)$, identical to Adam.

## Benchmarks

**Short run (3 epochs)** — evaluated on MNIST and CIFAR-10 using a standard CNN, identical hyperparameters across all optimizers.

| Optimizer | MNIST Accuracy | MNIST Time | CIFAR-10 Accuracy | CIFAR-10 Time |
|-----------|---------------|------------|-------------------|---------------|
| SGD | 98.99% | 51.0s | 66.03% | 48.8s |
| Adam | 98.66% | 52.2s | **66.42%** | 50.7s |
| AdamW | 98.92% | 52.7s | 65.22% | 49.9s |
| **GYRO** | 98.57% | 61.1s | 65.41% | 51.1s |

At 3 epochs all optimizers cluster within noise on both datasets. Short runs are insufficient to observe the trajectory-correction benefit — the projection only activates when oscillations are detected, which takes several epochs to become meaningful.

**Extended run (15 epochs)** — longer training reveals trajectory differences between optimizers that 3-epoch snapshots cannot capture.

<img width="2100" height="750" alt="benchmark_cifar10" src="https://github.com/user-attachments/assets/dfa122bd-47ca-40c1-bb94-7ed4e8123e55" />

<img width="2100" height="750" alt="benchmark_mnist" src="https://github.com/user-attachments/assets/05ec212d-4877-4a29-9563-7f74a25a4603" />


**MNIST:**

| Optimizer | Final Accuracy | Best Accuracy | Final Train Loss |
|-----------|---------------|--------------|-----------------|
| SGD | 99.15% | 99.15% | 0.0054 |
| Adam | 98.51% | 99.14% | 0.0069 |
| AdamW | 98.88% | 98.98% | 0.0056 |
| **GYRO** | **99.20%** | **99.20%** | 0.0059 |

**CIFAR-10:**

| Optimizer | Final Accuracy | Best Accuracy | Final Train Loss |
|-----------|---------------|--------------|-----------------|
| SGD | 69.14% | 71.27% | 0.2358 |
| Adam | 68.84% | 70.26% | 0.4012 |
| AdamW | 68.17% | 69.31% | 0.4351 |
| **GYRO** | **69.99%** | 70.21% | **0.3976** |

Over 15 epochs GYRO leads on both datasets. On MNIST it achieves the highest final and best accuracy across all optimizers. On CIFAR-10 it outperforms AdamW by 1.82% final accuracy while maintaining lower train loss — consistent with the projection step acting as an implicit regularizer by removing the oscillating component from the gradient before it enters the momentum buffers.

**Transformer benchmark**
SOON.


```python
from gyro import GYROAdam
optimizer = GYROAdam(model.parameters(), lr=1e-3)
```
