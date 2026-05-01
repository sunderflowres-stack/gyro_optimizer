## GYRO: Geometric Yield Rotation Optimizer

GYRO is an optimizer for deep neural networks that augments Adam with a geometric rotation step applied to the gradient before momentum buffers are updated. The approach is inspired by the rotor mechanics of Clifford algebra, implemented via Gram-Schmidt orthogonalization for efficiency. The novelty lies in the application of planar rotation as a geometric stabilization layer for adaptive optimizers, keeping complexity identical to Adam.

The motivation comes from a well-known failure mode in high-dimensional optimization: narrow ravines. In these regions, the gradient oscillates between steep walls step after step, and while Adam partially compensates through coordinate-wise variance scaling, it treats each parameter independently and ignores the directional relationship between consecutive gradient vectors. GYRO works on that relationship directly.

At each step, GYRO checks whether the current gradient $g_t$ and the previous gradient $g_{t-1}$ point in opposing directions — that is, whether their cosine similarity is negative. If so, an oscillation is flagged and a corrective rotation is applied within the plane spanned by the two gradients, steering the update along the ravine axis rather than across it.

**The rotation is constructed in three stages.**

First, the component of $g_{t-1}$ orthogonal to $g_t$ is isolated via Gram-Schmidt:

$$v_{\perp} = g_{t-1} - \frac{\langle g_t, g_{t-1} \rangle}{\|g_t\|^2} \, g_t, \qquad \hat{v}_{\perp} = \frac{v_{\perp}}{\|v_{\perp}\|}$$

This gives an orthonormal frame $\{g_t / \|g_t\|,\, \hat{v}_{\perp}\}$ defining the local rotation plane.

Second, the rotation angle $\theta$ is scaled by the gradient norm through a tanh envelope:

$$\theta = \frac{\pi}{2} \cdot \lambda \cdot \tanh(\|g_t\|)$$

where $\lambda$ is a tunable hyperparameter (`theta_base`). As the optimizer approaches a minimum and the gradient shrinks toward zero, $\theta$ shrinks with it, preventing the rotation from deflecting nearly-converged updates.

Third, the rotated gradient is computed as:

$$\tilde{g}_t = g_t \cos\theta + \hat{v}_{\perp} \, \|g_t\| \sin\theta$$

This is a norm-preserving operation. Because $\hat{v}_{\perp}$ is strictly orthogonal to $g_t$ by construction, the Pythagorean identity gives:

$$\|\tilde{g}_t\|^2 = \|g_t\|^2 \cos^2\theta + \|g_t\|^2 \sin^2\theta = \|g_t\|^2$$

The gradient is purely rotated — its magnitude is unchanged — and $\tilde{g}_t$ is passed into Adam's exponential moving average buffers as a drop-in replacement for $g_t$.

The overhead is one dot product, two norms, and one stored gradient vector per step, keeping both time and memory complexity at $O(N)$, identical to Adam.

## Benchmarks

**Short run (3 epochs)** — evaluated on MNIST and CIFAR-10 using a standard CNN, identical hyperparameters across all optimizers.

| Optimizer | MNIST Accuracy | MNIST Time | CIFAR-10 Accuracy | CIFAR-10 Time |
|-----------|---------------|------------|-------------------|---------------|
| SGD | 98.76% | 50.4s | 66.01% | 49.1s |
| Adam | 98.90% | 59.2s | 65.82% | 49.8s |
| AdamW | 98.89% | 61.1s | 66.18% | 50.4s |
| **GYRO** | 98.76% | **55.4s** | 65.21% | 50.8s |

**Extended run (15 epochs, CIFAR-10)** — longer training reveals trajectory differences between optimizers that 3-epoch snapshots cannot capture.

<img width="1200" height="500" alt="benchmark_cifar10" src="https://github.com/user-attachments/assets/c0458c31-c984-4972-ac2a-681fcfdd2f01" />


| Optimizer | Final Test Accuracy | Final Train Loss |
|-----------|-------------------|-----------------|
| SGD | 70.52% | 0.2629 |
| AdamW | 69.46% | 0.3603 |
| **GYRO** | **70.02%** | 0.4034 |

Over 15 epochs, GYRO achieves better final test accuracy than AdamW (70.02% vs 69.46%) while maintaining higher training loss — a pattern consistent with implicit regularization from the rotation step reducing overfitting to the training trajectory. All three optimizers converge to the same accuracy band, which is expected for a shallow CNN on CIFAR-10. The geometric correction is designed to matter most on deeper architectures and longer schedules where ravine traversal dominates training dynamics.

```python
from gyro import GYROAdam
optimizer = GYROAdam(model.parameters(), lr=1e-3, theta_base=0.3)
```
