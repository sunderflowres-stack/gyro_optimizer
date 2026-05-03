import os
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from torch.optim import Adam, AdamW, SGD

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gyro import GYROAdam


def rosenbrock(x, a=1.0, b=100.0):
    """Global minimum at (a, a) = 0."""
    return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2


def narrow_ravine(x, scale=100.0):
    """
    Asymmetric quadratic ravine: steep in x[0], flat in x[1].
    f(x) = scale * x[0]^2 + x[1]^2
    Global minimum at origin.
    This directly simulates the narrow ravine geometry that causes
    gradient oscillation — steep walls in one direction, gentle slope
    in the other.
    """
    return scale * x[0] ** 2 + x[1] ** 2


def optimize(loss_fn, optimizer_class, name, start, steps=5000, **kwargs):
    x = start.clone().detach().requires_grad_(True)
    optimizer = optimizer_class([x], **kwargs)
    history = []

    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(x)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  [{name}] Diverged at step {len(history)}")
            history += [float('nan')] * (steps - len(history))
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_([x], max_norm=10.0)
        optimizer.step()
        history.append(loss.item())

    final = history[-1] if not torch.isnan(torch.tensor(history[-1])) else float('nan')
    print(f"  [{name}] Final loss: {final:.6f}")
    return history


def plot(results, title, filename, steps, log_scale=True):
    plt.figure(figsize=(10, 5))
    for name, hist in results.items():
        valid = [(i, v) for i, v in enumerate(hist) if not (v != v)]
        if valid:
            xs, ys = zip(*valid)
            plt.plot(xs, ys, label=name, linewidth=1.5)
    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Loss (log scale)' if log_scale else 'Loss')
    if log_scale:
        plt.yscale('log')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Chart saved as {out}")


def bench_rosenbrock(steps=5000):
    print("\n── Rosenbrock Benchmark")
    print("  Target: f(1, 1) = 0  |  Start: (-1.0, 1.0)\n")

    # Start near the ravine, not deep in the bowl
    start = torch.tensor([-1.0, 1.0], dtype=torch.float32)
    fn = rosenbrock
    results = {}

    optimizers = [
        (SGD,      "SGD",   dict(lr=1e-4, momentum=0.9)),
        (Adam,     "Adam",  dict(lr=1e-3)),
        (AdamW,    "AdamW", dict(lr=1e-3, weight_decay=1e-4)),
        (GYROAdam, "GYRO",  dict(lr=1e-3)),
    ]

    for cls, name, kwargs in optimizers:
        results[name] = optimize(fn, cls, name, start=start, steps=steps, **kwargs)

    plot(results,
         title=f'Rosenbrock Function ({steps} steps) — lower is better',
         filename='bench_rosenbrock.png',
         steps=steps)


def bench_ravine(steps=3000):
    print(f"\n── Narrow Ravine Benchmark")
    print(f"  f(x) = 100*x0^2 + x1^2  |  Target: 0  |  Start: (2.0, 2.0)\n")

    start = torch.tensor([2.0, 2.0], dtype=torch.float32)
    fn = narrow_ravine
    results = {}

    optimizers = [
        (SGD,      "SGD",   dict(lr=1e-2, momentum=0.9)),
        (Adam,     "Adam",  dict(lr=1e-2)),
        (AdamW,    "AdamW", dict(lr=1e-2, weight_decay=1e-4)),
        (GYROAdam, "GYRO",  dict(lr=1e-2)),
    ]

    for cls, name, kwargs in optimizers:
        results[name] = optimize(fn, cls, name, start=start, steps=steps, **kwargs)

    plot(results,
         title=f'Narrow Ravine: f(x) = 100·x₀² + x₁² ({steps} steps)',
         filename='bench_ravine.png',
         steps=steps)


def main():
    print("GYRO Synthetic Optimizer Benchmarks")
    print("=" * 50)
    bench_rosenbrock(steps=5000) # Although 5,000 steps is a lot for this task, it’s just right for demonstrating that the optimizer doesn’t disrupt convergence.
    bench_ravine(steps=3000)
    print("\nDone.")


if __name__ == "__main__":
    main()