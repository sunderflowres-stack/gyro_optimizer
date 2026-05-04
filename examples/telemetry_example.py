import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gyro import GYROAdam


def main():
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(32, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1)
    )
    optimizer = GYROAdam(model.parameters(), lr=1e-3, telemetry=True)
    criterion = nn.MSELoss()

    steps = 500
    history = {'cos': [], 'proj_rate': [], 'norm_g': [], 'norm_m': []}

    for step in range(steps):
        x = torch.randn(64, 32)
        y = torch.randn(64, 1)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        tel = optimizer.get_telemetry()
        history['cos'].append(tel['mean_cos'])
        history['proj_rate'].append(tel['projection_rate'])
        history['norm_g'].append(tel['mean_norm_g'])
        history['norm_m'].append(tel['mean_norm_m'])

        if (step + 1) % 100 == 0:
            print(f"Step {step+1:4d} | cos: {tel['mean_cos']:+.3f} | "
                  f"proj_rate: {tel['projection_rate']:.2f} | "
                  f"‖g‖: {tel['mean_norm_g']:.4f} | "
                  f"‖m‖: {tel['mean_norm_m']:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    steps_range = range(steps)

    axes[0, 0].plot(steps_range, history['cos'])
    axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Mean cosine similarity (g_t, m_t)')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)

    axes[0, 1].plot(steps_range, history['proj_rate'], color='orange')
    axes[0, 1].set_title('Projection rate (fraction of tensors corrected)')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    axes[1, 0].plot(steps_range, history['norm_g'], label='‖g_t‖')
    axes[1, 0].plot(steps_range, history['norm_m'], label='‖m_t‖', alpha=0.7)
    axes[1, 0].set_title('Gradient and momentum norms')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)

    axes[1, 1].axis('off')
    axes[1, 1].text(0.1, 0.5,
        "Projection triggers when\ncos(g_t, m_t) < -theta_base\n\n"
        "High projection rate early in\ntraining = oscillation detected.\n\n"
        "Rate should decrease as\ntraining stabilizes.",
        fontsize=11, verticalalignment='center')

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'telemetry.png')
    plt.savefig(out, dpi=150)
    print(f"\nChart saved as {out}")


if __name__ == '__main__':
    main()