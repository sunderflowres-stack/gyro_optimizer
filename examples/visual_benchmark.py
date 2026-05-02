import time
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import SGD, Adam, AdamW

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gyro import GYROAdam


class SimpleCNN(nn.Module):
    """Standard CNN architecture for benchmark evaluation."""
    def __init__(self, input_channels=1, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc_input_dim = 32 * 7 * 7 if input_channels == 1 else 32 * 8 * 8
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_dataloaders(dataset_name, batch_size=128):
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset  = datasets.MNIST('./data', train=False, transform=transform)
        channels = 1
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset  = datasets.CIFAR10('./data', train=False, transform=transform)
        channels = 3
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, channels


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100.0 * correct / total


def train_and_track(optimizer_class, opt_name, dataset_name, epochs=15, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, channels = get_dataloaders(dataset_name)

    model     = SimpleCNN(input_channels=channels).to(device)
    optimizer = optimizer_class(model.parameters(), **kwargs)
    criterion = nn.CrossEntropyLoss()

    history    = {'loss': [], 'acc': []}
    start_time = time.time()

    print(f"\nTraining {opt_name} on {dataset_name}...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        accuracy = evaluate(model, test_loader, device)
        history['loss'].append(avg_loss)
        history['acc'].append(accuracy)
        print(f" Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")

    elapsed = time.time() - start_time
    print(f"[{opt_name}] Final Accuracy: {history['acc'][-1]:.2f}% | Time: {elapsed:.1f}s")
    return history


def print_summary(all_results):
    print(f"\n{'='*60}")
    print(f"{'Optimizer':<12} {'Final Acc':>10} {'Best Acc':>10} {'Final Loss':>12}")
    print(f"{'-'*60}")
    for name, hist in all_results.items():
        final_acc  = hist['acc'][-1]
        best_acc   = max(hist['acc'])
        final_loss = hist['loss'][-1]
        print(f"{name:<12} {final_acc:>9.2f}% {best_acc:>9.2f}% {final_loss:>12.4f}")
    print(f"{'='*60}")


def plot_results(all_results, dataset_name, epochs):
    epoch_range = range(1, epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, hist in all_results.items():
        axes[0].plot(epoch_range, hist['loss'], label=name, marker='o', markersize=4)
        axes[1].plot(epoch_range, hist['acc'],  label=name, marker='s', markersize=4)

    axes[0].set_title(f'Training Loss ({dataset_name})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)

    axes[1].set_title(f'Test Accuracy ({dataset_name})')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f'benchmark_{dataset_name.lower()}.png'
    )
    plt.savefig(filename, dpi=150)
    print(f"\nChart saved as {filename}")


def main():
    epochs  = 15
    results = {}

    for dataset in ['MNIST', 'CIFAR10']:
        print(f"\n{'='*40}\nDataset: {dataset}\n{'='*40}")
        results[dataset] = {}
        results[dataset]['SGD']    = train_and_track(SGD,      "SGD",   dataset, epochs=epochs, lr=0.01,  momentum=0.9)
        results[dataset]['Adam']   = train_and_track(Adam,     "Adam",  dataset, epochs=epochs, lr=0.001)
        results[dataset]['AdamW']  = train_and_track(AdamW,    "AdamW", dataset, epochs=epochs, lr=0.001, weight_decay=0.01)
        results[dataset]['GYRO']   = train_and_track(GYROAdam, "GYRO",  dataset, epochs=epochs, lr=0.001)
        print_summary(results[dataset])
        plot_results(results[dataset], dataset, epochs)


if __name__ == "__main__":
    main()
