import time
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import SGD, AdamW

# Add parent directory to path to import gyro
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gyro import GYROAdam


class SimpleCNN(nn.Module):
    """Standard CNN architecture for benchmark evaluation."""
    def __init__(self, input_channels=1, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Adaptive factor for input size (MNIST 28x28 vs CIFAR 32x32)
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
    """Returns train and test dataloaders for specified dataset."""
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        channels = 1
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
        channels = 3
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, channels


def evaluate(model, test_loader, device):
    """Calculates accuracy on the test set."""
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


def train_and_track(optimizer_class, opt_name, dataset_name, epochs=10, **kwargs):
    """Trains model and returns history of loss and accuracy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, channels = get_dataloaders(dataset_name)

    model = SimpleCNN(input_channels=channels).to(device)
    optimizer = optimizer_class(model.parameters(), **kwargs)
    criterion = nn.CrossEntropyLoss()

    history = {'loss': [], 'acc': []}

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

    return history


def plot_results(all_results, dataset_name):
    """Generates and saves performance plots."""
    epochs = range(1, len(next(iter(all_results.values()))['loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    for name, hist in all_results.items():
        plt.plot(epochs, hist['loss'], label=name, marker='o', markersize=4)
    plt.title(f'Training Loss ({dataset_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    for name, hist in all_results.items():
        plt.plot(epochs, hist['acc'], label=name, marker='s', markersize=4)
    plt.title(f'Test Accuracy ({dataset_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'benchmark_{dataset_name.lower()}.png')
    plt.savefig(filename)
    print(f"\nChart saved as {filename}")


def main():
    dataset = 'CIFAR10'  # primary dataset for visualization
    epochs = 15
    
    results = {}
    
    # Baseline: AdamW
    results['AdamW'] = train_and_track(
        AdamW, "AdamW", dataset, epochs=epochs, lr=0.001, weight_decay=0.01
    )
    
    # Target: GYRO
    results['GYRO'] = train_and_track(
        GYROAdam, "GYRO", dataset, epochs=epochs, lr=0.001
    )
    
    # Optional: SGD with Momentum for reference
    results['SGD'] = train_and_track(
        SGD, "SGD", dataset, epochs=epochs, lr=0.01, momentum=0.9
    )

    plot_results(results, dataset)


if __name__ == "__main__":
    main()
