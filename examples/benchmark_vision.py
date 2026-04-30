import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import SGD, Adam, AdamW

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gyro import CliffordAdam


class SimpleCNN(nn.Module):
    """Standard CNN architecture for benchmark evaluation."""
    def __init__(self, input_channels=1, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7 if input_channels == 1 else 32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_dataloaders(dataset_name, batch_size=64):
    """Returns train and test dataloaders for specified dataset."""
    if dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        channels = 1
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
        channels = 3
    else:
        raise ValueError("Unsupported dataset.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
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


def train_and_evaluate(optimizer_class, opt_name, dataset_name, epochs=3, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, channels = get_dataloaders(dataset_name)
    
    model = SimpleCNN(input_channels=channels).to(device)
    optimizer = optimizer_class(model.parameters(), **kwargs)
    criterion = nn.CrossEntropyLoss()

    print(f"\nEvaluating {opt_name} on {dataset_name}...")
    start_time = time.time()
    
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
        print(f" Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")

    calc_time = time.time() - start_time
    accuracy = evaluate(model, test_loader, device)
    print(f"[{opt_name}] Final Accuracy: {accuracy:.2f}% | Time: {calc_time:.2f}s")
    return accuracy


def main():
    datasets_to_test = ['MNIST', 'CIFAR10']
    
    for ds in datasets_to_test:
        print(f"\n{'='*40}\nDataset: {ds}\n{'='*40}")
        train_and_evaluate(SGD, "SGD", ds, lr=0.01, momentum=0.9)
        train_and_evaluate(Adam, "Adam", ds, lr=0.001)
        train_and_evaluate(AdamW, "AdamW", ds, lr=0.001, weight_decay=0.01)
        train_and_evaluate(CliffordAdam, "GYRO", ds, lr=0.001, theta_base=0.3)


if __name__ == "__main__":
    main()
