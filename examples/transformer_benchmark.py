import os
import urllib.request
import torch
import torch.nn as nn
from torch.optim import AdamW
from gyro import GYROAdam

class TinyGPT(nn.Module):
    def __init__(self, vocab_size=256, d_model=128, n_heads=4, n_layers=4, seq_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=512,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)
        self.seq_len = seq_len

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos(pos)
        x = self.transformer(x)
        return self.head(x)

def get_text_data(seq_len=128, batch_size=64):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, 'shakespeare.txt')

    if not os.path.exists(path):
        print("Downloading Shakespeare dataset...")
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
            path
        )
        print("Done.")

    with open(path, 'r') as f:
        text = f.read()

    data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)
    num_batches = (len(data) - seq_len) // batch_size
    batches = []
    for i in range(num_batches):
        idx = i * batch_size
        x = torch.stack([data[idx + j: idx + j + seq_len] for j in range(batch_size)])
        y = torch.stack([data[idx + j + 1: idx + j + seq_len + 1] for j in range(batch_size)])
        batches.append((x, y))
    return batches

def train(optimizer_class, name, batches, epochs=10, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TinyGPT().to(device)
    optimizer = optimizer_class(model.parameters(), **kwargs)
    criterion = nn.CrossEntropyLoss()

    # Split into train/val to detect overfitting
    split = int(len(batches) * 0.9)
    train_batches = batches[:split]
    val_batches = batches[split:]

    print(f"\nTraining {name} on {'GPU' if device.type == 'cuda' else 'CPU'}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_batches:  # use all training data, not just 200
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, 256), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_batches:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits.view(-1, 256), y.view(-1)).item()

        print(f" Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_batches):.4f} | Val Loss: {val_loss/len(val_batches):.4f}")

if __name__ == '__main__':
    batches = get_text_data()
    train(AdamW,    'AdamW', batches, epochs=10, lr=1e-3, weight_decay=0.01)
    train(GYROAdam, 'GYRO',  batches, epochs=10, lr=1e-3, theta_base=0.3)
