"""Minimal, reproducible PyTorch training script (MNIST classifier).
pip install torch torchvision
python train.py --epochs 5 --batch-size 128
Features: seeding, GPU/CPU auto-selection, validation each epoch,
best-checkpoint saving and a JSON log of metrics.
"""
import argparse
import json
import random
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def set_seed(seed):
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class Net(nn.Module):
def __init__(self):
super().__init__()
self.net = nn.Sequential(
nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.3),
nn.Linear(128, 10),
)
def forward(self, x):
return self.net(x)

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
model.eval()

total_loss, correct, n = 0.0, 0, 0
for x, y in loader:
x, y = x.to(device), y.to(device)
out = model(x)
total_loss += loss_fn(out, y).item() * len(y)
correct += (out.argmax(1) == y).sum().item()
n += len(y)
return total_loss / n, correct / n

def main():
p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--batch-size", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", type=Path, default=Path("runs/latest"))
args = p.parse_args()
set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.out.mkdir(parents=True, exist_ok=True)
tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
full = datasets.MNIST("data", train=True, download=True, transform=tfm)
train_set, val_set = random_split(full, [55000, 5000], generator=torch.Generator().manual_seed(args.seed))
train_dl = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
val_dl = DataLoader(val_set, 512, num_workers=2)
model = Net().to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()
history, best = [], 0.0
for epoch in range(1, args.epochs + 1):
model.train()
start, running = time.time(), 0.0
for x, y in train_dl:
x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
opt.zero_grad(set_to_none=True)
loss = loss_fn(model(x), y)
loss.backward()
opt.step()
running += loss.item() * len(y)
val_loss, val_acc = evaluate(model, val_dl, loss_fn, device)
row = {"epoch": epoch, "train_loss": running / len(train_set), "val_loss": val_loss,

"val_acc": val_acc, "seconds": round(time.time() - start, 1)}
history.append(row)
print(row)
if val_acc > best:
best = val_acc
torch.save(model.state_dict(), args.out / "best.pt")
(args.out / "metrics.json").write_text(json.dumps(history, indent=2))
print(f"Best validation accuracy: {best:.4f}")

if __name__ == "__main__":
main()
