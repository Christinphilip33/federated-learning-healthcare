# pytorchexample/task.py
from collections import defaultdict
from typing import List, Tuple
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

# ---------- Model ----------
class Net(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                       # 32x32 -> 16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                       # 16x16 -> 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------- Datasets / loaders ----------
def _transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2023, 0.1994, 0.2010)),
    ])

def load_datasets() -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """Return raw WISDM wearable accelerometer datasets (no partitioning)."""
    tfm = _transforms()
    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    testset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    return trainset, testset

def load_centralized_dataset(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """One global train/val split for centralized eval on server."""
    trainset, testset = load_datasets()
    # Small validation split from training set
    val_ratio = 0.1
    val_len   = int(len(trainset) * val_ratio)
    train_len = len(trainset) - val_len
    train_subset, _ = random_split(trainset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valloader   = DataLoader(testset,    batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, valloader


# ---------- Dirichlet partitioning ----------
def _labels_from_dataset(ds) -> np.ndarray:
    # Dataset stores labels as targets
    return np.array(ds.targets, dtype=np.int64)

def _dirichlet_partition_indices(
    y: np.ndarray, num_clients: int, alpha: float, seed: int
) -> List[List[int]]:
    """Return index lists per client using class-balanced Dirichlet."""
    rng = np.random.default_rng(seed)
    num_classes = int(y.max()) + 1

    # buckets of indices per class
    cls_idx = [np.where(y == c)[0] for c in range(num_classes)]
    for c in range(num_classes):
        rng.shuffle(cls_idx[c])

    # Dirichlet for each class
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        n = len(cls_idx[c])
        # proportions for this class across clients
        p = rng.dirichlet(alpha * np.ones(num_clients))
        # split the class indices according to p
        splits = (np.cumsum(p) * n).astype(int)[:-1]
        parts = np.split(cls_idx[c], splits)
        for i, part in enumerate(parts):
            client_indices[i].extend(part.tolist())

    # final shuffle per client
    for i in range(num_clients):
        rng.shuffle(client_indices[i])
    return client_indices

def get_client_dataloaders_dirichlet(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    alpha: float,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Return (trainloader, valloader) for one client with Dirichlet split."""
    trainset, testset = load_datasets()
    y = _labels_from_dataset(trainset)
    idx_per_client = _dirichlet_partition_indices(y, num_partitions, alpha, seed)

    my_train_idx = idx_per_client[partition_id]
    client_train = Subset(trainset, my_train_idx)
    # Use test set for validation (simple)
    valloader   = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    trainloader = DataLoader(client_train, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    return trainloader, valloader


# ---------- Train/Test helpers (used by client_app) ----------
def train(model: nn.Module, loader: DataLoader, local_epochs: int, lr: float, device: torch.device) -> float:
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)

@torch.no_grad()
def test(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total
