from typing import Tuple, List
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

# ---------------- Model ----------------
class Net(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                       # 32->16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                       # 16->8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------- Datasets ----------------
def _transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])


def load_datasets():
    """Return raw CIFAR-10 train/test."""
    tfm = _transforms()
    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    return trainset, testset


def load_centralized_dataset(batch_size: int = 32):
    """Centralized test DataLoader (for server evaluation)."""
    _, testset = load_datasets()
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)


# ---------------- Dirichlet partitioning ----------------
def _labels_from_dataset(ds) -> np.ndarray:
    # CIFAR-10 exposes labels as ds.targets
    return np.array(ds.targets, dtype=np.int64)


def _dirichlet_partition_indices(
    y: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
) -> List[List[int]]:
    """Return list of index lists per client using Dirichlet(alpha)."""
    rng = np.random.default_rng(seed)

    num_classes = int(y.max()) + 1
    cls_idx = [np.where(y == c)[0] for c in range(num_classes)]

    # Shuffle indices within each class
    for c in range(num_classes):
        rng.shuffle(cls_idx[c])

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    # For each class, split its indices across clients using Dirichlet proportions
    for c in range(num_classes):
        n = len(cls_idx[c])

        # Proportions for this class across clients
        p = rng.dirichlet(alpha * np.ones(num_clients))

        # Convert proportions into split points
        splits = (np.cumsum(p) * n).astype(int)[:-1]
        parts = np.split(cls_idx[c], splits)

        # Assign class parts to clients
        for i, part in enumerate(parts):
            client_indices[i].extend(part.tolist())

    # Shuffle each client's final list of indices
    for i in range(num_clients):
        rng.shuffle(client_indices[i])

    return client_indices


def get_client_dataloaders_dirichlet(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    alpha: float,
    seed: int = 42,
    val_ratio: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Option A:
    - Partition TRAIN set using Dirichlet(alpha) across clients
    - For the selected client partition, split into (client_train, client_val)
    - Return (trainloader, valloader) that are client-specific
    """
    trainset, _ = load_datasets()

    # Build partitions from the training set
    y = _labels_from_dataset(trainset)
    idx_all = _dirichlet_partition_indices(y, num_partitions, alpha, seed)

    if partition_id < 0 or partition_id >= num_partitions:
        raise ValueError(f"partition_id must be in [0, {num_partitions-1}]")

    my_idx = idx_all[partition_id]
    if partition_id == 0:
        print("\n========== CLIENT 0 DATA DISTRIBUTION ==========")
        unique, counts = np.unique(y[my_idx], return_counts=True)
        print(dict(zip(unique.tolist(), counts.tolist())))
        print("Total samples:", len(my_idx))
        print("===============================================\n")


    if len(my_idx) < 2:
        raise ValueError(
            f"Client {partition_id} has too few samples ({len(my_idx)}). "
            f"Try a larger alpha or fewer partitions."
        )

    client_full = Subset(trainset, my_idx)

    # Split into train/val locally
    n_total = len(client_full)
    n_val = max(1, int(val_ratio * n_total))
    n_train = n_total - n_val

    # Reproducible per-client split
    g = torch.Generator().manual_seed(seed + partition_id)

    client_train, client_val = random_split(client_full, [n_train, n_val], generator=g)

    trainloader = DataLoader(client_train, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(client_val, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, valloader


# ---------------- Train/Test helpers ----------------
def train(
    model: nn.Module,
    loader: DataLoader,
    local_epochs: int,
    lr: float,
    device: torch.device,
) -> float:
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    tot_loss = 0.0
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * x.size(0)

    return tot_loss / len(loader.dataset)


@torch.no_grad()
def test(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return loss_sum / total, correct / total
