"""Train an MLP on the full MNIST dataset (70k samples) using PyTorch.

Downloads the full 70,000-sample MNIST set via OpenML (60k train / 10k test
after a stratified split), trains a two-layer MLP with backpropagation via
Adam, and exports the resulting weights to output/mnist.npz in a format
consumable by a plain numpy forward pass -- see README.md for the exact
contract and how to wire it into sudokureader.

Run:
    python train_mnist.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent / "src"))

from export import export_npz
from mlp import MLP

OUTPUT_PATH = Path(__file__).parent / "output" / "mnist.npz"

HIDDEN_UNITS = 128
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
TEST_SIZE = 10_000
RANDOM_STATE = 42


def load_data() -> tuple[DataLoader, DataLoader]:
    """Download the full 70k-sample MNIST set and return (train_loader, test_loader).

    Images are normalised to [0, 1] floats and kept flattened (784,) to match
    the MLP's input layer.
    """
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    train_set = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_set = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    return (
        DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(test_set, batch_size=BATCH_SIZE),
    )


def train_epoch(
    model: MLP,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return the mean loss over the dataset."""
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: MLP, loader: DataLoader, device: torch.device) -> float:
    """Return classification accuracy over the given loader."""
    model.eval()
    correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(dim=1) == y).sum().item()
    return correct / len(loader.dataset)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data()

    model = MLP(hidden_units=HIDDEN_UNITS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on {len(train_loader.dataset)} samples, "
          f"testing on {len(test_loader.dataset)} samples. Device: {device}\n")

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        accuracy = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{EPOCHS}  loss={loss:.4f}  test_accuracy={accuracy * 100:.2f}%")

    export_npz(model.cpu(), OUTPUT_PATH)
    print(f"\nModel exported to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
