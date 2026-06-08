"""Fine-tune the trained MNIST MLP on synthetic printed digits.

sudokureader's integration tests showed the MNIST-trained model misreads
printed sudoku digits in systematic ways (e.g. a printed "7" without a
crossbar reads as a handwritten "2") -- MNIST is handwritten, but puzzles
use printed fonts. This script reloads the trained model from
output/mnist.npz, then continues training it -- at a low learning rate, on
a mix of synthetic printed digits (src/synthetic_digits.py) and a replay
slice of real MNIST -- so it learns the printed style without forgetting
the handwritten one. It reports accuracy on both MNIST and a held-out
synthetic printed-digit set, then exports to output/mnist_printed.npz.

Run:
    python finetune_printed.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset

sys.path.insert(0, str(Path(__file__).parent / "src"))

from export import export_npz
from mlp import MLP
from synthetic_digits import generate_dataset
from train_mnist import HIDDEN_UNITS, evaluate, load_data, train_epoch

INPUT_PATH = Path(__file__).parent / "output" / "mnist.npz"
OUTPUT_PATH = Path(__file__).parent / "output" / "mnist_printed.npz"

EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 1e-4            # gentle -- preserve the existing MNIST knowledge
PRINTED_SAMPLES_PER_DIGIT = 1500
PRINTED_TEST_FRACTION = 0.2
MNIST_REPLAY_SAMPLES = 20_000   # real-MNIST slice mixed in so fine-tuning can't overwrite it
RANDOM_STATE = 42


def load_trained_model(path: Path) -> MLP:
    """Reconstruct an MLP from weights exported in the (in, out) .npz convention.

    Args:
        path: Path to an .npz with W1/b1/W2/b2 arrays, as written by export_npz.

    Returns:
        An MLP with its state_dict loaded from the file (transposed back to
        PyTorch's (out, in) nn.Linear weight convention).
    """
    data = np.load(path)
    model = MLP(hidden_units=HIDDEN_UNITS)
    model.load_state_dict({
        "fc1.weight": torch.from_numpy(data["W1"].T.copy()),
        "fc1.bias": torch.from_numpy(data["b1"].copy()),
        "fc2.weight": torch.from_numpy(data["W2"].T.copy()),
        "fc2.bias": torch.from_numpy(data["b2"].copy()),
    })
    return model


def load_printed_data() -> tuple[DataLoader, DataLoader]:
    """Render synthetic printed digits and return (train_loader, test_loader)."""
    X, y = generate_dataset(samples_per_digit=PRINTED_SAMPLES_PER_DIGIT, seed=RANDOM_STATE)
    X = X.reshape(len(X), -1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=PRINTED_TEST_FRACTION, stratify=y, random_state=RANDOM_STATE
    )
    train_set = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_set = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    return (
        DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(test_set, batch_size=BATCH_SIZE),
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mnist_train_loader, mnist_test_loader = load_data()
    printed_train_loader, printed_test_loader = load_printed_data()

    rng = np.random.default_rng(RANDOM_STATE)
    replay_idx = rng.choice(len(mnist_train_loader.dataset), size=MNIST_REPLAY_SAMPLES, replace=False)
    replay_set = Subset(mnist_train_loader.dataset, replay_idx.tolist())
    combined_loader = DataLoader(
        ConcatDataset([printed_train_loader.dataset, replay_set]),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = load_trained_model(INPUT_PATH).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Fine-tuning on {len(combined_loader.dataset)} samples "
          f"({len(printed_train_loader.dataset)} printed + {MNIST_REPLAY_SAMPLES} MNIST replay). "
          f"Device: {device}\n")

    mnist_acc = evaluate(model, mnist_test_loader, device)
    printed_acc = evaluate(model, printed_test_loader, device)
    print(f"Before fine-tuning   mnist_accuracy={mnist_acc * 100:.2f}%  "
          f"printed_accuracy={printed_acc * 100:.2f}%")

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, combined_loader, criterion, optimizer, device)
        mnist_acc = evaluate(model, mnist_test_loader, device)
        printed_acc = evaluate(model, printed_test_loader, device)
        print(f"Epoch {epoch}/{EPOCHS}  loss={loss:.4f}  "
              f"mnist_accuracy={mnist_acc * 100:.2f}%  printed_accuracy={printed_acc * 100:.2f}%")

    export_npz(model.cpu(), OUTPUT_PATH)
    print(f"\nModel exported to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
