from __future__ import annotations

import torch
from torch import nn

INPUT_UNITS = 784
OUTPUT_UNITS = 10


class MLP(nn.Module):
    """A two-layer fully connected network for MNIST digit classification.

    Args:
        hidden_units: Number of neurons in the hidden layer.
    """

    def __init__(self, hidden_units: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(INPUT_UNITS, hidden_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units, OUTPUT_UNITS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (batch, OUTPUT_UNITS)."""
        x = self.relu(self.fc1(x))
        return self.fc2(x)
