from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlp import INPUT_UNITS, OUTPUT_UNITS, MLP


def test_forward_pass_shape():
    """Output logits must have shape (batch, OUTPUT_UNITS)."""
    model = MLP(hidden_units=32)
    x = torch.rand(4, INPUT_UNITS)
    out = model(x)
    assert out.shape == (4, OUTPUT_UNITS)


def test_forward_pass_deterministic_in_eval_mode():
    """Repeated forward passes on the same input must be identical in eval mode."""
    model = MLP(hidden_units=32)
    model.eval()
    x = torch.rand(1, INPUT_UNITS)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.equal(out1, out2)


def test_hidden_layer_size_is_configurable():
    """The hidden layer width must follow the constructor argument."""
    model = MLP(hidden_units=16)
    assert model.fc1.out_features == 16
    assert model.fc2.in_features == 16
