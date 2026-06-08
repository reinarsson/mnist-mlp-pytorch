from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from export import export_npz
from mlp import INPUT_UNITS, OUTPUT_UNITS, MLP

HIDDEN_UNITS = 16


@pytest.fixture
def exported(tmp_path):
    model = MLP(hidden_units=HIDDEN_UNITS)
    path = tmp_path / "model.npz"
    export_npz(model, path)
    return model, path


def test_export_creates_expected_arrays(exported):
    """The .npz must contain W1/b1/W2/b2 with shapes matching the (in, out) convention."""
    _, path = exported
    data = np.load(path)
    assert set(data.keys()) == {"W1", "b1", "W2", "b2"}
    assert data["W1"].shape == (INPUT_UNITS, HIDDEN_UNITS)
    assert data["b1"].shape == (HIDDEN_UNITS,)
    assert data["W2"].shape == (HIDDEN_UNITS, OUTPUT_UNITS)
    assert data["b2"].shape == (OUTPUT_UNITS,)


def test_exported_weights_match_forward_pass(exported):
    """A numpy forward pass through the exported arrays must match PyTorch's output."""
    model, path = exported
    data = np.load(path)
    W1, b1, W2, b2 = data["W1"], data["b1"], data["W2"], data["b2"]

    x = np.random.rand(INPUT_UNITS).astype(np.float32)
    h = np.maximum(0.0, x @ W1 + b1)
    expected_logits = h @ W2 + b2

    model.eval()
    with torch.no_grad():
        actual_logits = model(torch.from_numpy(x).unsqueeze(0)).squeeze(0).numpy()

    np.testing.assert_allclose(expected_logits, actual_logits, rtol=1e-5, atol=1e-5)
