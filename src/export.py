from __future__ import annotations

from pathlib import Path

import numpy as np

from mlp import MLP


def export_npz(model: MLP, path: str | Path) -> None:
    """Export an MLP's weights to an .npz consumable by a plain numpy forward pass.

    Saves W1 (in x hidden), b1 (hidden,), W2 (hidden x out), b2 (out,) -- i.e.
    PyTorch's (out, in) weight matrices transposed to the (in, out) convention
    used by `relu(x @ W1 + b1) @ W2 + b2`. This is the format sudokureader's
    predict_fn expects.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    np.savez(
        path,
        W1=state["fc1.weight"].numpy().T,
        b1=state["fc1.bias"].numpy(),
        W2=state["fc2.weight"].numpy().T,
        b2=state["fc2.bias"].numpy(),
    )
