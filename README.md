# mnist-mlp-pytorch

Trains a two-layer MLP on the **full 70,000-sample MNIST dataset** using PyTorch, and exports
the result to a plain-numpy-compatible `.npz` for use in [sudokureader](https://github.com/reinarsson/sudokureader)'s
digit classifier slot.

This replaces the earlier model trained on only 1,000 samples with a custom pure-Python MLP
(see [multilayerperceptron](https://github.com/reinarsson/multilayerperceptron)), which
generalised poorly to printed digits.

## Architecture

`784 → 128 (ReLU) → 10`, trained with `CrossEntropyLoss` and the Adam optimizer.

## Usage

```bash
pip install -r requirements.txt
python train_mnist.py
```

This downloads the full 70k-sample MNIST set via OpenML (cached by scikit-learn under your
home directory; ~10 epochs takes a few minutes on CPU), trains, prints per-epoch
loss/accuracy, and writes `output/mnist.npz`.

## Export format

`output/mnist.npz` contains four arrays in the `(in, out)` convention so a forward pass is
just two matrix multiplies — no PyTorch dependency needed at inference time:

| Key | Shape | Meaning |
|---|---|---|
| `W1` | `(784, 128)` | input → hidden weights |
| `b1` | `(128,)` | hidden biases |
| `W2` | `(128, 10)` | hidden → output weights |
| `b2` | `(10,)` | output biases |

```python
import numpy as np

data = np.load("output/mnist.npz")
W1, b1, W2, b2 = data["W1"], data["b1"], data["W2"], data["b2"]

def predict_fn(cell):
    x = cell.flatten().astype(np.float32)
    h = np.maximum(0.0, x @ W1 + b1)               # ReLU
    logits = h @ W2 + b2
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()                            # softmax
    digit = int(probs.argmax())
    return digit, float(probs[digit])
```

## Wiring into sudokureader

Drop `output/mnist.npz` into `sudokureader/models/`, then replace the MLP-loading block in
`tests/test_sudoku_reader.py` (and any `predict_fn` you wire up) with the loader and forward
pass shown above — it's a direct swap for the old per-neuron weight format.

## Project structure

```
src/
  mlp.py          # MLP(nn.Module): 784 -> hidden -> 10
  export.py       # export_npz(): state_dict -> W1/b1/W2/b2 .npz
train_mnist.py    # downloads MNIST (70k samples via OpenML), trains, evaluates, exports
tests/
  test_mlp.py     # forward-pass shape and determinism
  test_export.py  # exported arrays match the PyTorch forward pass
output/           # mnist.npz lands here (gitignored)
```
