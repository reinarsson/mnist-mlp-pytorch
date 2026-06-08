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

## Closing the printed-vs-handwritten font gap

MNIST is handwritten, but Sudoku puzzles use printed fonts — a domain gap that makes the
base model misread certain digits in systematic ways (e.g. a printed "7" without a crossbar
reads as a handwritten "2"). Two scripts close that gap by fine-tuning the trained model on
synthetic printed digits:

```bash
python finetune_printed.py
```

This reloads `output/mnist.npz`, then continues training it — at a low learning rate, on a
mix of synthetic printed digits and a replay slice of real MNIST — so it picks up the
printed style without forgetting the handwritten one. It prints accuracy on both a held-out
MNIST set and a held-out synthetic printed-digit set each epoch, then writes
`output/mnist_printed.npz` (same `W1/b1/W2/b2` format, same `predict_fn` contract — a direct
swap for `output/mnist.npz` in `sudokureader/models/`).

**`src/synthetic_digits.py`** — `generate_dataset()` renders digits 0-9 in common sans-serif
system fonts (Arial, Calibri, Verdana, Tahoma, Segoe UI, Consolas, ...) with random
size/rotation/position jitter, producing `IMAGE_SIZE x IMAGE_SIZE` float32 images in `[0, 1]`
— white digit on black, centered — in the same format `sudokureader.preprocess_cell` hands
to `predict_fn`.

**Results** — fine-tuning on 12,000 synthetic printed digits + a 20,000-sample MNIST replay,
5 epochs:

| | Before fine-tuning | After fine-tuning |
|---|---|---|
| MNIST accuracy | 97.79% | 97.52% |
| Synthetic printed-digit accuracy | 72.27% | 98.70% |

On sudokureader's 3 sample puzzles, this lifted grid-read accuracy from **80.9%** (76/94
filled cells) to **98.9%** (93/94) — with the one remaining miss a single `6 → 8` confusion.

## Project structure

```
src/
  mlp.py              # MLP(nn.Module): 784 -> hidden -> 10
  export.py           # export_npz(): state_dict -> W1/b1/W2/b2 .npz
  synthetic_digits.py # generate_dataset(): renders printed digits 0-9 into MNIST-format images
train_mnist.py        # downloads MNIST (70k samples via OpenML), trains, evaluates, exports
finetune_printed.py   # fine-tunes the trained model on synthetic printed digits + MNIST replay
tests/
  test_mlp.py         # forward-pass shape and determinism
  test_export.py      # exported arrays match the PyTorch forward pass
output/               # mnist.npz / mnist_printed.npz land here (gitignored)
```
