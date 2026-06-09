"""Export a trained .npz model to TensorFlow Lite (LiteRT) format.

Loads W1/b1/W2/b2 weights from an .npz file produced by export_npz(),
reconstructs the MLP as a Keras model, and converts it to a .tflite
flatbuffer for on-device inference (Android/iOS via LiteRT).

Run:
    pip install tensorflow
    python scripts/export_tflite.py                         # uses defaults
    python scripts/export_tflite.py --input output/mnist_printed.npz --output output/mnist_printed.tflite
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

INPUT_PATH = Path(__file__).parent.parent / "output" / "mnist_printed.npz"
OUTPUT_PATH = Path(__file__).parent.parent / "output" / "mnist_printed.tflite"

INPUT_UNITS = 784


def export_tflite(input_path: Path, output_path: Path) -> None:
    """Load .npz weights and write a TFLite flatbuffer.

    Args:
        input_path: Path to a .npz with W1/b1/W2/b2 in (in, out) convention.
        output_path: Destination path for the .tflite file.
    """
    import tensorflow as tf  # imported here — TF is an optional dependency

    data = np.load(input_path)
    W1, b1, W2, b2 = data["W1"], data["b1"], data["W2"], data["b2"]
    hidden_units = W1.shape[1]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(INPUT_UNITS,)),
        tf.keras.layers.Dense(hidden_units, activation="relu"),
        tf.keras.layers.Dense(W2.shape[1]),
    ])
    model.layers[0].set_weights([W1, b1])
    model.layers[1].set_weights([W2, b2])

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    print(f"Exported {len(tflite_model):,} bytes to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=INPUT_PATH, help="Source .npz file")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Destination .tflite file")
    args = parser.parse_args()
    export_tflite(args.input, args.output)


if __name__ == "__main__":
    main()
