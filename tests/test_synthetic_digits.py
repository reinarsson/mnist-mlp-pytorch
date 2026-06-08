from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from synthetic_digits import FONT_PATHS, FONT_SIZES, IMAGE_SIZE, _render_digit, generate_dataset


def test_fonts_discovered():
    """At least one usable font must be found on every supported platform."""
    assert len(FONT_PATHS) > 0


def test_render_digit_returns_centered_normalised_glyph():
    rng = np.random.default_rng(0)
    img = _render_digit(7, FONT_PATHS[0], FONT_SIZES[0], rng)
    assert img is not None
    assert img.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert img.dtype == np.float32
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert img.max() > 0.0  # the glyph actually drew some ink


def test_generate_dataset_shape_dtype_and_range():
    X, y = generate_dataset(samples_per_digit=2, seed=0)
    assert X.shape == (20, IMAGE_SIZE, IMAGE_SIZE)
    assert X.dtype == np.float32
    assert y.shape == (20,)
    assert y.dtype == np.int64
    assert X.min() >= 0.0 and X.max() <= 1.0


def test_generate_dataset_is_balanced_across_digits():
    _, y = generate_dataset(samples_per_digit=3, seed=1)
    counts = np.bincount(y, minlength=10)
    assert counts.tolist() == [3] * 10


def test_generate_dataset_is_deterministic_given_seed():
    X1, y1 = generate_dataset(samples_per_digit=2, seed=42)
    X2, y2 = generate_dataset(samples_per_digit=2, seed=42)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_generate_dataset_raises_when_no_fonts_available(monkeypatch):
    monkeypatch.setattr("synthetic_digits.FONT_PATHS", ())
    with pytest.raises(FileNotFoundError):
        generate_dataset(samples_per_digit=1)
