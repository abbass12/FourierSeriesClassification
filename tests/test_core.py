"""Core correctness tests for the Fourier Signal Classification package."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from signals import generate_dataset, generate_grid, box_signal, sine_signal
from fourier import (
    compute_fourier_coefficients,
    detect_jumps,
    extract_jump_features,
    fourier_partial_sum,
    generalized_conjugate_partial_sum,
    signals_to_fourier_features,
    signals_to_fourier_with_jumps,
)


def test_dataset_reproducibility_with_noise():
    """A fixed seed must produce identical synthetic noisy data."""
    first_x, first_y = generate_dataset(4, 64, snr_db=20, seed=17)
    second_x, second_y = generate_dataset(4, 64, snr_db=20, seed=17)
    assert np.array_equal(first_x, second_x)
    assert np.array_equal(first_y, second_y)


def test_fourier_feature_dimensions_for_even_and_odd_mode_counts():
    """Feature arrays preserve declared dimensions for all tested mode counts."""
    x = generate_grid(128)
    signals = np.stack([sine_signal(x), box_signal(x)])
    for modes in [5, 10, 75]:
        coeffs = compute_fourier_coefficients(signals[0], modes)
        features = signals_to_fourier_features(signals, modes)
        assert coeffs.shape == (modes,)
        assert features.shape == (2, 2 * modes)


def test_fourier_reconstruction_has_finite_values():
    """Partial sums must be numerically finite for an ordinary sine signal."""
    x = generate_grid(256)
    signal = sine_signal(x)
    coeffs = compute_fourier_coefficients(signal, 51)
    reconstruction = fourier_partial_sum(coeffs, x)
    assert reconstruction.shape == signal.shape
    assert np.all(np.isfinite(reconstruction))


def test_jump_features_and_edge_function_are_well_formed():
    """The concentration-factor pipeline returns a fixed-size finite feature vector."""
    x = generate_grid(256)
    signal = box_signal(x)
    coeffs = compute_fourier_coefficients(signal, 50)
    edge = generalized_conjugate_partial_sum(coeffs, x, sigma_type="trig")
    locations, values = detect_jumps(edge, x)
    features = extract_jump_features(signal, x, n_modes=50, max_jumps=4)
    assert edge.shape == signal.shape
    assert np.all(np.isfinite(edge))
    assert locations.shape == values.shape
    assert features.shape == (8,)
    assert np.all(np.isfinite(features))


def test_fourier_plus_jump_feature_dimensions():
    """Combined features contain two Fourier components per mode plus jump features."""
    x = generate_grid(128)
    signals = np.stack([sine_signal(x), box_signal(x)])
    features = signals_to_fourier_with_jumps(signals, x, n_modes=30, max_jumps=4)
    assert features.shape == (2, 68)


def test_stratified_split_preserves_per_class_counts():
    """Each split should contain an equal count from every balanced class."""
    X, y = generate_dataset(10, 32, seed=5)
    from signals import train_test_split_signals
    split = train_test_split_signals(X, y, seed=5)
    for partition in ["y_train", "y_val", "y_test"]:
        _, counts = np.unique(split[partition], return_counts=True)
        assert len(counts) == 5
        assert len(set(counts.tolist())) == 1


def test_cnn_baseline_accepts_raw_signal_batch():
    """The convolutional baseline returns one logit vector per raw signal."""
    import torch
    from models import Conv1DSignalClassifier
    model = Conv1DSignalClassifier(n_classes=5)
    logits = model(torch.zeros((3, 128), dtype=torch.float32))
    assert logits.shape == (3, 5)
