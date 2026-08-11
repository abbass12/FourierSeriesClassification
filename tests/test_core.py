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


def test_jump_feature_ablation_modes_have_expected_dimensions():
    """Location-only and magnitude-only ablations retain fixed-size vectors."""
    from fourier import extract_jump_features, signals_to_fourier_with_jumps
    x = generate_grid(64)
    signal = np.where(x > 0, 1.0, -1.0)
    locations = extract_jump_features(signal, x, n_modes=16, max_jumps=3,
                                      feature_mode="locations")
    magnitudes = extract_jump_features(signal, x, n_modes=16, max_jumps=3,
                                       feature_mode="magnitudes")
    both = extract_jump_features(signal, x, n_modes=16, max_jumps=3,
                                 feature_mode="both")
    batch = signals_to_fourier_with_jumps(
        np.stack([signal, signal]), x, n_modes=16, max_jumps=3,
        feature_mode="locations"
    )
    assert locations.shape == (3,)
    assert magnitudes.shape == (3,)
    assert both.shape == (6,)
    assert batch.shape == (2, 35)


def test_ucr_loader_and_stratified_validation_split(tmp_path):
    """A local UCR-style dataset loads without test-label leakage."""
    from benchmarks import load_ucr_univariate, stratified_train_validation_split
    train = np.array([
        [10, 1.0, 2.0, 3.0, 4.0],
        [10, 2.0, 3.0, 4.0, 5.0],
        [20, 4.0, 3.0, 2.0, 1.0],
        [20, 5.0, 4.0, 3.0, 2.0],
    ])
    test = np.array([
        [10, 0.0, 1.0, 0.0, 1.0],
        [20, 1.0, 0.0, 1.0, 0.0],
    ])
    np.savetxt(tmp_path / "Tiny_TRAIN.tsv", train, delimiter="\t")
    np.savetxt(tmp_path / "Tiny_TEST.tsv", test, delimiter="\t")
    dataset = load_ucr_univariate(tmp_path, "Tiny")
    split = stratified_train_validation_split(
        dataset["X_train"], dataset["y_train"], validation_ratio=0.5, seed=1
    )
    assert dataset["X_train"].shape == (4, 4)
    assert dataset["X_test"].shape == (2, 4)
    assert dataset["n_classes"] == 2
    assert set(dataset["y_train"]) == {0, 1}
    assert set(split["y_train"]) == {0, 1}
    assert set(split["y_val"]) == {0, 1}


def test_evaluate_model_uses_classifier_output_dimension():
    """Binary classifiers must produce a 2x2, not hard-coded 5x5, confusion matrix."""
    import torch
    from models import SignalClassifier, evaluate_model, prepare_dataloader
    X = np.zeros((4, 8), dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    model = SignalClassifier(input_dim=8, n_classes=2)
    accuracy, confusion = evaluate_model(model, prepare_dataloader(X, y, batch_size=2, shuffle=False))
    assert 0.0 <= accuracy <= 1.0
    assert confusion.shape == (2, 2)
    assert confusion.sum() == len(y)
