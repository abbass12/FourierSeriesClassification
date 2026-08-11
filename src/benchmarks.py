"""Utilities for public univariate time-series benchmark experiments.

The loader supports the conventional UCR/UEA train/test text layout in which
one series is stored per row and the first column is its class label. It does
not download data automatically: users should obtain datasets from the
archive, preserve their license/citation information, and pass a local path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _find_split_file(dataset_dir: Path, dataset_name: str, split: str) -> Path:
    """Find a UCR-style TRAIN or TEST file in common TSV/TXT variants."""
    names = [
        f"{dataset_name}_{split}.tsv",
        f"{dataset_name}_{split}.txt",
        f"{dataset_name}_{split}.TSV",
        f"{dataset_name}_{split}.TXT",
    ]
    roots = [dataset_dir, dataset_dir / dataset_name]
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return candidate
    searched = ", ".join(str(root / name) for root in roots for name in names)
    raise FileNotFoundError(f"Could not locate the {split} split. Searched: {searched}")


def _read_ucr_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read labels plus uniformly sampled time-series values from a text file."""
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] < 2:
        raise ValueError(f"{path} must have a label column and at least one sample column")
    labels = values[:, 0]
    series = values[:, 1:]
    if not np.all(np.isfinite(series)):
        raise ValueError(f"{path} contains non-finite series values")
    return series.astype(np.float32), labels


def encode_labels(train_labels: np.ndarray, test_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[float, int]]:
    """Map archive labels to contiguous integers, rejecting unseen test labels."""
    classes = np.unique(train_labels)
    mapping = {float(label): index for index, label in enumerate(classes)}
    unseen = set(np.unique(test_labels).tolist()) - set(classes.tolist())
    if unseen:
        raise ValueError(f"Test set contains labels absent from the train set: {sorted(unseen)}")
    y_train = np.asarray([mapping[float(label)] for label in train_labels], dtype=np.int64)
    y_test = np.asarray([mapping[float(label)] for label in test_labels], dtype=np.int64)
    return y_train, y_test, mapping


def z_normalize_per_series(X: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Z-normalize each series independently without cross-split leakage."""
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    return ((X - mean) / np.maximum(std, epsilon)).astype(np.float32)


def load_ucr_univariate(
    dataset_dir: str | Path,
    dataset_name: str,
    normalize_per_series: bool = True,
) -> Dict[str, object]:
    """Load one UCR-style univariate classification dataset from local files.

    Parameters
    ----------
    dataset_dir:
        Directory containing either the data files directly or a child folder
        named after ``dataset_name``.
    dataset_name:
        Dataset identifier used in file names, for example ``ECG200``.
    normalize_per_series:
        Whether to apply per-series z-normalization. This is reported in the
        experiment output and can be disabled for datasets where scale carries
        semantic information.
    """
    root = Path(dataset_dir).expanduser().resolve()
    train_file = _find_split_file(root, dataset_name, "TRAIN")
    test_file = _find_split_file(root, dataset_name, "TEST")
    X_train, labels_train = _read_ucr_file(train_file)
    X_test, labels_test = _read_ucr_file(test_file)
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train and test series lengths differ; variable-length datasets are not supported")
    y_train, y_test, label_mapping = encode_labels(labels_train, labels_test)
    if normalize_per_series:
        X_train = z_normalize_per_series(X_train)
        X_test = z_normalize_per_series(X_test)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "dataset_name": dataset_name,
        "n_classes": int(len(label_mapping)),
        "series_length": int(X_train.shape[1]),
        "label_mapping": {str(key): value for key, value in label_mapping.items()},
        "normalize_per_series": normalize_per_series,
        "train_file": str(train_file),
        "test_file": str(test_file),
    }


def stratified_train_validation_split(
    X: np.ndarray,
    y: np.ndarray,
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Split a fixed training partition into reproducible stratified train/validation data."""
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be in (0, 1)")
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for label in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == label))
        n_val = max(1, int(round(len(indices) * validation_ratio)))
        if n_val >= len(indices):
            raise ValueError(f"Class {label} has too few samples for a stratified validation split")
        val_parts.append(indices[:n_val])
        train_parts.append(indices[n_val:])
    train_idx = rng.permutation(np.concatenate(train_parts))
    val_idx = rng.permutation(np.concatenate(val_parts))
    return {
        "X_train": X[train_idx],
        "y_train": y[train_idx],
        "X_val": X[val_idx],
        "y_val": y[val_idx],
    }
