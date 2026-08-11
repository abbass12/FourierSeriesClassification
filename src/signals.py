"""
Signal Generation Module
========================
Generates five types of piecewise-smooth signals for classification:
1. Sine (sinusoidal with compact support)
2. Box (rectangular pulse)
3. Sawtooth (linear ramp with compact support)
4. Exponential (exponential decay with compact support)
5. Gaussian (Gaussian pulse with compact support)

All signals are defined on [-pi, pi] with configurable parameters.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List


# Signal type mapping
SIGNAL_TYPES = {
    'sine': 0,
    'box': 1,
    'sawtooth': 2,
    'exponential': 3,
    'gaussian': 4,
}

SIGNAL_NAMES = list(SIGNAL_TYPES.keys())


def generate_grid(n_points: int = 1500) -> np.ndarray:
    """Generate equally spaced grid on [-pi, pi)."""
    x = np.linspace(-np.pi, np.pi, n_points + 1)
    return x[:-1]


def add_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian noise to a signal at a specified SNR (in dB)."""
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise


def normalize(signal: np.ndarray) -> np.ndarray:
    """Normalize signal to have max absolute value of 1."""
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        return signal / max_val
    return signal


# ============================================================
# Signal Generators
# ============================================================

def sine_signal(x: np.ndarray, a: float = 1.5, b: float = 2.0,
                c: float = 1.0) -> np.ndarray:
    """
    Sine signal with compact support on [-a, a].
    f(x) = c * sin(b * x) for |x| < a, 0 otherwise.
    """
    y = np.zeros(len(x))
    mask = np.abs(x) < a
    y[mask] = c * np.sin(b * x[mask])
    return normalize(y)


def box_signal(x: np.ndarray, a: float = 1.5, b: float = 1.0) -> np.ndarray:
    """
    Box (rectangular) signal with compact support on [-a, a].
    f(x) = b for |x| < a, 0 otherwise.
    Has jump discontinuities at x = -a and x = a.
    """
    y = np.zeros(len(x))
    mask = np.abs(x) < a
    y[mask] = b
    return normalize(y)


def sawtooth_signal(x: np.ndarray, a: float = 1.5,
                    b: float = 1.0) -> np.ndarray:
    """
    Sawtooth (linear ramp) signal with compact support on [-a, a].
    f(x) = -b * x for |x| < a, 0 otherwise.
    Has jump discontinuities at x = -a and x = a.
    """
    y = np.zeros(len(x))
    mask = np.abs(x) < a
    y[mask] = -b * x[mask]
    return normalize(y)


def exponential_signal(x: np.ndarray, a: float = 1.5, b: float = 2.0,
                       c: float = -1.0) -> np.ndarray:
    """
    Exponential signal with compact support on [-a, a].
    f(x) = c + exp(-b*x) for |x| < a, 0 otherwise.
    Has jump discontinuities at x = -a and x = a.
    """
    y = np.zeros(len(x))
    mask = np.abs(x) < a
    y[mask] = c + np.exp(-b * x[mask])
    return normalize(y)


def gaussian_signal(x: np.ndarray, a: float = 1.5,
                    b: int = 2) -> np.ndarray:
    """
    Gaussian signal with compact support on [-a, a].
    f(x) = exp(-a * x^(2b)) for |x| < a, 0 otherwise.
    """
    y = np.zeros(len(x))
    mask = np.abs(x) < a
    y[mask] = np.exp(-a * (x[mask] ** (2 * b)))
    y = np.nan_to_num(y)
    return normalize(y)


# Map signal names to generator functions
SIGNAL_GENERATORS = {
    'sine': sine_signal,
    'box': box_signal,
    'sawtooth': sawtooth_signal,
    'exponential': exponential_signal,
    'gaussian': gaussian_signal,
}


# ============================================================
# Dataset Generation
# ============================================================

def generate_random_params(signal_type: str, n_samples: int,
                           seed: Optional[int] = None) -> List[Dict]:
    """Generate random parameter sets for a given signal type."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    params_list = []

    if signal_type == 'sine':
        for _ in range(n_samples):
            params_list.append({
                'a': rng.uniform(np.pi / 4, np.pi / 2),
                'b': rng.choice(np.concatenate([
                    np.linspace(-2 * np.pi, -0.3, 50),
                    np.linspace(0.3, 2 * np.pi, 50)
                ])),
                'c': rng.choice(np.concatenate([
                    np.linspace(-100, -0.1, 50),
                    np.linspace(0.1, 100, 50)
                ])),
            })
    elif signal_type == 'box':
        for _ in range(n_samples):
            params_list.append({
                'a': rng.uniform(np.pi / 4, np.pi / 2),
                'b': rng.choice(np.concatenate([
                    np.linspace(-10, -0.1, 50),
                    np.linspace(0.1, 10, 50)
                ])),
            })
    elif signal_type == 'sawtooth':
        for _ in range(n_samples):
            params_list.append({
                'a': rng.uniform(np.pi / 4, np.pi / 2),
                'b': rng.choice(np.concatenate([
                    np.linspace(-10, -0.1, 50),
                    np.linspace(0.1, 10, 50)
                ])),
            })
    elif signal_type == 'exponential':
        for _ in range(n_samples):
            params_list.append({
                'a': rng.uniform(np.pi / 4, np.pi / 2),
                'b': rng.choice(np.concatenate([
                    np.linspace(-1, -0.1, 50),
                    np.linspace(0.1, 1, 50)
                ])),
                'c': rng.choice(np.concatenate([
                    np.linspace(-3, -1.01, 50),
                    np.linspace(-1.01, 1, 50)
                ])),
            })
    elif signal_type == 'gaussian':
        for _ in range(n_samples):
            params_list.append({
                'a': rng.uniform(np.pi / 4, np.pi / 2),
                'b': rng.integers(1, 10),
            })

    return params_list


def generate_dataset(n_samples_per_type: int = 1000,
                     n_points: int = 1500,
                     snr_db: Optional[float] = None,
                     seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a complete dataset of signals.

    Returns:
        X: array of shape (n_samples_per_type * 5, n_points)
        y: array of shape (n_samples_per_type * 5,) with labels 0-4
    """
    x = generate_grid(n_points)
    all_signals = []
    all_labels = []

    for signal_type, label in SIGNAL_TYPES.items():
        generator = SIGNAL_GENERATORS[signal_type]
        params_list = generate_random_params(signal_type, n_samples_per_type,
                                             seed=seed + label)

        for params in params_list:
            signal = generator(x, **params)
            if snr_db is not None:
                signal = add_noise(signal, snr_db)
            all_signals.append(signal)
            all_labels.append(label)

    X = np.array(all_signals)
    y = np.array(all_labels)

    return X, y


def train_test_split_signals(X: np.ndarray, y: np.ndarray,
                             train_ratio: float = 0.7,
                             val_ratio: float = 0.1,
                             seed: int = 42) -> Dict:
    """Split dataset into train/val/test sets."""
    rng = np.random.default_rng(seed)
    n = len(y)
    indices = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        'X_train': X[train_idx], 'y_train': y[train_idx],
        'X_val': X[val_idx], 'y_val': y[val_idx],
        'X_test': X[test_idx], 'y_test': y[test_idx],
    }
