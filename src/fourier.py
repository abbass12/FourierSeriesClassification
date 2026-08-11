"""
Fourier Analysis Module
=======================
Implements:
1. Discrete Fourier Transform (DFT) and inverse
2. Fourier series partial sums
3. Concentration factors for edge detection (Gelb-Tadmor method)
4. Jump discontinuity detection and quantification
"""

import numpy as np
from typing import Tuple, Optional
from scipy.special import sici


def compute_fourier_coefficients(signal: np.ndarray,
                                 n_modes: Optional[int] = None) -> np.ndarray:
    """
    Compute Fourier coefficients using FFT.

    Args:
        signal: 1D signal array of length M
        n_modes: Number of Fourier modes to keep (default: M//2)

    Returns:
        Complex Fourier coefficients of length exactly n_modes
    """
    M = len(signal)
    if n_modes is None:
        n_modes = M // 2

    # Use FFT and shift to center zero-frequency
    fft_vals = np.fft.fftshift(np.fft.fft(signal)) / M

    # Extract n_modes centered around zero frequency
    center = M // 2
    start = center - n_modes // 2
    coeffs = fft_vals[start:start + n_modes]

    # Ensure exact length (handle odd/even edge cases)
    if len(coeffs) < n_modes:
        coeffs = np.pad(coeffs, (0, n_modes - len(coeffs)))
    elif len(coeffs) > n_modes:
        coeffs = coeffs[:n_modes]

    return coeffs


def fourier_partial_sum(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Reconstruct signal from Fourier coefficients using partial sum.

    S_N f(x) = sum_{|k| <= N} f_hat_k * e^{ikx}

    Args:
        coeffs: Complex Fourier coefficients
        x: Grid points for reconstruction

    Returns:
        Reconstructed signal values at grid points
    """
    N = len(coeffs)
    half_N = N // 2
    if N % 2 == 0:
        k = np.arange(-half_N, half_N)
    else:
        k = np.arange(-half_N, half_N + 1)
    k = k[:N]  # ensure same length as coeffs

    # Vectorized computation: outer product
    # result[i] = sum_j coeffs[j] * exp(i * k[j] * x[i])
    phase = np.outer(x, k)  # shape (len(x), N)
    reconstruction = np.real(np.dot(np.exp(1j * phase), coeffs))

    return reconstruction


def coeffs_to_real_features(coeffs: np.ndarray) -> np.ndarray:
    """
    Convert complex Fourier coefficients to real-valued feature vector.
    Interleaves real and imaginary parts.

    Args:
        coeffs: Complex array of shape (n_modes,)

    Returns:
        Real array of shape (2 * n_modes,)
    """
    return np.concatenate([np.real(coeffs), np.imag(coeffs)])


# ============================================================
# Concentration Factors for Edge Detection (Gelb-Tadmor)
# ============================================================

def trig_sigma(k: np.ndarray, N: int) -> np.ndarray:
    """
    Trigonometric (Fourier) concentration factor.
    sigma_F^alpha(eta) = -pi / Si(alpha) * sin(alpha * eta)

    Reference: Gelb & Tadmor, ACHA 1999, SIAM J. Numer. Anal. 2000
    """
    Si_pi, _ = sici(np.pi)  # Si(pi) ≈ 1.8519
    eta = np.abs(k) / N
    sigma = np.pi * np.sin(np.pi * eta) / Si_pi
    return sigma


def poly_sigma(k: np.ndarray, N: int, p: int = 1) -> np.ndarray:
    """
    Polynomial concentration factor.
    sigma_p(eta) = p * pi * eta^p
    """
    eta = np.abs(k) / N
    sigma = p * np.pi * (eta ** p)
    return sigma


def exp_sigma(k: np.ndarray, N: int, alpha: float = 2.0) -> np.ndarray:
    """
    Exponential concentration factor.
    sigma(eta) = C * eta * exp(1 / (alpha * eta * (eta - 1)))

    Vanishes at eta=0 and eta=1, smooth on (0,1).
    """
    eta = np.abs(k) / N
    sigma = np.zeros_like(eta, dtype=float)

    # Avoid division by zero at boundaries
    interior = (eta > 1e-10) & (eta < 1 - 1e-10)
    if np.any(interior):
        arg = 1.0 / (alpha * eta[interior] * (eta[interior] - 1))
        sigma[interior] = eta[interior] * np.exp(arg)

    # Normalize so that the integral condition is approximately satisfied
    if np.sum(np.abs(sigma)) > 0:
        # Approximate normalization using discrete sum
        dx = 1.0 / N
        integral = dx * np.sum(sigma[k > 0] / (np.abs(k[k > 0]) / N + 1e-10))
        if integral > 0:
            sigma = sigma * np.pi / integral

    return sigma


def generalized_conjugate_partial_sum(
        coeffs: np.ndarray,
        x: np.ndarray,
        sigma_type: str = 'trig') -> np.ndarray:
    """
    Compute the Generalized Conjugate Partial Fourier Sum for edge detection.

    S_N^sigma[f](x) = i * sum_{k=-N}^{N} f_hat(k) * sgn(k) * sigma(|k|/N) * e^{ikx}

    This sum concentrates at jump discontinuities of f.

    Args:
        coeffs: Fourier coefficients
        x: Grid points
        sigma_type: Type of concentration factor ('trig', 'poly', 'exp')

    Returns:
        Edge function values at grid points (peaks at discontinuities)
    """
    N = len(coeffs)
    # Create k array matching coeffs length exactly
    half_N = N // 2
    if N % 2 == 0:
        k = np.arange(-half_N, half_N)
    else:
        k = np.arange(-half_N, half_N + 1)

    # Ensure k has same length as coeffs
    k = k[:N]

    # Compute concentration factors
    N_max = max(np.max(np.abs(k)), 1)
    if sigma_type == 'trig':
        sigma = trig_sigma(k, N_max)
    elif sigma_type == 'poly':
        sigma = poly_sigma(k, N_max)
    elif sigma_type == 'exp':
        sigma = exp_sigma(k, N_max)
    else:
        raise ValueError(f"Unknown sigma type: {sigma_type}")

    # Sign function
    sgn = np.sign(k).astype(float)

    # Compute S_N^sigma[f](x) = i * sum_k f_hat_k * sgn(k) * sigma_k * e^{ikx}
    modified_coeffs = 1j * coeffs * sgn * sigma

    # Reconstruct
    phase = np.outer(x, k)
    edge_function = np.real(np.dot(np.exp(1j * phase), modified_coeffs))

    return edge_function


def detect_jumps(edge_function: np.ndarray, x: np.ndarray,
                 threshold: float = 0.3,
                 min_distance: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect jump locations and magnitudes from the edge function.

    Args:
        edge_function: Output of generalized_conjugate_partial_sum
        x: Grid points
        threshold: Minimum magnitude to consider as a jump
        min_distance: Minimum distance between detected jumps (in grid points)

    Returns:
        jump_locations: x-coordinates of detected jumps
        jump_values: Magnitudes of the jumps
    """
    # Find peaks in absolute value of edge function
    abs_edge = np.abs(edge_function)
    max_val = np.max(abs_edge)

    if max_val < 1e-10:
        return np.array([]), np.array([])

    # Normalize and threshold
    normalized = abs_edge / max_val
    candidates = np.where(normalized > threshold)[0]

    if len(candidates) == 0:
        return np.array([]), np.array([])

    # Group nearby candidates and take the peak of each group
    jump_indices = []
    jump_vals = []

    i = 0
    while i < len(candidates):
        # Find the extent of this group
        group_start = i
        while (i < len(candidates) - 1 and
               candidates[i + 1] - candidates[i] <= min_distance):
            i += 1
        group_end = i

        # Find peak within group
        group = candidates[group_start:group_end + 1]
        peak_idx = group[np.argmax(abs_edge[group])]
        jump_indices.append(peak_idx)
        jump_vals.append(edge_function[peak_idx])

        i += 1

    jump_locations = x[jump_indices]
    jump_values = np.array(jump_vals)

    return jump_locations, jump_values


def extract_jump_features(signal: np.ndarray, x: np.ndarray,
                          n_modes: int = 50,
                          sigma_type: str = 'trig',
                          max_jumps: int = 4,
                          feature_mode: str = 'both') -> np.ndarray:
    """
    Extract jump discontinuity features from a signal.

    Returns a fixed-length feature vector of jump locations, signed
    magnitudes, or both, zero-padded to ``max_jumps``. The default
    ``both`` preserves the historical location-then-magnitude layout.

    Args:
        signal: Input signal
        x: Grid points
        n_modes: Number of Fourier modes for edge detection
        sigma_type: Concentration factor type
        max_jumps: Maximum number of jumps to detect.
        feature_mode: One of ``'locations'``, ``'magnitudes'``, or ``'both'``.

    Returns:
        Feature vector of length ``max_jumps`` for a single descriptor type,
        or ``2 * max_jumps`` for ``'both'``.
    """
    if feature_mode not in {'locations', 'magnitudes', 'both'}:
        raise ValueError("feature_mode must be 'locations', 'magnitudes', or 'both'")
    # Compute Fourier coefficients
    coeffs = compute_fourier_coefficients(signal, n_modes)

    # Compute edge function
    edge_fn = generalized_conjugate_partial_sum(coeffs, x, sigma_type)

    # Detect jumps
    locations, values = detect_jumps(edge_fn, x)

    # Create fixed-length descriptors (zero-padded) after ordering peaks by
    # absolute inferred jump magnitude.
    ordered_locations = np.zeros(max_jumps)
    ordered_values = np.zeros(max_jumps)
    n_detected = min(len(locations), max_jumps)
    if n_detected > 0:
        sort_idx = np.argsort(-np.abs(values))[:n_detected]
        ordered_locations[:n_detected] = locations[sort_idx]
        ordered_values[:n_detected] = values[sort_idx]

    if feature_mode == 'locations':
        return ordered_locations
    if feature_mode == 'magnitudes':
        return ordered_values
    return np.concatenate([ordered_locations, ordered_values])


# ============================================================
# Batch Processing for Dataset
# ============================================================

def signals_to_fourier_features(X: np.ndarray,
                                n_modes: int = 50) -> np.ndarray:
    """
    Convert a batch of signals to Fourier coefficient features.

    Args:
        X: Signal array of shape (n_samples, n_points)
        n_modes: Number of Fourier modes

    Returns:
        Feature array of shape (n_samples, 2 * n_modes)
    """
    n_samples = X.shape[0]
    features = np.zeros((n_samples, 2 * n_modes))

    for i in range(n_samples):
        coeffs = compute_fourier_coefficients(X[i], n_modes)
        features[i] = coeffs_to_real_features(coeffs)

    return features


def signals_to_fourier_with_jumps(X: np.ndarray, x: np.ndarray,
                                  n_modes: int = 50,
                                  sigma_type: str = 'trig',
                                  max_jumps: int = 4,
                                  feature_mode: str = 'both') -> np.ndarray:
    """
    Convert a batch of signals to Fourier coefficients + jump features.

    Args:
        X: Signal array of shape (n_samples, n_points)
        x: Grid points
        n_modes: Number of Fourier modes
        sigma_type: Concentration factor type
        max_jumps: Maximum number of jumps.
        feature_mode: Descriptor ablation setting passed to
            :func:`extract_jump_features`.

    Returns:
        Feature array of shape ``(n_samples, 2 * n_modes + jump_dim)``.
    """
    if feature_mode not in {'locations', 'magnitudes', 'both'}:
        raise ValueError("feature_mode must be 'locations', 'magnitudes', or 'both'")
    n_samples = X.shape[0]
    fourier_dim = 2 * n_modes
    jump_dim = 2 * max_jumps if feature_mode == 'both' else max_jumps
    features = np.zeros((n_samples, fourier_dim + jump_dim))

    for i in range(n_samples):
        # Fourier features
        coeffs = compute_fourier_coefficients(X[i], n_modes)
        features[i, :fourier_dim] = coeffs_to_real_features(coeffs)

        # Jump features
        features[i, fourier_dim:] = extract_jump_features(
            X[i], x, n_modes=n_modes, sigma_type=sigma_type,
            max_jumps=max_jumps, feature_mode=feature_mode
        )

    return features
