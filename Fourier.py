"""
Fourier Transform Module
========================

This module provides functions for computing Fourier series representations
and reconstructing signals from Fourier coefficients. It supports multiple
computation methods including precomputed IDFT matrices for efficiency.

Key Features:
    - Discrete Fourier Transform (DFT) computation
    - Inverse Fourier series reconstruction with multiple algorithms
    - Spectral filtering functions (trigonometric, polynomial, exponential)
    - Partial Fourier sum computation for jump analysis

Classes:
    FourierTransformer: Encapsulates IDFT matrix computation and caching

Constants:
    DEFAULT_GRID_SIZE: Default number of grid points (1500)
    SUPPORTED_N_VALUES: List of supported mode counts for precomputation
"""

import numpy as np
from typing import List, Dict, Literal, Union, Optional
from numpy.typing import NDArray


# =============================================================================
# Constants
# =============================================================================

DEFAULT_GRID_SIZE: int = 1500
"""Default number of grid points for signal discretization."""

SUPPORTED_N_VALUES: List[int] = [
    15, 20, 40, 45, 80, 135, 160, 320, 405, 640, 1215, 1280, 1500
]
"""Supported values of N (number of Fourier modes) for precomputed matrices."""

SPECTRAL_TYPES: Dict[str, int] = {'Trig': 1, 'Poly': 2, 'Exp': 3}
"""Mapping of spectral filter type names to numeric codes."""

ALGORITHM_CODES: Dict[str, int] = {'ifft': 1, 'forloop': 2, 'precompute': 3}
"""Mapping of algorithm names to numeric codes."""


# =============================================================================
# Module-level Configuration
# =============================================================================

# Grid configuration
ngrid: int = DEFAULT_GRID_SIZE
xx: NDArray[np.floating] = np.linspace(-np.pi, np.pi, ngrid + 1)
x: NDArray[np.floating] = xx[0:-1]

# IDFT matrix cache (populated on module load)
_idft_matrix_cache: Dict[int, NDArray[np.complexfloating]] = {}


# =============================================================================
# IDFT Matrix Computation
# =============================================================================

def _compute_idft_matrices() -> None:
    """
    Precompute IDFT matrices for all supported N values.

    This function populates the module-level cache with IDFT matrices
    for efficient Fourier series reconstruction. Called automatically
    when the module is imported.

    The IDFT matrix allows reconstruction via matrix multiplication:
        f(x) = IDFT_matrix @ coefficients
    """
    global _idft_matrix_cache

    for n in SUPPORTED_N_VALUES:
        nmodes = n
        idftmat = np.zeros((ngrid, nmodes), dtype=complex)
        nn = np.linspace(-nmodes / 2, nmodes / 2, nmodes + 1)

        for i in range(ngrid):
            for j in range(nmodes):
                idftmat[i][j] = np.e ** (1j * x[i] * nn[j])

        _idft_matrix_cache[n] = idftmat

        # Also compute matrix for n-1 modes
        idftmat_minus1 = np.zeros((ngrid, nmodes - 1), dtype=complex)
        for i in range(ngrid):
            for j in range(nmodes - 1):
                idftmat_minus1[i][j] = np.e ** (1j * x[i] * nn[j])

        _idft_matrix_cache[n - 1] = idftmat_minus1


# =============================================================================
# Core Fourier Functions
# =============================================================================

def dft(
    N: int,
    M: int = 40
) -> NDArray[np.complexfloating]:
    """
    Compute the Discrete Fourier Transform matrix.

    Parameters
    ----------
    N : int
        Number of spatial points in the signal.
    M : int, optional
        Number of frequency modes (default: 40).

    Returns
    -------
    NDArray[np.complexfloating]
        DFT matrix of shape (M, N) for transforming signals to frequency domain.

    Examples
    --------
    >>> dft_matrix = dft(100, 40)
    >>> signal = np.sin(np.linspace(-np.pi, np.pi, 100))
    >>> coefficients = dft_matrix @ signal / 100
    """
    power = np.zeros((M, N), dtype=complex)
    xj = np.linspace(-np.pi, np.pi, N)

    for i in range(M):
        for j in range(N):
            power[i][j] = -1j * (i - M / 2) * xj[j]

    return np.e ** power


def fourier_series(
    cn: NDArray[np.complexfloating],
    X: NDArray[np.floating],
    method: Literal['ifft', 'forloop', 'precompute'] = 'precompute'
) -> NDArray[np.floating]:
    """
    Reconstruct a signal from its Fourier coefficients.

    Parameters
    ----------
    cn : NDArray[np.complexfloating]
        Fourier coefficients (complex array of length N).
    X : NDArray[np.floating]
        Spatial points at which to evaluate the reconstruction.
    method : {'ifft', 'forloop', 'precompute'}, optional
        Algorithm to use for reconstruction:
        - 'precompute': Use precomputed IDFT matrix (fastest for cached sizes)
        - 'forloop': Direct summation (most flexible)
        - 'ifft': FFT-based reconstruction (fast for large arrays)

    Returns
    -------
    NDArray[np.floating]
        Reconstructed real-valued signal at points X.

    Notes
    -----
    The Fourier series reconstruction formula:
        f(x) = sum_{n=-N/2}^{N/2} c_n * exp(i*n*x)

    Examples
    --------
    >>> cn = np.array([0, 0, 1, 0, 0], dtype=complex)  # Single mode
    >>> x = np.linspace(-np.pi, np.pi, 100)
    >>> signal = fourier_series(cn, x, method='forloop')
    """
    N = len(cn)

    if ALGORITHM_CODES[method] == 3:  # precompute
        return np.dot(_idft_matrix_cache[N], cn).real

    elif (ALGORITHM_CODES[method] == 2) or (N % 2 == 0):  # forloop
        fx = []
        for x_val in X:
            result = 0
            for i in range(int(-N / 2), int(N / 2)):
                result += cn[int(i + (N / 2))] * np.e ** (1j * i * x_val)
            fx.append(result.real)
        return np.array(fx)

    else:  # ifft
        Cn = np.zeros(len(X), dtype=complex)
        Cn[0] = cn[int(N / 2)]
        Cn[1:int(N / 2) + 1] = cn[int(N / 2) + 1:N + 1]
        Cn[len(X) - int(N / 2):len(X)] = cn[0:int(N / 2)]
        fx = len(X) * np.fft.ifftshift(np.fft.ifft(Cn))
        return fx.real


# =============================================================================
# Spectral Filter Functions
# =============================================================================

def _exponential_basis(
    x_val: float,
    n: int,
    sign: int = 1
) -> complex:
    """
    Compute complex exponential basis function e^(i*n*x) or e^(-i*n*x).

    Parameters
    ----------
    x_val : float
        Spatial coordinate.
    n : int
        Frequency mode number.
    sign : int, optional
        +1 for positive exponent, -1 for negative (default: 1).

    Returns
    -------
    complex
        Value of the exponential basis function.
    """
    if sign == 1:
        return np.e ** (1j * x_val * n)
    else:
        return np.e ** (-1j * x_val * n)


def trig_sigma(n: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute trigonometric spectral filter (Lanczos sigma factors).

    This filter provides smooth spectral truncation using a sinc-like
    function, reducing Gibbs phenomenon at discontinuities.

    Parameters
    ----------
    n : NDArray[np.floating]
        Array of frequency mode indices.

    Returns
    -------
    NDArray[np.floating]
        Sigma factors for each mode.
    """
    Si_pi = 1.85193705198247  # Integral of sin(t)/t from 0 to pi
    sig = np.pi * np.sin((np.pi * np.abs(n)) / np.max(n)) / Si_pi
    return sig


def poly_sigma(n: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute polynomial spectral filter (linear taper).

    Parameters
    ----------
    n : NDArray[np.floating]
        Array of frequency mode indices.

    Returns
    -------
    NDArray[np.floating]
        Linear taper factors for each mode.
    """
    sig = np.pi * np.abs(n) / np.max(n)
    return sig


def exp_sigma(
    n: NDArray[np.floating],
    N: int
) -> NDArray[np.floating]:
    """
    Compute exponential spectral filter.

    This filter provides smooth truncation with exponential decay
    characteristics.

    Parameters
    ----------
    n : NDArray[np.floating]
        Array of frequency mode indices.
    N : int
        Total number of coefficients.

    Returns
    -------
    NDArray[np.floating]
        Exponential filter factors for each mode.
    """
    alpha = 2
    tau = np.linspace(1 / N, 1 - (1 / N), 1000)
    res = tau[1] - tau[0]
    const = np.pi / (res * sum(np.e ** (1 / (alpha * tau * (tau - 1)))))

    with np.errstate(divide='ignore', invalid='ignore'):
        sig = const * (np.abs(n) / np.max(n)) * np.e ** (
            1 / (alpha * (np.abs(n) / max(n)) * ((np.abs(n) / max(n)) - 1))
        )

    sig[n == 0] = 0
    sig[0] = 0
    sig[-1] = 0
    return sig


def partial_fourier_sum(
    M2: int,
    M1: int,
    x_vals: NDArray[np.floating],
    cn: NDArray[np.complexfloating],
    spectral_type: Literal['Trig', 'Poly', 'Exp'] = 'Trig'
) -> NDArray[np.floating]:
    """
    Compute partial Fourier sum with spectral filtering for jump detection.

    This function computes a filtered partial Fourier sum useful for
    detecting and analyzing jump discontinuities in signals.

    Parameters
    ----------
    M2 : int
        Number of spatial output points.
    M1 : int
        Number of frequency modes to use.
    x_vals : NDArray[np.floating]
        Spatial evaluation points.
    cn : NDArray[np.complexfloating]
        Fourier coefficients.
    spectral_type : {'Trig', 'Poly', 'Exp'}, optional
        Type of spectral filter to apply (default: 'Trig'):
        - 'Trig': Trigonometric (Lanczos) filter
        - 'Poly': Polynomial (linear taper) filter
        - 'Exp': Exponential filter

    Returns
    -------
    NDArray[np.floating]
        Filtered partial Fourier sum values at evaluation points.
    """
    type_code = SPECTRAL_TYPES[spectral_type]
    n = np.linspace(-M1 / 2, M1 / 2, M1)

    # Build evaluation matrix
    D2 = np.zeros((M2 - 1, M1), dtype=complex)
    for p in range(M2 - 1):
        for q in range(M1 - 1):
            D2[p][q] = _exponential_basis(x_vals[p], n[q])

    # Apply spectral filter
    if type_code == 1:
        sig = trig_sigma(n)
    elif type_code == 2:
        sig = poly_sigma(n)
    elif type_code == 3:
        sig = exp_sigma(n, len(cn))

    SN = cn * (1j * np.sign(n) * sig)
    fx = np.dot(D2, SN)

    return fx.real


# =============================================================================
# Backwards Compatibility Aliases (Deprecated)
# =============================================================================

# These aliases maintain backwards compatibility with existing code
idftmatdict = _idft_matrix_cache
SpectDict = SPECTRAL_TYPES
N = SUPPORTED_N_VALUES
algorithm = ALGORITHM_CODES
FourierSeries = fourier_series
et = _exponential_basis
trigSig = trig_sigma
polySig = poly_sigma
expSig = exp_sigma
partialFourierSum = partial_fourier_sum
calculateIdftMat = _compute_idft_matrices


# =============================================================================
# Module Initialization
# =============================================================================

def _initialize_module() -> None:
    """Initialize module by precomputing IDFT matrices."""
    _compute_idft_matrices()


# Auto-initialize on import
if __name__ != '__main__':
    _initialize_module()


def main() -> None:
    """Main entry point when run as script."""
    _compute_idft_matrices()


if __name__ == '__main__':
    main()
