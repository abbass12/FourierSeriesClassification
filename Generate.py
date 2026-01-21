"""
Signal Generation Module
========================

This module provides a high-level interface for generating batches of
synthetic signals for machine learning classification experiments.

The Generate function creates multiple signal instances with randomized
parameters, supporting both spatial domain and Fourier domain representations.

Features:
    - Batch generation of signals with random parameters
    - Support for 5 signal types: Box, Saw, Exp, Sin, Gaus
    - Fourier coefficient computation with jump discontinuity analysis
    - Noise injection for robustness testing
    - Configurable normalization options

Example:
    >>> from Generate import Generate
    >>> from Fourier import x
    >>> signals, a_params, b_params = Generate('Box', x, Amount=100, fourier=True, N=40)
"""

import numpy as np
from typing import List, Tuple, Union, Literal, Optional
from numpy.typing import NDArray

from FunctionDefinitions import box, saw, exp, sinu, gaus
from Fourier import FourierSeries, x as default_x


# =============================================================================
# Constants
# =============================================================================

SIGNAL_TYPE_CODES = {
    'Box': 1,
    'Saw': 2,
    'Exp': 3,
    'Sin': 4,
    'Gaus': 5
}
"""Mapping of signal type names to numeric codes."""

SignalType = Literal['Box', 'Saw', 'Exp', 'Sin', 'Gaus']
"""Type alias for valid signal type names."""


# =============================================================================
# Parameter Generation Helpers
# =============================================================================

def _generate_random_params_box_saw(
    length: int
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Generate random parameters for Box and Sawtooth signals.

    Parameters
    ----------
    length : int
        Number of parameter pairs to generate.

    Returns
    -------
    Tuple[NDArray, NDArray]
        Tuple of (a, b) parameter arrays with random permutation.
    """
    a = np.linspace(0.10, 2.90, length)
    b = np.append(
        np.linspace(-100, -0.01, int(length / 2)),
        np.linspace(0.01, 100, int(length / 2))
    )
    return np.random.permutation(a), np.random.permutation(b)


def _generate_random_params_exp(
    length: int
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Generate random parameters for Exponential signals.

    Parameters
    ----------
    length : int
        Number of parameter sets to generate.

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray]
        Tuple of (a, b, c) parameter arrays with random permutation.
    """
    a = np.linspace(np.pi / 4, np.pi / 2, length)
    b = np.append(
        np.linspace(-1, -0.1, int(length / 2)),
        np.linspace(0.1, 1, int(length / 2))
    )
    c = np.append(
        np.linspace(-3, -1.01, int(length / 2)),
        np.linspace(-1.01, 1, int(length / 2))
    )
    return (
        np.random.permutation(a),
        np.random.permutation(b),
        np.random.permutation(c)
    )


def _generate_random_params_sin(
    length: int
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Generate random parameters for Sinusoidal signals.

    Parameters
    ----------
    length : int
        Number of parameter sets to generate.

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray]
        Tuple of (a, b, c) parameter arrays with random permutation.
    """
    a = np.linspace(np.pi / 4, np.pi / 2, length)
    b = np.append(
        np.linspace(-2 * np.pi, -0.3, int(length / 2)),
        np.linspace(0.3, 2 * np.pi, int(length / 2))
    )
    c = np.append(
        np.linspace(-100, -0.1, int(length / 2)),
        np.linspace(0.1, 100, int(length / 2))
    )
    return (
        np.random.permutation(a),
        np.random.permutation(b),
        np.random.permutation(c)
    )


def _generate_random_params_gaus(
    length: int
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Generate random parameters for Gaussian signals.

    Parameters
    ----------
    length : int
        Number of parameter pairs to generate.

    Returns
    -------
    Tuple[NDArray, NDArray]
        Tuple of (a, b) parameter arrays with random permutation.
    """
    a = np.linspace(np.pi / 4, np.pi / 2, length)
    b = np.linspace(1, 10, length)
    return np.random.permutation(a), np.random.permutation(b)


# =============================================================================
# Main Generation Function
# =============================================================================

def Generate(
    signal: SignalType,
    x: NDArray[np.floating],
    Amount: int,
    normalized: bool = True,
    fourier: bool = False,
    jump: bool = False,
    N: int = 40,
    type: Literal['Trig', 'Poly', 'Exp'] = 'Trig',
    Noise: bool = False,
    noiseParameter: float = 0.1,
    method: Literal['ifft', 'forloop', 'precompute'] = 'precompute',
    randomParam: bool = True,
    a: float = 0,
    b: float = 0,
    c: float = 0
) -> Union[
    Tuple[List, NDArray, NDArray],
    Tuple[List, NDArray, NDArray, NDArray]
]:
    """
    Generate a batch of signals with randomized parameters.

    This function creates multiple instances of a specified signal type,
    each with different random parameters. It supports generation in both
    spatial and Fourier domains, with optional noise and jump analysis.

    Parameters
    ----------
    signal : {'Box', 'Saw', 'Exp', 'Sin', 'Gaus'}
        Type of signal to generate.
    x : NDArray[np.floating]
        Spatial domain points for signal evaluation.
    Amount : int
        Number of signal instances to generate.
    normalized : bool, optional
        If True, normalize signals to [-1, 1] range (default: True).
    fourier : bool, optional
        If True, generate Fourier domain signals (default: False).
    jump : bool, optional
        If True, include jump discontinuity analysis (default: False).
    N : int, optional
        Number of Fourier modes (default: 40).
    type : {'Trig', 'Poly', 'Exp'}, optional
        Spectral filter type for jump detection (default: 'Trig').
    Noise : bool, optional
        If True, add Gaussian noise to signals (default: False).
    noiseParameter : float, optional
        Standard deviation of noise (default: 0.1).
    method : {'ifft', 'forloop', 'precompute'}, optional
        Fourier series reconstruction algorithm (default: 'precompute').
    randomParam : bool, optional
        If True, use random parameters; otherwise use provided a, b, c (default: True).
    a : float, optional
        Fixed 'a' parameter when randomParam=False (default: 0).
    b : float, optional
        Fixed 'b' parameter when randomParam=False (default: 0).
    c : float, optional
        Fixed 'c' parameter when randomParam=False (default: 0).

    Returns
    -------
    Union[Tuple[List, NDArray, NDArray], Tuple[List, NDArray, NDArray, NDArray]]
        Tuple containing:
        - List of generated signals (spatial or Fourier domain)
        - Array of 'a' parameters used
        - Array of 'b' parameters used
        - Array of 'c' parameters used (only for Exp and Sin signals)

    Examples
    --------
    Generate 100 box signals in spatial domain:

    >>> x = np.linspace(-np.pi, np.pi, 1500)[:-1]
    >>> signals, a_vals, b_vals = Generate('Box', x, Amount=100)

    Generate 50 sinusoidal signals in Fourier domain:

    >>> signals, a_vals, b_vals, c_vals = Generate(
    ...     'Sin', x, Amount=50, fourier=True, N=40
    ... )

    Generate signals with jump discontinuity data:

    >>> signals, a_vals, b_vals = Generate(
    ...     'Box', x, Amount=100, fourier=True, jump=True, N=40
    ... )
    """
    signalOutput: List = []
    length = Amount + 2
    signal_code = SIGNAL_TYPE_CODES[signal]

    print(signal_code)

    # -------------------------------------------------------------------------
    # Box Signal Generation
    # -------------------------------------------------------------------------
    if signal_code == 1:
        if randomParam:
            a, b = _generate_random_params_box_saw(length)

        if fourier:
            for i in range(Amount):
                if jump:
                    result = box(
                        x, a[i], b[i],
                        normalized=normalized, jump=True,
                        type=type, fourier=True, N=N
                    )
                    cn = result[0]
                    Jump = result[1]
                    signalOutput.append([
                        FourierSeries(cn, x, method=method)[0:-1],
                        Jump
                    ])
                else:
                    cn = box(
                        x, a[i], b[i],
                        normalized=normalized, fourier=True, N=N
                    )
                    signalOutput.append(FourierSeries(cn, x, method=method))
            return [signalOutput, a[0:Amount], b[0:Amount]]

        elif Noise:
            for i in range(Amount):
                F = box(
                    x, a[i], b[i],
                    noise=True, normalized=normalized,
                    noiseParameter=noiseParameter
                )
                signalOutput.append(F)
            return [signalOutput, a[0:Amount], b[0:Amount]]

        else:
            for i in range(Amount):
                F = box(x, a[i], b[i], jump=jump, normalized=normalized)
                signalOutput.append(F)
            return [signalOutput, a[0:Amount], b[0:Amount]]

    # -------------------------------------------------------------------------
    # Sawtooth Signal Generation
    # -------------------------------------------------------------------------
    elif signal_code == 2:
        if randomParam:
            a = np.linspace(np.pi / 4, np.pi / 2, length)
            b = np.append(
                np.linspace(-100, -0.01, int(length / 2)),
                np.linspace(0.01, 100, int(length / 2))
            )
            a = np.random.permutation(a)
            b = np.random.permutation(b)

        if fourier:
            for i in range(Amount):
                if jump:
                    result = saw(
                        x, a[i], b[i],
                        normalized=normalized, jump=True,
                        type=type, fourier=True, N=N
                    )
                    cn = result[0]
                    Jump = result[1]
                    signalOutput.append([
                        FourierSeries(cn, x, method=method)[0:-1],
                        Jump
                    ])
                else:
                    cn = saw(
                        x, a[i], b[i],
                        normalized=normalized, fourier=True, N=N
                    )
                    signalOutput.append(FourierSeries(cn, x, method=method))
            return [signalOutput, a[0:Amount], b[0:Amount]]

        elif Noise:
            for i in range(Amount):
                F = saw(
                    x, a[i], b[i],
                    noise=True, normalized=normalized,
                    noiseParameter=noiseParameter
                )
                signalOutput.append(F)
            return [signalOutput, a[0:Amount], b[0:Amount]]

        else:
            for i in range(Amount):
                F = saw(x, a[i], b[i], jump=jump, normalized=normalized)
                signalOutput.append(F)
            return [signalOutput, a[0:Amount], b[0:Amount]]

    # -------------------------------------------------------------------------
    # Exponential Signal Generation
    # -------------------------------------------------------------------------
    elif signal_code == 3:
        if randomParam:
            a, b, c = _generate_random_params_exp(length)

        if fourier:
            for i in range(Amount):
                if jump:
                    result = exp(
                        x, a[i], b[i], c[i],
                        normalized=normalized, jump=True,
                        type=type, fourier=True, N=N
                    )
                    cn = result[0]
                    Jump = result[1]
                    signalOutput.append([
                        FourierSeries(cn, x, method=method)[0:-1],
                        Jump
                    ])
                else:
                    cn = exp(
                        x, a[i], b[i], c[i],
                        normalized=normalized, fourier=True, N=N
                    )
                    signalOutput.append(FourierSeries(cn, x, method=method))
            return [signalOutput, a[0:Amount], b[0:Amount], c[0:Amount]]

        elif Noise:
            for i in range(Amount):
                F = exp(
                    x, a[i], b[i], c[i],
                    noise=True, normalized=normalized,
                    noiseParameter=noiseParameter
                )
                signalOutput.append(F)
            return [signalOutput, a[0:Amount], b[0:Amount], c[0:Amount]]

        else:
            for i in range(Amount):
                F = exp(
                    x, a[i], b[i], c[i],
                    jump=jump, normalized=normalized
                )
                signalOutput.append(F)
            return [signalOutput, a[0:Amount], b[0:Amount], c[0:Amount]]

    # -------------------------------------------------------------------------
    # Sinusoidal Signal Generation
    # -------------------------------------------------------------------------
    elif signal_code == 4:
        if randomParam:
            a, b, c = _generate_random_params_sin(length)

        if fourier:
            for i in range(Amount):
                if jump:
                    result = sinu(
                        x, a[i], b[i], c[i],
                        normalized=normalized, jump=True,
                        type=type, fourier=True, N=N
                    )
                    cn = result[0]
                    Jump = result[1]
                    signalOutput.append([
                        FourierSeries(cn, x, method=method)[0:-1],
                        Jump
                    ])
                else:
                    cn = sinu(
                        x, a[i], b[i], c[i],
                        normalized=normalized, fourier=True, N=N
                    )
                    signalOutput.append(FourierSeries(cn, x, method=method))
            return [signalOutput, a[0:Amount], b[0:Amount], c[0:Amount]]

        elif Noise:
            for i in range(Amount):
                F = sinu(
                    x, a[i], b[i], c[i],
                    noise=True, normalized=normalized,
                    noiseParameter=noiseParameter
                )
                signalOutput.append(F)
            return [signalOutput, a[0:Amount], b[0:Amount], c[0:Amount]]

        else:
            for i in range(Amount):
                F = sinu(
                    x, a[i], b[i], c[i],
                    jump=jump, normalized=normalized
                )
                signalOutput.append(F)
            return [signalOutput, a[0:Amount], b[0:Amount], c[0:Amount]]

    # -------------------------------------------------------------------------
    # Gaussian Signal Generation
    # -------------------------------------------------------------------------
    elif signal_code == 5:
        if randomParam:
            a, b = _generate_random_params_gaus(length)

        if fourier:
            for i in range(Amount):
                if jump:
                    result = gaus(
                        x, a[i], int(b[i]),
                        normalized=normalized, jump=True,
                        type=type, fourier=True, N=len(x), M=N
                    )
                    cn = result[0]
                    Jump = result[1]
                    signalOutput.append([
                        FourierSeries(cn, x, method=method)[0:-1],
                        Jump
                    ])
                else:
                    cn = gaus(
                        x, a[i], int(b[i]),
                        normalized=normalized, fourier=True,
                        N=len(x), M=N
                    )
                    signalOutput.append(FourierSeries(cn, x, method=method))
            return [signalOutput, a[0:Amount], b[0:Amount]]

        elif Noise:
            for i in range(Amount):
                F = gaus(
                    x, a[i], int(b[i]),
                    noise=True, normalized=normalized,
                    noiseParameter=noiseParameter
                )
                signalOutput.append(F)
            return [signalOutput, a[0:Amount], b[0:Amount]]

        else:
            for i in range(Amount):
                F = gaus(
                    x, a[i], int(b[i]),
                    jump=jump, normalized=normalized
                )
                signalOutput.append(F)
            return [signalOutput, a[0:Amount], b[0:Amount]]


# Backwards compatibility alias
SignalsDict = SIGNAL_TYPE_CODES
