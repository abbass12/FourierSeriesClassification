"""
Function Definitions Module
===========================

This module defines five signal generator functions used for Fourier series
classification experiments. Each function can generate signals in both
spatial and frequency (Fourier) domains.

Signal Types:
    - Box: Rectangular pulse function
    - Saw: Sawtooth wave function
    - Exp: Exponential function with compact support
    - Sinu: Sinusoidal function with compact support
    - Gaus: Gaussian function with compact support

Each function supports:
    - Spatial domain generation
    - Fourier coefficient computation (analytical or DFT)
    - Normalization options
    - Noise injection
    - Jump discontinuity extraction for spectral analysis
"""

import numpy as np
from typing import Tuple, Union, Literal, Optional
from numpy.typing import NDArray

from Fourier import (
    dft,
    partial_fourier_sum,
    partialFourierSum  # Backwards compatibility
)
from FunctionOperations import (
    add_noise,
    extract_jump,
    addNoise,      # Backwards compatibility
    extractJump    # Backwards compatibility
)


# Type aliases for return types
SignalArray = NDArray[np.floating]
FourierCoeffs = NDArray[np.complexfloating]
SignalWithJump = Tuple[SignalArray, SignalArray]
FourierWithJump = Tuple[FourierCoeffs, SignalArray]


def box(
    x: SignalArray,
    a: float = 2,
    b: float = 5,
    normalized: bool = True,
    jump: bool = False,
    type: Literal['Trig', 'Poly', 'Exp'] = 'Trig',
    noise: bool = False,
    noiseParameter: float = 0.1,
    fourier: bool = False,
    N: int = 40
) -> Union[SignalArray, FourierCoeffs, SignalWithJump, FourierWithJump]:
    """
    Generate a box (rectangular pulse) function or its Fourier coefficients.

    The box function is defined as:
        f(x) = b  if |x| < a
        f(x) = 0  otherwise

    Parameters
    ----------
    x : SignalArray
        Spatial domain points for evaluation.
    a : float, optional
        Half-width of the box (default: 2).
    b : float, optional
        Height/amplitude of the box (default: 5).
    normalized : bool, optional
        If True, normalize output to [-1, 1] range (default: True).
    jump : bool, optional
        If True, also return jump discontinuity data (default: False).
    type : {'Trig', 'Poly', 'Exp'}, optional
        Spectral filter type for jump detection (default: 'Trig').
    noise : bool, optional
        If True, add Gaussian noise to output (default: False).
    noiseParameter : float, optional
        Standard deviation of noise (default: 0.1).
    fourier : bool, optional
        If True, return Fourier coefficients instead of spatial signal (default: False).
    N : int, optional
        Number of Fourier modes (default: 40).

    Returns
    -------
    Union[SignalArray, FourierCoeffs, SignalWithJump, FourierWithJump]
        Depending on parameters:
        - Signal values at x (default)
        - Fourier coefficients (if fourier=True)
        - Tuple of (signal, jumps) (if jump=True)
        - Tuple of (coefficients, filtered_jumps) (if fourier=True and jump=True)

    Notes
    -----
    Analytical Fourier coefficients for box function:
        c_0 = 2ab / (2*pi)
        c_k = 2b*sin(ak) / (k * 2*pi)  for k != 0

    Examples
    --------
    >>> x = np.linspace(-np.pi, np.pi, 1000)
    >>> signal = box(x, a=1.5, b=1.0)
    >>> coeffs = box(x, a=1.5, b=1.0, fourier=True, N=40)
    """
    if fourier:
        k = np.linspace(int(-N / 2), int(N / 2), N + 1)
        fkhat = np.zeros(len(k))
        fkhat[k == 0] = 2 * a * b
        fkhat[k != 0] = (2 * b * np.sin(a * k[k != 0])) / k[k != 0]
        y = fkhat / (2 * np.pi)

        if normalized:
            if jump:
                y = y[0:-1] / np.max(np.abs(box(x, a, b, normalized=False)))
                return y, partialFourierSum(1500, len(y), x, y, type)
            return y[0:-1] / np.max(np.abs(box(x, a, b, normalized=False)))
        return y

    y = np.zeros(len(x))
    y[np.abs(x) < a] = b

    if noise:
        if normalized:
            y = y / np.max(abs(y))
        return addNoise(y, noiseParameter, x)

    if normalized:
        if jump:
            y = y / np.max(abs(y))
            fk = extractJump(x, y, a)
            return y[0:-1], fk[0:-1]
        return y / np.max(abs(y))

    return y


def saw(
    x: SignalArray,
    a: float = 2,
    b: float = 5,
    normalized: bool = True,
    jump: bool = False,
    type: Literal['Trig', 'Poly', 'Exp'] = 'Trig',
    noise: bool = False,
    noiseParameter: float = 0.1,
    fourier: bool = False,
    N: int = 40
) -> Union[SignalArray, FourierCoeffs, SignalWithJump, FourierWithJump]:
    """
    Generate a sawtooth wave function or its Fourier coefficients.

    The sawtooth function is defined as:
        f(x) = -b*x  if |x| < a
        f(x) = 0     otherwise

    Parameters
    ----------
    x : SignalArray
        Spatial domain points for evaluation.
    a : float, optional
        Half-width of support (default: 2).
    b : float, optional
        Slope magnitude (default: 5).
    normalized : bool, optional
        If True, normalize output to [-1, 1] range (default: True).
    jump : bool, optional
        If True, also return jump discontinuity data (default: False).
    type : {'Trig', 'Poly', 'Exp'}, optional
        Spectral filter type for jump detection (default: 'Trig').
    noise : bool, optional
        If True, add Gaussian noise to output (default: False).
    noiseParameter : float, optional
        Standard deviation of noise (default: 0.1).
    fourier : bool, optional
        If True, return Fourier coefficients instead of spatial signal (default: False).
    N : int, optional
        Number of Fourier modes (default: 40).

    Returns
    -------
    Union[SignalArray, FourierCoeffs, SignalWithJump, FourierWithJump]
        Signal values, Fourier coefficients, or tuple with jump data.

    Notes
    -----
    Analytical Fourier coefficients for sawtooth:
        c_0 = 0
        c_k = 2ib*(sin(ak) - ak*cos(ak)) / k^2  for k != 0
    """
    if fourier:
        k = np.linspace(int(-N / 2), int(N / 2), N + 1)
        fkhat = np.zeros(len(k)).astype(complex)
        fkhat[k == 0] = 0
        fkhat[k != 0] = (
            2 * 1j * b * (np.sin(a * k[k != 0]) - a * k[k != 0] * np.cos(a * k[k != 0]))
        ) / (k[k != 0] ** 2)
        y = fkhat / (2 * np.pi)

        if normalized:
            if jump:
                y = y[0:-1] / np.max(np.abs(saw(x, a, b, normalized=False)))
                return y, partialFourierSum(1500, len(y), x, y, type)
            return y[0:-1] / np.max(np.abs(saw(x, a, b, normalized=False)))
        return y

    y = np.zeros(len(x))
    y[np.abs(x) < a] = -b * x[np.abs(x) < a]

    if noise:
        if normalized:
            y = y / np.max(abs(y))
        return addNoise(y, noiseParameter, x)

    if normalized:
        if jump:
            y = y / np.max(abs(y))
            fk = extractJump(x, y, a)
            return y[0:-1], fk[0:-1]
        return y / np.max(abs(y))

    return y


def exp(
    x: SignalArray,
    a: float = 2,
    b: float = 2,
    c: float = -1,
    normalized: bool = True,
    jump: bool = False,
    type: Literal['Trig', 'Poly', 'Exp'] = 'Trig',
    noise: bool = False,
    noiseParameter: float = 0.1,
    fourier: bool = False,
    N: int = 40
) -> Union[SignalArray, FourierCoeffs, SignalWithJump, FourierWithJump]:
    """
    Generate an exponential function with compact support or its Fourier coefficients.

    The exponential function is defined as:
        f(x) = c + e^(-b*x)  if |x| < a
        f(x) = 0             otherwise

    Parameters
    ----------
    x : SignalArray
        Spatial domain points for evaluation.
    a : float, optional
        Half-width of support (default: 2).
    b : float, optional
        Exponential decay rate (default: 2).
    c : float, optional
        Vertical offset (default: -1).
    normalized : bool, optional
        If True, normalize output to [-1, 1] range (default: True).
    jump : bool, optional
        If True, also return jump discontinuity data (default: False).
    type : {'Trig', 'Poly', 'Exp'}, optional
        Spectral filter type for jump detection (default: 'Trig').
    noise : bool, optional
        If True, add Gaussian noise to output (default: False).
    noiseParameter : float, optional
        Standard deviation of noise (default: 0.1).
    fourier : bool, optional
        If True, return Fourier coefficients instead of spatial signal (default: False).
    N : int, optional
        Number of Fourier modes (default: 40).

    Returns
    -------
    Union[SignalArray, FourierCoeffs, SignalWithJump, FourierWithJump]
        Signal values, Fourier coefficients, or tuple with jump data.

    Notes
    -----
    Analytical Fourier coefficients use hyperbolic sinh functions.
    """
    if fourier:
        k = np.linspace(int(-N / 2), int(N / 2), N + 1).astype(complex)
        fkhat = np.zeros(len(k)).astype(complex)
        fkhat[k == 0] = (2 * ((a * b * c) + np.sinh(a * b))) / b
        fkhat[k != 0] = (
            (2 * c * np.sin(a * k[k != 0])) / k[k != 0]
            + (2 * np.sinh(a * (b + (1j * k[k != 0])))) / (b + (1j * k[k != 0]))
        )
        y = fkhat / (2 * np.pi)

        if normalized:
            if jump:
                y = y[0:-1] / np.max(np.abs(exp(x, a, b, c, normalized=False)))
                return y, partialFourierSum(1500, len(y), x, y, type)
            return y[0:-1] / np.max(np.abs(exp(x, a, b, c, normalized=False)))
        return y

    y = np.zeros(len(x))
    y[np.abs(x) < a] = c + np.e ** (-b * x[np.abs(x) < a])

    if noise:
        if normalized:
            y = y / np.max(abs(y))
        return addNoise(y, noiseParameter, x)

    if normalized:
        if jump:
            y = y / np.max(abs(y))
            fk = extractJump(x, y, a)
            return y[0:-1], fk[0:-1]
        return y / np.max(abs(y))

    return y


def sinu(
    x: SignalArray,
    a: float = 2,
    b: float = 2,
    c: float = -1,
    normalized: bool = True,
    jump: bool = False,
    type: Literal['Trig', 'Poly', 'Exp'] = 'Trig',
    noise: bool = False,
    noiseParameter: float = 0.1,
    fourier: bool = False,
    N: int = 40
) -> Union[SignalArray, FourierCoeffs, SignalWithJump, FourierWithJump]:
    """
    Generate a sinusoidal function with compact support or its Fourier coefficients.

    The sinusoidal function is defined as:
        f(x) = c*sin(b*x)  if |x| < a
        f(x) = 0           otherwise

    Parameters
    ----------
    x : SignalArray
        Spatial domain points for evaluation.
    a : float, optional
        Half-width of support (default: 2).
    b : float, optional
        Angular frequency (default: 2).
    c : float, optional
        Amplitude (default: -1).
    normalized : bool, optional
        If True, normalize output to [-1, 1] range (default: True).
    jump : bool, optional
        If True, also return jump discontinuity data (default: False).
    type : {'Trig', 'Poly', 'Exp'}, optional
        Spectral filter type for jump detection (default: 'Trig').
    noise : bool, optional
        If True, add Gaussian noise to output (default: False).
    noiseParameter : float, optional
        Standard deviation of noise (default: 0.1).
    fourier : bool, optional
        If True, return Fourier coefficients instead of spatial signal (default: False).
    N : int, optional
        Number of Fourier modes (default: 40).

    Returns
    -------
    Union[SignalArray, FourierCoeffs, SignalWithJump, FourierWithJump]
        Signal values, Fourier coefficients, or tuple with jump data.
    """
    if fourier:
        k = np.linspace(int(-N / 2), int(N / 2), N + 1).astype(complex)
        fkhat = np.zeros(len(k)).astype(complex)
        fkhat[k == b] = (1j * c * np.sin(2 * a * b)) / (2 * b) - 1j * a * c
        fkhat[k == -b] = (1 / 2) * 1j * c * ((2 * a) - ((np.sin(2 * a * b)) / b))
        fkhat[k != b] = (
            2 * 1j * c * (
                (b * np.cos(a * b) * np.sin(a * k[k != b]))
                - (k[k != b] * np.sin(a * b) * np.cos(a * k[k != b]))
            )
        ) / ((b ** 2) - (k[k != b] ** 2))
        y = fkhat / (2 * np.pi)

        if normalized:
            if jump:
                y = y[0:-1] / np.max(np.abs(sinu(x, a, b, c, normalized=False)))
                return y, partialFourierSum(1500, len(y), x, y, type)
            return y[0:-1] / np.max(np.abs(sinu(x, a, b, c, normalized=False)))
        return y

    y = np.zeros(len(x))
    y[np.abs(x) < a] = c * np.sin(b * x[np.abs(x) < a])

    if noise:
        if normalized:
            y = y / np.max(abs(y))
        return addNoise(y, noiseParameter, x)

    if normalized:
        if jump:
            y = y / np.max(abs(y))
            fk = extractJump(x, y, a)
            return y[0:-1], fk[0:-1]
        return y / np.max(abs(y))

    return y


def gaus(
    x: SignalArray,
    a: float = 2,
    b: float = 2,
    normalized: bool = True,
    jump: bool = False,
    type: Literal['Trig', 'Poly', 'Exp'] = 'Trig',
    noise: bool = False,
    noiseParameter: float = 0.1,
    fourier: bool = False,
    N: int = 40,
    M: int = 40
) -> Union[SignalArray, FourierCoeffs, SignalWithJump, FourierWithJump]:
    """
    Generate a Gaussian-like function with compact support or its Fourier coefficients.

    The Gaussian function is defined as:
        f(x) = e^(-a * x^(2b))  if |x| < a
        f(x) = 0                otherwise

    Note: This is not a standard Gaussian but a generalized super-Gaussian
    with adjustable shape via the exponent 2b.

    Parameters
    ----------
    x : SignalArray
        Spatial domain points for evaluation.
    a : float, optional
        Half-width of support and decay parameter (default: 2).
    b : float, optional
        Shape exponent (b=1 gives standard Gaussian decay) (default: 2).
    normalized : bool, optional
        If True, normalize output to [-1, 1] range (default: True).
    jump : bool, optional
        If True, also return jump discontinuity data (default: False).
    type : {'Trig', 'Poly', 'Exp'}, optional
        Spectral filter type for jump detection (default: 'Trig').
    noise : bool, optional
        If True, add Gaussian noise to output (default: False).
    noiseParameter : float, optional
        Standard deviation of noise (default: 0.1).
    fourier : bool, optional
        If True, return Fourier coefficients via DFT (default: False).
    N : int, optional
        Number of spatial points for DFT (default: 40).
    M : int, optional
        Number of Fourier modes (default: 40).

    Returns
    -------
    Union[SignalArray, FourierCoeffs, SignalWithJump, FourierWithJump]
        Signal values, Fourier coefficients, or tuple with jump data.

    Notes
    -----
    Unlike other functions, gaus() uses numerical DFT for Fourier coefficients
    since no simple analytical formula exists for this generalized form.
    """
    if fourier:
        y = (np.dot(dft(N, M), gaus(x, a, b))) / N

        if normalized:
            if jump:
                y = y[0:-1] / np.max(np.abs(gaus(x, a, b, normalized=False)))
                return y[0:-1], partialFourierSum(1500, len(y), x, y, type)
            return y[0:-1] / np.max(np.abs(gaus(x, a, b, normalized=False)))
        return y

    y = np.zeros(len(x))
    y[np.abs(x) < a] = np.e ** (-a * (x[np.abs(x) < a] ** (2 * b)))
    y = np.nan_to_num(y)

    if noise:
        if normalized:
            y = y / np.max(abs(y))
        return addNoise(y, noiseParameter, x)

    if normalized:
        if jump:
            y = y / np.max(abs(y))
            fk = extractJump(x, y, a)
            return y[0:-1], fk[0:-1]
        return y / np.max(abs(y))

    return y


# =============================================================================
# Signal Type Mapping
# =============================================================================

SIGNAL_FUNCTIONS = {
    'Box': box,
    'Saw': saw,
    'Exp': exp,
    'Sin': sinu,
    'Gaus': gaus
}
"""Mapping of signal type names to generator functions."""

SIGNAL_TYPES = list(SIGNAL_FUNCTIONS.keys())
"""List of available signal type names."""
