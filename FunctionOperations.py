"""
Function Operations Module
==========================

This module provides utility functions for signal processing operations,
including noise addition and jump discontinuity extraction for Fourier
series analysis.

Functions:
    add_noise: Add Gaussian noise to a signal
    extract_jump: Extract jump discontinuity information from a signal
"""

import numpy as np
from typing import Tuple, Dict
from numpy.typing import NDArray


def add_noise(
    y: NDArray[np.floating],
    noise_param: float,
    x: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Add Gaussian noise to a signal.

    Parameters
    ----------
    y : NDArray[np.floating]
        The original signal values.
    noise_param : float
        Standard deviation of the Gaussian noise to add.
    x : NDArray[np.floating]
        The domain points (used to determine noise array length).

    Returns
    -------
    NDArray[np.floating]
        Signal with added Gaussian noise.

    Examples
    --------
    >>> x = np.linspace(-np.pi, np.pi, 100)
    >>> y = np.sin(x)
    >>> noisy_y = add_noise(y, 0.1, x)
    """
    return y + np.random.normal(0, noise_param, len(x))


def extract_jump(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    a: float
) -> NDArray[np.floating]:
    """
    Extract jump discontinuity information from a signal.

    This function identifies and extracts the jump discontinuities at
    positions +a and -a in the signal. It computes the jump magnitude
    by comparing values on either side of the discontinuity.

    Parameters
    ----------
    x : NDArray[np.floating]
        The domain points (spatial coordinates).
    y : NDArray[np.floating]
        The signal values at each domain point.
    a : float
        The position parameter defining where jumps occur (at +a and -a).

    Returns
    -------
    NDArray[np.floating]
        Array of same length as x, with non-zero values only at jump
        locations, representing the jump magnitudes.

    Notes
    -----
    The algorithm:
    1. Finds grid points closest to +a and -a
    2. Computes left and right limits at each jump location
    3. Determines jump magnitude based on which side has larger absolute value
    4. Returns sparse array with jump values at discontinuity locations

    Examples
    --------
    >>> x = np.linspace(-np.pi, np.pi, 1000)
    >>> y = np.where(np.abs(x) < 1, 1.0, 0.0)  # Box function
    >>> jumps = extract_jump(x, y, 1.0)
    """
    fk = np.zeros(len(x))
    h = x[1] - x[0]  # Grid spacing

    # Find grid points closest to jump locations at -a and +a
    jumpsx = (
        x[np.abs(x + a) <= h / 2][0],
        x[np.abs(x - a) <= h / 2][0]
    )

    # Compute left and right limits at each jump location
    lefty: Dict[float, float] = {}
    righty: Dict[float, float] = {}

    lefty[jumpsx[0]] = y[np.abs(x - (jumpsx[0] - h)) < h / 2][0]
    lefty[jumpsx[1]] = y[np.abs(x - (jumpsx[1] - h)) < h / 2][0]
    righty[jumpsx[0]] = y[np.abs(x - (jumpsx[0] + h)) < h / 2][0]
    righty[jumpsx[1]] = y[np.abs(x - (jumpsx[1] + h)) < h / 2][0]

    # Determine jump magnitudes
    jL = righty[jumpsx[0]]
    jR = righty[jumpsx[1]]

    if np.abs(lefty[jumpsx[0]]) > np.abs(righty[jumpsx[0]]):
        jL = lefty[jumpsx[0]] * -1
    if np.abs(lefty[jumpsx[1]]) > np.abs(righty[jumpsx[1]]):
        jR = lefty[jumpsx[1]] * -1

    # Assign jump values to output array
    if len(x[np.abs(x - a) <= h / 2]) == 1:
        fk[np.abs(x + a) <= h / 2] = jL
        fk[np.abs(x - a) <= h / 2] = jR
    else:
        fk[np.abs(x + a) <= h / 2][0] = jL
        fk[np.abs(x - a) <= h / 2][0] = jR

    return fk


# Backwards compatibility aliases (deprecated)
addNoise = add_noise
extractJump = extract_jump
