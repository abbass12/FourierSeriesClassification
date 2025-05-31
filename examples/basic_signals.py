"""
Example script demonstrating basic signal generation and visualization.

This example shows how to generate different types of signals,
compute their Fourier coefficients, and visualize the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from fourier_classification.signals import box_signal, saw_signal, exp_signal, sin_signal, gaussian_signal
from fourier_classification.fourier import fourier_series
from fourier_classification.visualization import plot_signal_and_fourier, plot_signals_comparison

# Create domain
x = np.linspace(-np.pi, np.pi, 1500)

# Generate different types of signals
box = box_signal(x, a=2, b=5, normalized=True)
saw = saw_signal(x, a=2, b=5, normalized=True)
exp = exp_signal(x, a=2, b=2, c=-1, normalized=True)
sin = sin_signal(x, a=2, b=2, c=-1, normalized=True)
gaus = gaussian_signal(x, a=2, b=2, normalized=True)

# Compute Fourier coefficients
box_fourier = box_signal(x, a=2, b=5, normalized=True, fourier=True, n_modes=40)
saw_fourier = saw_signal(x, a=2, b=5, normalized=True, fourier=True, n_modes=40)
exp_fourier = exp_signal(x, a=2, b=2, c=-1, normalized=True, fourier=True, n_modes=40)
sin_fourier = sin_signal(x, a=2, b=2, c=-1, normalized=True, fourier=True, n_modes=40)
gaus_fourier = gaussian_signal(x, a=2, b=2, normalized=True, fourier=True, n_modes=40)

# Plot signals comparison
signals_dict = {
    'Box': box,
    'Sawtooth': saw,
    'Exponential': exp,
    'Sinusoidal': sin,
    'Gaussian': gaus
}

fig = plot_signals_comparison(x, signals_dict, title="Signal Types Comparison")
plt.savefig("signal_comparison.png")
plt.close(fig)

# Plot individual signals with their Fourier coefficients
for name, signal, coeffs in [
    ('Box', box, box_fourier),
    ('Sawtooth', saw, saw_fourier),
    ('Exponential', exp, exp_fourier),
    ('Sinusoidal', sin, sin_fourier),
    ('Gaussian', gaus, gaus_fourier)
]:
    fig = plot_signal_and_fourier(x, signal, coeffs, title=f"{name} Signal")
    plt.savefig(f"{name.lower()}_signal.png")
    plt.close(fig)

print("All plots have been saved.")
