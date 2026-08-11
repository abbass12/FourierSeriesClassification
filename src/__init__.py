"""
Fourier Signal Classification
==============================
A package for classifying signals using Fourier series coefficients
and jump discontinuity features with neural networks.
"""

from .signals import (
    generate_grid, generate_dataset, train_test_split_signals,
    SIGNAL_NAMES, SIGNAL_TYPES, SIGNAL_GENERATORS,
    sine_signal, box_signal, sawtooth_signal,
    exponential_signal, gaussian_signal,
    add_noise, normalize,
)

from .fourier import (
    compute_fourier_coefficients, fourier_partial_sum,
    coeffs_to_real_features, generalized_conjugate_partial_sum,
    detect_jumps, extract_jump_features,
    signals_to_fourier_features, signals_to_fourier_with_jumps,
    trig_sigma, poly_sigma, exp_sigma,
)

from .models import (
    SignalClassifier, Conv1DSignalClassifier, SignalClassifierWithJumps,
    train_model, evaluate_model,
    prepare_dataloader, prepare_dataloader_with_jumps,
)
