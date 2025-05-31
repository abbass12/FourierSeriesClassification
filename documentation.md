# Fourier Series Classification - Code Structure and Usage Documentation

## Overview

This document provides a detailed explanation of the refactored Fourier Series Classification codebase structure and usage instructions. The code has been reorganized into a proper Python package with clear module separation, consistent naming conventions, and comprehensive documentation.

## Package Structure

The refactored codebase follows a standard Python package structure:

```
fourier_classification/
├── README.md                 # Project overview and basic usage
├── requirements.txt          # Package dependencies
├── setup.py                  # Package installation script
├── examples/                 # Example scripts
│   ├── basic_signals.py      # Basic signal generation and visualization
│   ├── model_training.py     # Model training and evaluation
│   └── validate_refactoring.py # Validation script
└── fourier_classification/   # Main package
    ├── __init__.py           # Package initialization
    ├── signals.py            # Signal generation functions
    ├── fourier.py            # Fourier transformation functions
    ├── operations.py         # Signal operations (noise, jumps)
    ├── models.py             # Neural network models
    ├── visualization.py      # Plotting functions
    └── utils.py              # Utility functions
```

## Module Descriptions

### signals.py

This module provides functions for generating various types of 1D signals:

- `box_signal`: Generates a box/rectangular signal
- `saw_signal`: Generates a sawtooth/ramp signal
- `exp_signal`: Generates an exponential signal
- `sin_signal`: Generates a sinusoidal signal
- `gaussian_signal`: Generates a Gaussian signal
- `generate_signals`: Generates multiple signals of a specified type

Each function can generate either the raw signal or its Fourier coefficients, with options for normalization, jump detection, and noise addition.

### fourier.py

This module provides functions for Fourier series calculations and transformations:

- `dft`: Computes the Discrete Fourier Transform matrix
- `fourier_series`: Computes the Fourier series approximation for given coefficients
- `partial_fourier_sum`: Computes the partial Fourier sum with concentration factors
- `trigonometric_signal`, `polynomial_signal`, `exponential_signal`: Compute different types of concentration factors

### operations.py

This module provides functions for signal operations:

- `add_noise`: Adds Gaussian noise to a signal
- `extract_jump`: Extracts jump information from a signal

### models.py

This module provides neural network models for signal classification:

- `create_feed_forward_model`: Creates a feed-forward neural network model
- `train_model`: Trains a neural network model
- `evaluate_model`: Evaluates a trained model on test data
- `prepare_data_for_model_a`, `prepare_data_for_model_b`, `prepare_data_for_model_c`: Prepare data for different model types

### visualization.py

This module provides functions for visualizing signals and results:

- `plot_signal`: Plots a signal
- `plot_signals_comparison`: Plots multiple signals for comparison
- `plot_fourier_coefficients`: Plots Fourier coefficients
- `plot_signal_and_fourier`: Plots a signal and its Fourier coefficients
- `plot_signal_with_jumps`: Plots a signal and its jump information
- `plot_confusion_matrix`: Plots a confusion matrix
- `plot_training_history`: Plots training history
- Various interactive plotting functions using Plotly

### utils.py

This module provides utility functions:

- `save_model`, `load_model`: Save and load trained models
- `save_data`, `load_data`: Save and load data
- `create_domain`: Creates a domain array for signal generation
- `create_labels`: Creates labels for generated signals
- `normalize_signal`: Normalizes a signal to the range [-1, 1]
- `prepare_dataset`: Prepares a dataset of signals for classification

## Installation

### Requirements

The package requires the following dependencies:
- Python 3.6 or higher
- NumPy
- TensorFlow 2.4 or higher
- Matplotlib
- Plotly
- Pandas
- scikit-learn

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fourier_classification.git
cd fourier_classification
```

2. Install the package and dependencies:
```bash
pip install -e .
```

## Usage Examples

### Basic Signal Generation

```python
import numpy as np
from fourier_classification.signals import box_signal
from fourier_classification.visualization import plot_signal

# Create domain
x = np.linspace(-np.pi, np.pi, 1500)

# Generate a box signal
signal = box_signal(x, a=2, b=5, normalized=True)

# Plot the signal
fig = plot_signal(x, signal, title="Box Signal")
fig.savefig("box_signal.png")
```

### Computing Fourier Coefficients

```python
import numpy as np
from fourier_classification.signals import box_signal
from fourier_classification.fourier import fourier_series
from fourier_classification.visualization import plot_fourier_coefficients

# Create domain
x = np.linspace(-np.pi, np.pi, 1500)

# Compute Fourier coefficients
fourier_coeffs = box_signal(x, a=2, b=5, normalized=True, fourier=True, n_modes=40)

# Plot Fourier coefficients
fig = plot_fourier_coefficients(fourier_coeffs, title="Box Signal Fourier Coefficients")
fig.savefig("box_fourier_coeffs.png")

# Reconstruct signal from coefficients
reconstructed = fourier_series(fourier_coeffs, x, method='precompute')
```

### Training a Classification Model

```python
import numpy as np
from fourier_classification.utils import create_domain, prepare_dataset
from fourier_classification.models import (
    create_feed_forward_model, 
    train_model, 
    evaluate_model,
    prepare_data_for_model_b
)

# Create domain
domain = create_domain(start=-np.pi, end=np.pi, num_points=1500)

# Prepare dataset
signal_types = ['Box', 'Saw', 'Exp', 'Sin', 'Gaus']
signals, labels = prepare_dataset(
    signal_types, 
    num_per_type=100, 
    domain=domain, 
    fourier=True, 
    n_modes=40
)

# Prepare data for Model B (Fourier coefficients)
x_train, x_test, y_train, y_test = prepare_data_for_model_b(signals, labels)

# Create and train model
model = create_feed_forward_model(input_shape=(x_train.shape[1],))
model, history = train_model(model, x_train, y_train, epochs=100)

# Evaluate model
results = evaluate_model(model, x_test, y_test, class_names=signal_types)
print(f"Test accuracy: {results['accuracy']:.4f}")
```

### Working with Jump Information

```python
import numpy as np
from fourier_classification.signals import box_signal
from fourier_classification.visualization import plot_signal_with_jumps

# Create domain
x = np.linspace(-np.pi, np.pi, 1500)

# Generate a box signal with jump information
signal, jumps = box_signal(x, a=2, b=5, normalized=True, jump=True)

# Plot signal with jumps
fig = plot_signal_with_jumps(x, signal, jumps, title="Box Signal with Jumps")
fig.savefig("box_signal_with_jumps.png")
```

## Running the Examples

The package includes several example scripts in the `examples` directory:

1. Basic signal generation and visualization:
```bash
python examples/basic_signals.py
```

2. Model training and evaluation:
```bash
python examples/model_training.py
```

3. Validation of the refactored code:
```bash
python examples/validate_refactoring.py
```

## Validation Results

The refactored code has been validated to ensure it produces the same results as the original implementation. The validation script checks:

1. Signal generation for all signal types
2. Fourier coefficient calculation
3. Fourier series reconstruction
4. Jump detection
5. Noise addition

The validation results are saved in the `validation_results` directory, which contains:
- Signal plots for each signal type
- Fourier coefficient plots
- Reconstruction comparisons
- Jump detection visualizations
- Noisy signal visualizations

## Known Issues and Limitations

1. The sinusoidal signal reconstruction shows NaN values in some cases, which was also present in the original implementation.
2. There are occasional RuntimeWarnings about division by zero in the Fourier coefficient calculation for sinusoidal signals.

## Next Steps for Enhancement

While the current refactoring focuses on code organization and maintainability, future enhancements could include:

1. Implementing alternative neural network architectures (CNNs, RNNs, LSTMs)
2. Adding support for 2D and 3D signals
3. Optimizing performance with vectorized operations
4. Implementing adaptive parameter selection
5. Adding more comprehensive error handling and validation

## Conclusion

The refactored Fourier Series Classification package provides a clean, well-organized implementation of the methodology described in the paper. It maintains all the functionality of the original code while improving readability, maintainability, and usability through proper documentation and examples.
