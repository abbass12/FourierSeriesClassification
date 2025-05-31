# Fourier Series Classification

A package for classifying 1D signals using Fourier Series and Machine Learning.

## Overview

The `Fourier-Series-Python` package offers a robust toolkit for analyzing and classifying 1-dimensional signals using the power of Fourier series combined with machine learning techniques. This library is designed for researchers, engineers, and data scientists looking to extract meaningful features from signals and build predictive models.

At its core, the package leverages the Fourier series to decompose a periodic signal $f(t)$ (with period $T$) into a sum of scaled sine and cosine functions (or complex exponentials). This representation provides a unique spectral signature for the signal.

### Key Mathematical Concepts

**1. Fourier Series Representation:**
A periodic function $f(t)$ can be expressed as:
$$
f(t) = a_0 + \sum_{n=1}^{\infty} \left( a_n \cos\left(rac{2\pi n t}{T}ight) + b_n \sin\left(rac{2\pi n t}{T}ight) ight)
$$
where $a_0$ is the average value of the function, and $a_n, b_n$ are the amplitudes of the cosine and sine terms for the $n$-th harmonic.

Alternatively, using complex exponentials:
$$
f(t) = \sum_{n=-\infty}^{\infty} c_n e^{i rac{2\pi n t}{T}}
$$

**2. Calculation of Fourier Coefficients:**
The coefficients are determined by the following integrals:
- $a_0 = rac{1}{T} \int_{T} f(t) dt$
- $a_n = rac{2}{T} \int_{T} f(t) \cos\left(rac{2\pi n t}{T}ight) dt \quad (n \ge 1)$
- $b_n = rac{2}{T} \int_{T} f(t) \sin\left(rac{2\pi n t}{T}ight) dt \quad (n \ge 1)$

And for the complex form:
- $c_n = rac{1}{T} \int_{T} f(t) e^{-i rac{2\pi n t}{T}} dt$

**3. Practical Implementation (Truncated Series):**
In practice, the infinite series is truncated to $N$ terms:
$$
S_N(t) = a_0 + \sum_{n=1}^{N} \left( a_n \cos\left(rac{2\pi n t}{T}ight) + b_n \sin\left(rac{2\pi n t}{T}ight) ight)
$$
The vector of these $2N+1$ coefficients $([a_0, a_1, b_1, \dots, a_N, b_N])$ serves as a feature vector, capturing the dominant spectral characteristics of the signal. These feature vectors are then used to train machine learning models for tasks like signal classification.

### Core Functionalities

This library enables you to:

- **Generate and Plot Fourier Coefficients**: Compute and visualize the Fourier coefficients for various functions, allowing flexible exploration of signal components.
- **1D Signal Generation**: Create diverse 1D signals (e.g., Box, Sawtooth, Exponential, Sinusoidal, Gaussian), essential for developing and validating machine learning models.
- **Signal Analysis**: Includes tools for operations like adding noise, introducing jumps, and potentially detecting such features using spectral properties.
- **Machine Learning Integration**: Streamline the process of preparing datasets from signal coefficients, training neural network models (and other classifiers), and evaluating their performance on signal classification tasks.

This package implements methodologies and concepts that can be further explored in the research paper: **[Using Fourier Series and Machine Learning to Classify 1D-Signals](https://drive.google.com/file/d/1qCV_bS05ocZtUJVro-OqHE7ys68Yv-Mi/view?usp=sharing)**.

## Installation

### Requirements

- Python 3.6 or higher
- NumPy
- TensorFlow 2.4 or higher
- Matplotlib
- Plotly
- Pandas
- scikit-learn

### Install from source

```bash
git clone https://github.com/yourusername/fourier_classification.git
cd fourier_classification
pip install -e .
```

## Usage

### Basic Example

```python
import numpy as np
from fourier_classification.signals import box_signal
from fourier_classification.fourier import fourier_series
from fourier_classification.visualization import plot_signal_and_fourier

# Create domain
x = np.linspace(-np.pi, np.pi, 1500)

# Generate a box signal
signal = box_signal(x, a=2, b=5, normalized=True)

# Compute Fourier coefficients
fourier_coeffs = box_signal(x, a=2, b=5, normalized=True, fourier=True, n_modes=40)

# Plot signal and Fourier coefficients
fig = plot_signal_and_fourier(x, signal, fourier_coeffs, title="Box Signal")
fig.savefig("box_signal.png")
```

### Signal Classification

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

## Module Structure

- `signals.py`: Functions for generating various types of 1D signals
- `fourier.py`: Functions for Fourier series calculations and transformations
- `operations.py`: Functions for signal operations (noise, jumps)
- `models.py`: Neural network models for signal classification
- `visualization.py`: Functions for visualizing signals and results
- `utils.py`: Utility functions for data handling and preprocessing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@article{srour2023fourier,
  title={Using Fourier Series and Machine Learning to Classify 1D-Signals},
  author={Srour, Abbass},
  journal={},
  year={2023}
}
```
