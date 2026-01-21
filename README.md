# Fourier Series Signal Classification

A machine learning project for classifying signals using Fourier series representations. This project explores how neural networks can classify different mathematical functions based on their spatial domain representation, Fourier coefficients, or a combination with jump discontinuity data.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abbass12/FourierSeriesClassification/blob/main/SignalClassification.ipynb)
[![Binder - Signal Generation Widget](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/abbass12/FourierSeriesClassification/HEAD?urlpath=voila%2Frender%2FSignalGeneration.ipynb)

## Overview

This project investigates the effectiveness of different signal representations for neural network classification:

- **Spatial Domain**: Direct signal values on a grid (1500 points)
- **Fourier Domain**: Complex Fourier coefficients (20-1000 modes)
- **Combined**: Spatial/Fourier data with jump discontinuity information

### Signal Types

The project generates and classifies five types of signals with compact support on [-a, a]:

| Signal | Mathematical Definition | Parameters |
|--------|------------------------|------------|
| **Box** | f(x) = b for \|x\| < a, else 0 | a: width, b: height |
| **Sawtooth** | f(x) = -bx for \|x\| < a, else 0 | a: width, b: slope |
| **Exponential** | f(x) = c + e^(-bx) for \|x\| < a, else 0 | a: width, b: decay, c: offset |
| **Sinusoidal** | f(x) = c*sin(bx) for \|x\| < a, else 0 | a: width, b: frequency, c: amplitude |
| **Gaussian** | f(x) = e^(-a*x^(2b)) for \|x\| < a, else 0 | a: decay, b: shape |

## Installation

```bash
# Clone the repository
git clone https://github.com/abbass12/FourierSeriesClassification.git
cd FourierSeriesClassification

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- NumPy
- TensorFlow 2.x
- Pandas
- Plotly
- SciPy
- ipywidgets (for interactive notebooks)
- voila (for web widgets)

## Project Structure

```
FourierSeriesClassification/
├── Fourier.py              # Core Fourier transform operations
├── FunctionDefinitions.py  # Signal generator functions (box, saw, exp, sinu, gaus)
├── FunctionOperations.py   # Utility functions (noise, jump extraction)
├── Generate.py             # Batch signal generation pipeline
├── modules.py              # Plotting utilities
├── SignalClassification.ipynb  # Main classification experiments
├── SignalGeneration.ipynb      # Interactive signal generation
├── ModelC.ipynb                # Advanced model with jump data
├── Results/                    # Output visualizations
│   ├── B.1.png, B.2.png       # Model B results
│   └── Model C/               # Model C results
├── archive/                    # Trained models and legacy code
│   └── Model{1-4}TrainedTF/   # Saved TensorFlow models
└── *.csv                       # Result data files
```

## Usage

### Generating Signals

```python
import numpy as np
from Generate import Generate
from Fourier import x

# Generate 100 box signals in spatial domain
signals, a_params, b_params = Generate('Box', x, Amount=100)

# Generate 50 sinusoidal signals in Fourier domain (40 coefficients)
signals, a_params, b_params, c_params = Generate(
    'Sin', x, Amount=50, fourier=True, N=40
)

# Generate signals with jump discontinuity data
signals, a_params, b_params = Generate(
    'Box', x, Amount=100, fourier=True, jump=True, N=40
)
```

### Using Signal Functions Directly

```python
import numpy as np
from FunctionDefinitions import box, saw, exp, sinu, gaus

x = np.linspace(-np.pi, np.pi, 1500)[:-1]

# Generate a box signal
signal = box(x, a=1.5, b=1.0, normalized=True)

# Get Fourier coefficients
coeffs = box(x, a=1.5, b=1.0, fourier=True, N=40)

# Generate with noise
noisy_signal = box(x, a=1.5, b=1.0, noise=True, noiseParameter=0.1)
```

### Fourier Series Reconstruction

```python
from Fourier import fourier_series, dft

# Reconstruct signal from coefficients
reconstructed = fourier_series(coeffs, x, method='precompute')

# Available methods: 'precompute' (fastest), 'forloop', 'ifft'
```

## Models and Results

### Model A: Spatial Domain Training

Trained on grid-point data (1500 points per signal).

| Training Size | Test on Spatial | Test on Fourier (N=80) |
|--------------|-----------------|------------------------|
| 500 signals | 97.0% | 91% |
| 5000 signals | 97.6% | 97% |
| 50000 signals | 98.9% | 97.8% |

### Model B: Fourier Domain Training

Trained directly on Fourier coefficients.

| Training (N) | Test N=20 | Test N=80 | Test Spatial |
|-------------|-----------|-----------|--------------|
| N=20 | 97.96% | 92.88% | 92.70% |
| N=80 | 94.82% | 97.94% | 97.46% |

### Model C: Combined with Jump Data

Trained on spatial data augmented with jump discontinuity information.

| Test Data | Accuracy |
|-----------|----------|
| Spatial + Jump | 95.66% |
| Fourier N=80 + Jump | 54.64% |
| Fourier N=20 + Jump | 40.84% |

## Key Findings

1. **Spatial vs Fourier**: Models trained on spatial data generalize well to Fourier representations with sufficient coefficients (N >= 80)

2. **Coefficient Count**: Classification accuracy improves with more Fourier coefficients, approaching spatial domain performance around N=160

3. **Jump Information**: Adding jump discontinuity data helps spatial domain classification but doesn't transfer well to Fourier domain

4. **Cross-Domain Generalization**: Models trained on Fourier coefficients can classify spatial signals with high accuracy when using sufficient modes

## Notebooks

- **SignalClassification.ipynb**: Main experiments comparing Models A, B, C
- **SignalGeneration.ipynb**: Interactive widget for visualizing signal generation
- **ModelC.ipynb**: Detailed experiments with jump discontinuity augmentation
- **training.ipynb / training2.ipynb**: Model training pipelines

## API Reference

### Fourier.py

- `fourier_series(cn, X, method)`: Reconstruct signal from Fourier coefficients
- `dft(N, M)`: Compute DFT matrix
- `partial_fourier_sum(M2, M1, x, cn, spectral_type)`: Filtered partial sum for jump detection

### FunctionDefinitions.py

- `box(x, a, b, ...)`: Box/rectangular pulse function
- `saw(x, a, b, ...)`: Sawtooth wave function
- `exp(x, a, b, c, ...)`: Exponential function
- `sinu(x, a, b, c, ...)`: Sinusoidal function
- `gaus(x, a, b, ...)`: Gaussian-like function

### Generate.py

- `Generate(signal, x, Amount, ...)`: Batch generate signals with random parameters

## License

This project is available for academic and research purposes.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{fourier_series_classification,
  author = {abbass12},
  title = {Fourier Series Signal Classification},
  url = {https://github.com/abbass12/FourierSeriesClassification}
}
```
