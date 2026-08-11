# Fourier Series Signal Classification

**Version 2.0.0**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abbass12/FourierSeriesClassification/blob/main/notebooks/Interactive_Signal_Demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/abbass12/FourierSeriesClassification/HEAD?urlpath=voila%2Frender%2Fnotebooks%2FInteractive_Signal_Demo.ipynb)

## Overview

This repository contains the complete codebase for the paper **"Using Fourier Series and Machine Learning to Classify Signals"** submitted to MDPI Mathematics.

We investigate whether Fourier series coefficients can effectively replace raw grid-point data for neural network-based signal classification, and whether incorporating explicit jump discontinuity information (via the Gelb-Tadmor concentration factor method) improves classification performance.

## Key Results

| Model | Input Representation | Accuracy (N=50) |
|-------|---------------------|-----------------|
| Model A | Raw signal data (1500 points) | 93.0% |
| Model B | Fourier coefficients only | 93.2% |
| Model C | Fourier coefficients + jump data | 93.6% |

Model C achieves the highest accuracy at all tested mode counts, with the advantage most pronounced at lower N values where jump information compensates for limited frequency resolution.

## Repository Structure

```
fourier-signal-classification/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── run_experiment.py            # Main experiment runner
├── src/                         # Source code package
│   ├── __init__.py
│   ├── signals.py               # Signal generation (5 types)
│   ├── fourier.py               # Fourier analysis & jump detection
│   ├── models.py                # PyTorch neural network models
│   ├── plotting.py              # Publication-quality figure generation
│   └── run_experiments.py       # Full experiment suite
├── notebooks/                   # Interactive demos
│   └── Interactive_Signal_Demo.ipynb
├── results/                     # Experiment outputs
│   ├── experiment_results.json  # Raw numerical results
│   └── figures/                 # Generated figures for paper
├── paper/                       # LaTeX manuscript
│   ├── main.tex
│   └── references.bib
└── tests/                       # Unit tests
```

## Quick Start

### Installation

```bash
git clone https://github.com/abbass12/FourierSeriesClassification.git
cd FourierSeriesClassification
pip install -r requirements.txt
```

### Run Experiments

```bash
python run_experiment.py
```

This will train all three models and save results to `results/experiment_results.json`.

### Generate Figures

```python
from src.plotting import generate_all_paper_figures
import json

with open('results/experiment_results.json') as f:
    results = json.load(f)

generate_all_paper_figures(results, 'results/figures')
```

### Google Colab (Free GPU)

Open the notebook in Colab for interactive demos and GPU-accelerated training:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abbass12/FourierSeriesClassification/blob/main/notebooks/Interactive_Signal_Demo.ipynb)

## Methodology

### Signal Types
1. **Sine** - Sinusoidal with compact support
2. **Box** - Rectangular pulse (has jump discontinuities)
3. **Sawtooth** - Linear ramp with compact support (has jump discontinuities)
4. **Exponential** - Exponential decay with compact support
5. **Gaussian** - Gaussian pulse with compact support

### Edge Detection (Gelb-Tadmor Method)
We use the Generalized Conjugate Partial Fourier Sum with concentration factors to detect jump discontinuities directly from Fourier coefficients. Three types of concentration factors are implemented:
- Trigonometric (Fourier factors)
- Polynomial factors
- Exponential factors

Reference: Gelb, A. & Tadmor, E. (2000). "Detection of edges in spectral data II. Nonlinear enhancement." SIAM J. Numer. Anal., 38(4), 1389-1408.

## Dependencies

- Python 3.9+
- PyTorch 2.0+
- NumPy
- SciPy
- Matplotlib
- Seaborn
- ipywidgets (for interactive notebooks)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{srour2026fourier,
  author  = {Srour, Abbass},
  title   = {Using Fourier Series and Machine Learning to Classify Signals},
  journal = {Mathematics},
  year    = {2026}
}
```

## License

MIT License

## Author

Abbass Srour - University of Michigan - abbasss@umich.edu
