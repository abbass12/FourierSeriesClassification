# Fourier Series Signal Classification

**Version 2.1.3-preprint**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abbass12/FourierSeriesClassification/blob/main/notebooks/Confirmatory_Validation_Colab.ipynb)
[![Launch on Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/abbass12/FourierSeriesClassification/main?labpath=notebooks%2FInteractive_Signal_Demo.ipynb)

## Project Status

This repository contains a **reproducible pilot study**, not a completed or accepted journal article. It tests whether truncated Fourier coefficients and concentration-factor jump descriptors are useful features for classifying five families of synthetic one-dimensional signals.

An earlier one-off experiment suggested that jump features improved classification accuracy. After correcting the evaluation protocol and running a small repeated-seed smoke test, that improvement did **not** reproduce. The current code and manuscript preserve this negative result transparently. Do not cite the repository as evidence that concentration-factor jump features improve general signal classification.

See [`SUBMISSION_READINESS.md`](SUBMISSION_READINESS.md) for the required work before journal submission.

## Completed Smoke Validation

The current smoke study uses three independent seeds, 80 examples per synthetic class, a stratified 70/10/20 split, 12 epochs, and 50 Fourier modes.

| Model | Representation | Mean test accuracy | Interpretation |
|---|---|---:|---|
| A | 1500 raw samples | 87.92% | Pilot result only |
| B | Real and imaginary Fourier coefficients | 87.92% | Pilot result only |
| C | Fourier coefficients plus inferred jump descriptors | 80.42% | No improvement demonstrated |

Raw per-seed outputs and statistical summaries are in [`test_results/repeated_seed_smoke_v2/`](test_results/repeated_seed_smoke_v2/). The result is underpowered and limited to synthetic data.

## Public-Benchmark Screening

A separate fixed-split ECG200 screening uses the UCR archive's 100/100 train/test partition and three training seeds. It is not a leaderboard result or a clinical claim.

| Model | Mean test accuracy | Mean macro F1 |
|---|---:|---:|
| Raw MLP | 89.33% | 88.22% |
| Fourier MLP | 91.67% | 90.86% |
| Fourier plus inferred jumps | 79.00% | 74.26% |
| Compact 1D CNN | 85.00% | 83.30% |

The current inferred-jump implementation again did not improve classification. The initial descriptor ablation found 84.33% for locations only, 83.33% for magnitudes only, and 79.00% for the combined descriptor, versus 91.67% for Fourier features alone. The initial concentration-factor screening found 79.00%, 79.00%, and 85.00% for trigonometric, polynomial, and exponential combined descriptors, respectively. These are narrow three-seed screening results, not general conclusions. Results, protocol, and limitations are in [`analysis/ecg200_screening_interpretation.md`](analysis/ecg200_screening_interpretation.md), with outputs in [`test_results/ECG200_screening/`](test_results/ECG200_screening/), [`test_results/ECG200_locations/`](test_results/ECG200_locations/), [`test_results/ECG200_magnitudes/`](test_results/ECG200_magnitudes/), [`test_results/ECG200_poly/`](test_results/ECG200_poly/), and [`test_results/ECG200_exp/`](test_results/ECG200_exp/). The data itself is not redistributed in this repository; obtain and cite it from the [UCR archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/).

## Repository Structure

```text
FourierSeriesClassification/
├── README.md                         # This overview
├── SUBMISSION_READINESS.md           # Required gates before external submission
├── SUBMISSION_PLAN.md                # Journal and outreach planning notes
├── requirements.txt                  # Python dependencies
├── environment.yml                   # Binder/Jupyter environment specification
├── run_experiment.py                 # Legacy one-off experiment runner
├── run_repeated_validation.py        # Repeated-seed synthetic validation runner
├── run_ucr_benchmark.py               # Fixed-split UCR-format benchmark runner
├── src/
│   ├── signals.py                    # Reproducible signal generation and stratified splits
│   ├── fourier.py                    # FFT features and concentration-factor edge extraction
│   ├── models.py                     # PyTorch classifiers and early stopping
│   ├── plotting.py                   # Figure-generation utilities
│   └── benchmarks.py                 # Local UCR-format dataset loading
├── notebooks/
│   ├── Interactive_Signal_Demo.ipynb
│   └── Full_Experiment_Colab.ipynb
├── paper/
│   ├── main.tex                      # Transparent pre-submission pilot manuscript
│   └── references.bib                # Audited bibliography
├── tests/
│   └── test_core.py                  # Unit tests
├── test_results/                     # Recorded validation outputs
├── analysis/                         # Research log, journal fit, and audit notes
└── outreach/
    └── feedback_request_drafts.md    # Drafts only; do not send automatically
```

## Installation

```bash
git clone https://github.com/abbass12/FourierSeriesClassification.git
cd FourierSeriesClassification
python -m pip install -r requirements.txt
python -m pip install pytest
```

## Verify the Code

```bash
python -m pytest -q tests/test_core.py
```

The current suite checks deterministic synthetic-noise generation, Fourier feature dimensionality, finite partial sums, jump-feature output shape, and stratified splitting.

## Run the Smoke Protocol

```bash
python run_repeated_validation.py \
  --seeds 11 23 37 \
  --samples-per-type 80 \
  --epochs 12 \
  --batch-size 32 \
  --output-dir test_results/repeated_seed_smoke
```

## Run the Planned Confirmatory Protocol

A free-GPU Colab notebook is included, but it requires a Google account session to connect a runtime. The confirmatory study must be expanded with modern baselines, concentration-factor ablations, and at least one public benchmark before a journal submission is appropriate.

```bash
python run_repeated_validation.py \
  --seeds 11 23 37 53 71 89 107 131 149 167 \
  --samples-per-type 1000 \
  --points 1500 \
  --modes 50 \
  --epochs 80 \
  --batch-size 64 \
  --output-dir test_results/confirmatory_synthetic
```

## Methods

### Signal families

The controlled generator produces five balanced families on $[-\pi,\pi)$: compactly supported sine, box, sawtooth, exponential, and Gaussian signals. Parameters vary within predefined family-specific ranges and each signal is normalized to unit maximum absolute amplitude.

### Feature representations

- **Model A:** raw samples.
- **Model B:** real and imaginary parts of truncated Fourier coefficients.
- **Model C:** Model B features plus up to four inferred jump locations and signed magnitudes, extracted with a concentration-factor edge operator.

The edge-extraction construction is motivated by Gelb and Tadmor's spectral edge-detection work. It does not itself guarantee a classification benefit.

## Citation

If you use the code, cite the code release rather than an unvalidated performance claim. The package's BibTeX entry is in [`paper/references.bib`](paper/references.bib) under `srour2026code`.

## License and Contributions

No repository license has been selected yet. Do not assume reuse rights beyond those granted by the repository owner. Contributions, external emails, and publication submissions require the author’s explicit approval.
