# Synthetic Fourier Signal Dataset v1.0.0

## Purpose

This versioned dataset is a large, deterministic synthetic benchmark for classifying one-dimensional piecewise-smooth signals under controlled additive Gaussian noise. The supervised target is the **signal family**, while signal parameters, true support boundaries, Fourier coefficients, and noise condition are stored as metadata for mechanism and edge-recovery analyses.

## Fixed Design

| Component | Specification |
|---|---|
| Sampling domain | Uniform grid on \([-\pi,\pi)\) |
| Samples per signal | 1,024 float32 values |
| Signal families / class labels | `sine` (0), `box` (1), `sawtooth` (2), `exponential` (3), `gaussian` (4) |
| Noise model | Independent zero-mean additive Gaussian noise, calibrated to the clean-signal energy |
| SNR conditions | Clean, 30 dB, 20 dB, 10 dB, and 0 dB |
| Samples per class-SNR stratum | 6,000 |
| Split allocation per stratum | 4,000 train, 1,000 validation, 1,000 test |
| Total signals | 150,000: 100,000 train, 25,000 validation, 25,000 test |
| Signal storage | Sharded compressed NumPy `.npz` files, float32 signals and int64 labels |
| Fourier metadata | First 64 complex DFT coefficients stored as real and imaginary float32 arrays |
| Edge metadata | True compact-support boundaries and clean-signal parameters stored in a per-sample CSV/JSONL shard |
| Reproducibility | Root seed 20260812; independent deterministic child seeds by split, class, SNR condition, and shard |

## Intended Evaluation

The default classification target is the family label. All SNR levels appear in each split, enabling both pooled-noise and SNR-stratified reporting. Parameters are sampled independently per split, not random-split after generation. This avoids exact parameter duplication across train, validation, and test partitions.

The dataset is a controlled benchmark, not a substitute for external real-signal evaluation. Its known generator and boundary metadata enable additional tasks: Fourier reconstruction, concentration-factor edge recovery, robustness versus noise, and comparison against oracle-boundary features.

## Artifact Layout

```text
synthetic_fourier_noise_v1.0.0/
├── manifest.json
├── schema.json
├── README.md
├── metadata/
│   ├── train_metadata.csv
│   ├── val_metadata.csv
│   └── test_metadata.csv
├── shards/
│   ├── train_000.npz ...
│   ├── val_000.npz ...
│   └── test_000.npz ...
├── checksums.sha256
└── validation_report.md
```

The generation process avoids adding large binary dataset shards to Git. The repository records the generator, frozen manifest, schema, validation code, aggregate reports, and checksums. A release bundle may carry the shards separately.
