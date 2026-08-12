# Synthetic Fourier Noise Dataset v1.0.0 — Release Record

## Release Summary

This release contains **150,000 one-dimensional synthetic signals** generated from five labeled signal families under five controlled noise conditions. The data are intended for reproducible Fourier-representation and signal-classification experiments. Each example includes a noisy signal, its clean precursor, Fourier-domain features, label metadata, and the deterministic generation seeds needed to reproduce its stratum.

| Split | Signals | Per family × noise stratum |
|---|---:|---:|
| Train | 100,000 | 4,000 |
| Validation | 25,000 | 1,000 |
| Test | 25,000 | 1,000 |
| **Total** | **150,000** | — |

The five classes are `sine`, `box`, `sawtooth`, `exponential`, and `gaussian`. The noise conditions are clean, 30 dB, 20 dB, 10 dB, and 0 dB additive white Gaussian noise. There are 75 `.npz` shards in the `shards/` directory, alongside split metadata, a schema, a manifest, a checksum file, and a validation report.

## Generation Provenance

The initial complete generation was executed on a Google Colab Tesla T4 with PyTorch 2.11.0+cu128 and CUDA 12.8 from repository revision `fcd5f925f28480a1e02ba999272d4881ea3de06a`. The GPU generation completed in 67.69 seconds. It produced 150,000 signals, 75 shards, and 150,000 metadata records. The validated Colab archive had SHA-256 checksum `2ba4f9cd54deef408181509116fa1a76cfa1c1bf26f72068135dd889e461a683`.

A persistent delivery copy was regenerated with the same versioned deterministic protocol and independently validated in the release environment. The data are generated stratum-by-stratum using platform-independent seeds derived from the release specification. GPU and CPU runs are recorded as separate executions because device-level floating-point implementations may produce small numerical differences even when the protocol, labels, seeds, and distributions are the same.

## Independent Validation

The persistent release copy passed the complete validation suite. The validator checked all shards for schema compliance, finite values, signal length, labels, exact split/class/SNR balance, metadata integrity, stored checksums, SNR calibration, and sampled DFT consistency.

| Validation check | Persistent release result |
|---|---:|
| Signals validated | 150,000 |
| Shards validated | 75 |
| Metadata records | 150,000 |
| Verified checksummed files | 80 |
| Split/class/SNR balance | Exact |
| Maximum absolute SNR calibration error | \(1.7630280950697852\times 10^{-6}\) dB |
| Maximum sampled DFT error | \(3.497493139548169\times 10^{-8}\) |

## Reproduction and Validation

Generate the complete dataset using the provided generator:

```bash
python3 dataset_generation/generate_synthetic_fourier_dataset.py \
  --output-dir datasets/synthetic_fourier_noise_v1.0.0/release_artifact \
  --device auto
```

Validate an existing artifact directory:

```bash
python3 dataset_generation/validate_synthetic_fourier_dataset.py \
  --dataset-dir datasets/synthetic_fourier_noise_v1.0.0/release_artifact
```

The Google Colab notebook `notebooks/Generate_Synthetic_Fourier_Dataset_Colab.ipynb` runs the same complete GPU protocol and produces a compressed release archive.
