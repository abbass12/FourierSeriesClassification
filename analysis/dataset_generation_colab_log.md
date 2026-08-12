# Synthetic Fourier Dataset v1.0.0 Colab Execution Log

- **Runtime verification:** Google Colab Tesla T4, 15,360 MiB GPU memory; PyTorch 2.11.0+cu128; CUDA 12.8; driver 580.82.07.
- **Source revision:** `fcd5f925f28480a1e02ba999272d4881ea3de06a`.
- **Generation status:** The GPU generator started successfully. At the latest check it had written 14 of 75 balanced shards, covering all training SNR conditions for sine and box and the first four conditions for sawtooth. Each completed shard contains 4,000 signals.
- **Dataset target:** 150,000 total signals, with 75 strata across train/validation/test, five families, and five noise conditions.

The generator completed all 75 planned shards: 25 training shards of 4,000 samples each, 25 validation shards of 1,000 samples each, and 25 test shards of 1,000 samples each. This confirms that all five signal families and all five noise conditions were emitted for every split. The Colab cell then proceeded to metadata assembly, checksum creation, and full-dataset validation.

The complete T4 run finished in 67.69 seconds. The validator passed all checks across 150,000 signals and 75 shards. It verified exact class-SNR-split balance, 150,000 metadata records, 80 checksummed files, maximum per-signal SNR calibration error of 1.8321749806204934e-06 dB, and maximum sampled DFT discrepancy of 4.169896859391993e-08. The generated dataset directory occupied 712 MB before archive packaging.

The full validation report confirmed exact split, class, and SNR balance for all 150,000 signals. The next operation is archive packaging so the 712 MB validated dataset remains accessible after the transient Colab runtime is released.

Archive packaging completed successfully. The resulting file is `synthetic_fourier_noise_v1.0.0.zip`, size 0.67 GiB, with SHA-256 `2ba4f9cd54deef408181509116fa1a76cfa1c1bf26f72068135dd889e461a683`. The notebook download transfer was initiated to preserve the artifact outside Colab.

