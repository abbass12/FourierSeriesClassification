# Unit-Test Results

**Version:** 2.1.0-preprint

**Command executed**

```bash
python3 -m pytest -q tests/test_core.py
```

**Outcome:** `9 passed in 4.02s`

## Covered checks

| Test | Result | Purpose |
|---|---:|---|
| Reproducible noisy synthetic data | Pass | Confirms a fixed data seed yields identical noisy data. |
| Fourier feature dimensions | Pass | Verifies exact dimensions for even and odd mode counts. |
| Fourier reconstruction finiteness | Pass | Verifies finite numerical outputs for partial sums. |
| Jump-feature pipeline | Pass | Verifies finite edge/jump outputs and fixed-size feature vectors. |
| Combined feature dimensions | Pass | Verifies Fourier-plus-jump feature length. |
| Stratified split | Pass | Verifies equal per-class counts across partitions for balanced synthetic data. |
| CNN baseline shape | Pass | Verifies the one-dimensional CNN accepts raw batches and returns one five-class logit vector per sample. |
| Jump feature ablations | Pass | Verifies location-only, magnitude-only, and combined jump-feature dimensions. |
| UCR benchmark loader | Pass | Verifies UCR-style TSV loading, contiguous label encoding, and stratified training/validation partitioning. |

These tests establish only basic implementation checks. They do not validate scientific claims, benchmark competitiveness, or real-world generalization.
