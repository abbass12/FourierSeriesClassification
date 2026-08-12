# Manuscript Result Validation v2.2.2

## Scope

This report records a clean local regeneration of every empirical table currently reported in `paper/main.tex`, followed by automated consistency checks. The frozen commands and expected values are defined in `experiments/manuscript_validation_protocol_v2.2.2.json`.

## Clean Regeneration

The full protocol regenerated the following outputs under `validation/fresh_cpu_v2.2.2/` using CPU execution:

| Result family | Artifact directory | Outcome |
|---|---|---|
| Synthetic three-seed smoke table | `repeated_seed_smoke_v2/` | Regenerated and passed validation |
| ECG200 primary screening table | `ECG200_screening/` | Regenerated and passed validation |
| ECG200 locations-only ablation | `ECG200_locations/` | Regenerated and passed validation |
| ECG200 magnitudes-only ablation | `ECG200_magnitudes/` | Regenerated and passed validation |
| ECG200 polynomial-factor ablation | `ECG200_poly/` | Regenerated and passed validation |
| ECG200 exponential-factor ablation | `ECG200_exp/` | Regenerated and passed validation |

The automated validator completed **seven checks**: six empirical-artifact checks plus a manuscript displayed-value check. It recomputed each reported mean and sample standard deviation from the new per-seed CSV files, compared these to the new JSON summaries and frozen protocol, and confirmed that the paper contains the corresponding displayed values.

## Code Checks

```text
pytest -q
10 passed in 4.10s
```

The test suite covers signal generation, Fourier feature dimensions and reconstruction, jump-feature ablations, CNN shape handling, UCR loading and stratified splitting, and binary confusion-matrix handling.

## Provenance

`validation/fresh_cpu_v2.2.2/SHA256SUMS.txt` supplies SHA-256 digests for the frozen protocol, fresh results, manuscript, validation scripts, and generated reports. `validation/fresh_cpu_v2.2.2/run_manifest.json` preserves every command used for the clean regeneration.

## Interpretation Boundary

This validation establishes that the paper's current empirical tables are reproducible under their stated **three-seed pilot/screening protocols**. It does not convert them into confirmatory, generalizable conclusions. The manuscript must retain its pilot framing until the multi-dataset and larger-seed confirmatory plan is completed.
