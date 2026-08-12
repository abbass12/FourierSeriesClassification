# Manuscript Result Validation Report

Protocol: `2.2.2`

| Validation target | Status | Primary artifact |
|---|---|---|
| synthetic_smoke | pass | `test_results/repeated_seed_smoke_v2/validation_summary.json` |
| ecg200_screening | pass | `test_results/ECG200_screening/ECG200_summary.json` |
| ecg200_jump_location_ablation | pass | `test_results/ECG200_locations/ECG200_summary.json` |
| ecg200_jump_magnitude_ablation | pass | `test_results/ECG200_magnitudes/ECG200_summary.json` |
| ecg200_sigma_polynomial_ablation | pass | `test_results/ECG200_poly/ECG200_summary.json` |
| ecg200_sigma_exponential_ablation | pass | `test_results/ECG200_exp/ECG200_summary.json` |
| manuscript_displayed_values | pass | `paper/main.tex` |

## Scope

All checks confirm internal artifact consistency, configuration agreement, recomputed summary statistics, and displayed manuscript values. They do not establish external generalization or confirmatory statistical power.
