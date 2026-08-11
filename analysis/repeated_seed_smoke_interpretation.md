# Repeated-Seed Smoke Validation, v2

**Configuration:** 3 seeds (`11, 23, 37`), 80 synthetic samples per class, 1500 time samples, 50 Fourier modes, 12 epochs, CPU. The split was stratified and the best-validation checkpoint was restored.

| Representation | Mean test accuracy | Sample SD | Approximate 95% CI |
|---|---:|---:|---:|
| Model A, raw samples | 87.92% | 1.44 pp | 86.28% to 89.55% |
| Model B, Fourier coefficients | 87.92% | 1.91 pp | 85.76% to 90.08% |
| Model C, Fourier plus inferred jumps | 80.42% | 3.82 pp | 76.10% to 84.74% |

## Interpretation

This is a **smoke validation**, not a final study. It is intentionally underpowered and uses a smaller training budget than a GPU submission run. Nonetheless, it does not reproduce the prior single-run claim that jump features improve accuracy. Therefore that claim must be removed from the paper until a larger repeated-seed ablation establishes it, if it does.

The test script and raw outputs are preserved under `test_results/repeated_seed_smoke_v2/`. The next rigorous run should use at least 10 seeds, a larger dataset, tuned but validation-only hyperparameters, and a standard time-series benchmark or real-world dataset. Any claims should then report mean ± standard deviation, confidence intervals, and paired test results.
