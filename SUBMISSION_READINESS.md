# Submission Readiness Gate

**Version:** 2.1.3-preprint

## Current decision

**Do not submit the manuscript yet.** The single-run result previously presented as evidence that concentration-factor jump features improve accuracy was not reproduced by the corrected, stratified three-seed smoke validation. The present manuscript is an honest pre-submission pilot and a confirmatory-study protocol, not a final journal article claiming a superior classifier.

## Completed

| Requirement | Status | Evidence |
|---|---:|---|
| Modular, reproducible code | Complete | `src/` package and version-controlled repository |
| Deterministic synthetic data and noise | Complete | Seeded NumPy generator in `src/signals.py` |
| Stratified split | Complete | `train_test_split_signals()` and unit test |
| Restored best validation checkpoint | Complete | Deep-copied state dictionary in `src/models.py` |
| Unit tests | Complete | `test_results/unit_tests.md`, 10 passing tests |
| Repeated-seed smoke workflow | Complete | `run_repeated_validation.py` and `test_results/repeated_seed_smoke_v2/` |
| Citation audit | Complete for current draft | `paper/references.bib` and `analysis/` research log |
| Public fixed-split screening | Complete but insufficient alone | ECG200 protocol, results, and initial ablations under `test_results/ECG200_*` |
| Colab execution notebook | Complete | `notebooks/Full_Experiment_Colab.ipynb` |
| Draft outreach letters | Complete | `outreach/feedback_request_drafts.md` |

## Blocking work before a journal submission

| Requirement | Why it is blocking | Required output |
|---|---|---|
| Full repeated-seed study | Three seeds and 80 examples/class are underpowered. | At least 10 fixed seeds; all per-seed data preserved. |
| Larger sample regime | Current smoke configuration is not a final benchmark. | At least 1000 examples per class, or a justified power analysis. |
| Hyperparameter protocol | Architecture/training choices must be selected without using test data. | Search space, validation selection rule, and compute budget. |
| Strong baselines | Raw MLP alone is not representative of modern time-series classification. | CNN/InceptionTime-like and ROCKET/MiniROCKET-like comparisons. |
| Ablation replication | Initial ECG200 descriptor and concentration-factor screens are complete, but one short binary dataset is insufficient. | Replicated ablation matrix across the full benchmark suite. |
| Multi-dataset external evaluation | One ECG200 screen does not establish general applicability. | A prespecified public benchmark suite with licensing/citation details and complete per-dataset outputs. |
| Statistics | A single accuracy does not quantify uncertainty. | Per-seed metrics, confidence intervals, effect sizes, paired tests. |
| Paper compile and author verification | MDPI format and all author declarations must be correct. | Clean PDF, verified affiliations, funding, authorship, AI disclosure, conflicts, data statement. |
| External feedback | The feature construction merits domain-expert review. | Optional but strongly recommended feedback from spectral-methods and time-series experts. |

## External actions requiring author approval

The following actions are intentionally not performed automatically:

1. Sending any feedback or collaboration email.
2. Creating accounts or accepting journal terms.
3. Uploading files or submitting a manuscript to a journal.
4. Paying, authorizing, or agreeing to any publication charge.
5. Making a public release beyond the existing GitHub repository.

## Recommended next run

```bash
# Google Colab GPU or local GPU
python run_repeated_validation.py \
  --seeds 11 23 37 53 71 89 107 131 149 167 \
  --samples-per-type 1000 \
  --points 1500 \
  --modes 50 \
  --epochs 80 \
  --batch-size 64 \
  --output-dir test_results/confirmatory_synthetic
```

This command is a starting protocol, not a guarantee of a publishable outcome. Run it only after Colab authentication is active and after adding the planned baseline and ablation branches.
