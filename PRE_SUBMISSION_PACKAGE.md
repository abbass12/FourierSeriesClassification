# Pre-Submission Package Guide

**Release:** 2.1.3-preprint

This package is a reproducible, evidence-based refresh of the original Fourier-signal-classification thesis code. It is **not yet a journal-submission package** because the current evidence is insufficient for a defensible claim that inferred concentration-factor jump descriptors improve classification.

## Included evidence

| Item | Location | Status |
|---|---|---|
| MDPI-style pilot manuscript | `paper/main.tex` | Updated with transparent synthetic and ECG200 screening results |
| Audited bibliography | `paper/references.bib` | Core mathematical, time-series, software, and UCR sources included |
| Reproducible synthetic protocol | `run_repeated_validation.py` | Stratified, repeated-seed workflow with CNN baseline and descriptor/factor options |
| Fixed-split UCR protocol | `run_ucr_benchmark.py` | Local UCR-format loader with accuracy and macro-F1 reporting |
| Interactive applet | `notebooks/Interactive_Signal_Demo.ipynb` | Binder- and Colab-compatible bootstrap |
| GPU screening notebook | `notebooks/Confirmatory_Validation_Colab.ipynb` | Prepared, but Colab allocation is currently blocked by account session capacity |
| Implementation tests | `tests/` and `test_results/unit_tests.md` | 10 tests passed in the latest local verification |
| Screening and ablation outputs | `test_results/ECG200_*` | Three-seed ECG200 external screen plus descriptor/factor ablations |
| Expert-feedback drafts | `outreach/feedback_request_drafts.md` | Drafts only; no messages sent |
| Verification checksums | `SUPPLEMENT_MANIFEST.md` | Checksums for the versioned research snapshot `64c26b1` |

## Completed empirical findings

The corrected three-seed synthetic smoke study found equal mean accuracy for raw and Fourier MLP representations (87.92%) and lower mean accuracy for the current Fourier-plus-jump model (80.42%). A separate three-seed fixed-split ECG200 screen found 91.67% mean accuracy for the Fourier MLP, 89.33% for the raw MLP, 85.00% for the compact CNN, and 79.00% for the combined jump-descriptor model. Location-only, magnitude-only, and all three concentration-factor variants tested in the same ECG200 screening remained below the Fourier-only MLP.

> These results are narrow, preliminary findings. They support the conclusion that the prior single-run improvement claim is not currently reproducible. They do not establish that concentration factors are ineffective in all classification settings.

## Submission blockers

The source of truth is `SUBMISSION_READINESS.md`. In short, the work still needs a larger repeated-seed synthetic study, a prespecified multi-dataset public benchmark suite, stronger modern baselines such as ROCKET/MiniROCKET and an InceptionTime-style model, replicated ablations, a locked validation-only hyperparameter protocol, a compiled manuscript, and author verification of every declaration. The existing Google Colab account is authenticated but has reached its active-session cap; no user session was terminated because the session manager did not expose a safe disposable session.

## External actions deliberately not taken

No professor/researcher email has been sent. No journal account was created, no manuscript was uploaded, no journal terms were accepted, and no publication charge was authorized. Those actions require the author’s explicit final approval after the research gates are complete.
