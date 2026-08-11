# Publication and Feedback Plan

**Status:** Pre-submission research plan, version 2.1.3-preprint. This document replaces the prior plan because the claimed single-run performance gain for Fourier-plus-jump features was not reproduced by the corrected repeated-seed smoke validation.

## Publication decision

The manuscript is **not ready for journal submission**. It currently documents a transparent synthetic-data pilot and a confirmatory protocol. The pilot does not establish that concentration-factor jump descriptors improve classification, nor does it establish generalization to real signals. A final target journal should be selected only after the blocking validation work in [`SUBMISSION_READINESS.md`](SUBMISSION_READINESS.md) is complete.

## Provisional journal fit

| Journal or venue category | Potential fit after confirmatory work | Condition for a credible submission | Current recommendation |
|---|---|---|---|
| Applied and Computational Harmonic Analysis | High only for a genuinely new mathematical/spectral-method contribution | A rigorous advance in edge recovery or spectral feature construction, supported by theory and experiments | Not appropriate for the current pilot alone |
| Digital Signal Processing | Potentially suitable for applied signal-classification work | Real or standard public signals, competitive signal-processing baselines, and a clear empirical contribution | Consider after benchmark validation |
| Journal of Computational and Applied Mathematics | Potentially suitable for computational-method work | Careful numerical methodology, reproducibility, and substantive computational insight beyond a course-project replication | Consider after a robust ablation and analysis |
| Mathematics | Potentially suitable for an applied-mathematics methods paper | Transparent scope fit, strong mathematical exposition, and completed validation; check the current official author instructions before submission | Possible later option, not a fast-track default |
| Conference/workshop or archival preprint | Useful for early scholarly feedback | Clearly label as a preprint/pilot and do not claim a validated accuracy improvement | Consider after author review |

Publisher metrics, publication charges, and review times change frequently and are deliberately not listed here. They must be checked on the official journal page at the time of submission. The current scope and author-instruction notes are preserved under [`analysis/`](analysis/).

## Evidence that must exist before targeting a journal

The final article needs a complete repeated-seed study, validation-only hyperparameter selection, meaningful ablations of the concentration factor and jump descriptors, at least one modern classification baseline, and at least one public real-world or standardized benchmark. It must report all runs rather than only the best run, including confidence intervals, class-wise performance, paired comparisons, runtime, parameter counts, and limitations. The current code supports stratified splits, deterministic synthetic generation, validated checkpoint restoration, an MLP raw baseline, a Fourier MLP, a Fourier-plus-jump MLP, and a compact CNN baseline; the public-benchmark and stronger-baseline components remain incomplete.

## Expert feedback strategy

The purpose of initial outreach is methodological feedback, not a request for endorsement, authorship, or expedited publication. Draft messages are retained in [`outreach/feedback_request_drafts.md`](outreach/feedback_request_drafts.md) and must be reviewed by the author before sending.

| Potential contact | Relevance | Verification source |
|---|---|---|
| Anne E. Gelb, Dartmouth College | Co-author of the foundational concentration-factor spectral edge-detection papers | [Dartmouth faculty profile](https://faculty-directory.dartmouth.edu/anne-e-gelb) |
| Eitan Tadmor, University of Maryland | Co-author of the foundational spectral edge-detection work | [University of Maryland profile](https://www.math.umd.edu/~tadmor/) |
| Jeffrey A. Fessler, University of Michigan | Statistical signal processing, machine learning, optimization, inverse problems, and MRI | [University of Michigan profile](https://medschool.umich.edu/profile/1686/jeffrey-fessler) |

Do not send messages automatically. Each email must include a concise, honest description of the current pilot, a link to the reproducible repository, and one or two specific questions. It must not state or imply that the method has already been validated or accepted.

## Future cover-letter template

A journal cover letter must not be used until all empirical claims in the manuscript are supported by the completed evidence package. At that point, it should state the final manuscript title, its narrow validated contribution, the target journal’s scope fit, the code/data availability, prior-publication status, author declarations, and any suggested reviewers. It must not describe uncompleted experiments, unverified performance, or a submission as “accepted.”

## Immediate sequence

First, resolve the Colab session-capacity block or use another available GPU environment. Second, run and archive the GPU screening protocol from `notebooks/Confirmatory_Validation_Colab.ipynb`. Third, complete the concentration-factor and jump-descriptor ablations and add public-benchmark baselines. Fourth, revise the manuscript around the actual results. Fifth, obtain author approval before any external outreach or journal submission.
