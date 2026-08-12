# Paper Improvement Roadmap v2.2.0

## Executive Assessment

The project has become stronger because the previous single-run claim was tested rather than protected. The completed three-seed Tesla T4 screening found that the **Fourier-only MLP** had the best mean held-out accuracy on the specified synthetic protocol, while the current Fourier-plus-jump representation did not produce a statistically supported advantage. The precise machine-readable result is retained in `test_results/colab_screening/validation_summary.json`.

> The best publishable contribution is **not** “jump descriptors improve classification.” It is a rigorous, reproducible investigation of *when spectral features and concentration-factor jump information help, fail to help, or hurt time-series classification*.

That shift changes the work from a fragile positive-result paper into a useful negative-result and diagnostic methods paper. The original mathematics stays relevant, but the contribution must be framed as a controlled empirical study of spectral representation learning, not as a proof that a hand-built peak descriptor boosts accuracy.

| Current evidence | What it supports | What it does not support |
|---|---|---|
| Synthetic three-seed GPU screen | The experimental code executes reproducibly on a T4 and Fourier-only features are competitive within one specified generator | General time-series performance, a reliable model ranking, or a jump-feature benefit |
| ECG200 three-seed screen | The Fourier-only implementation transfers to one small public fixed-split dataset | A clinical claim, a UCR leaderboard claim, or a general benchmark conclusion |
| Descriptor and concentration-factor ablations | The present zero-padded peak-vector design is sensitive to descriptor choice | A claim that concentration-factor theory is invalid or irrelevant to classification |

## 1. Change the Central Research Question

The current title correctly calls the work a pilot, but it is too narrow and overly associated with a feature that presently underperforms. Replace the primary research question with the following:

> **Under what signal structure, noise level, and data regime do Fourier representations and concentration-factor-derived jump information improve, match, or degrade time-series classification relative to raw-sample and modern baseline methods?**

A suitable future title is:

> **When Do Fourier and Concentration-Factor Features Help Time-Series Classification? A Reproducible Multi-Benchmark Study**

This title admits all scientifically valid outcomes. If jump features remain harmful, the paper still contributes an important falsification: naive conversion of spectral edge estimates into fixed-length peak vectors is not an adequate bridge between edge-detection theory and classification.

## 2. Replace the Current Paper Structure

The present manuscript reads like a transparent pilot report. A journal version should lead with the empirical question and then make the mathematics serve the evaluation.

| Current emphasis | Recommended replacement |
|---|---|
| Mathematical definition followed by a small pilot | Motivation, explicit hypotheses, and a benchmark protocol followed by concise mathematical construction |
| “Fourier plus inferred jumps” as the presumed contribution | A family of spectral representations, including negative controls and ablations |
| Accuracy tables for one synthetic setting and ECG200 | Aggregated multi-dataset results, per-dataset critical-difference/rank analyses, and failure-mode studies |
| A future-tense confirmatory plan | Completed registered protocol and a reproducibility statement |

The revised article should include: an introduction with falsifiable hypotheses; a method section separating theoretical edge reconstruction from classifier representation design; a benchmark section with predeclared datasets and split rules; a results section organized by primary endpoint, robustness, and failure analysis; and a discussion that states exactly what was learned about the mismatch between edge recovery and discriminative utility.

## 3. Build the Evaluation That Reviewers Will Expect

A single ECG200 result cannot carry the journal paper. The UCR/UEA ecosystem provides standardized datasets and reference results, while the `aeon` toolkit makes archive datasets and established algorithms accessible programmatically.[1] A credible scope is **12–20 preselected univariate UCR datasets**, stratified before experiments into categories where discontinuities are plausible and where they are not.

Use a preregistered dataset panel such as the following. Do not select datasets after observing results.

| Dataset stratum | Suggested datasets | Purpose |
|---|---|---|
| ECG or shape-sensitive signals | ECG200, ECGFiveDays, TwoLeadECG, NonInvasiveFetalECGThorax1 | Tests whether abrupt morphological transitions matter |
| Sensor and motion signals | GunPoint, FordA, ItalyPowerDemand, Wafer | Tests scalable classification under varied length and noise conditions |
| Spectral or periodic signals | Coffee, Ham, Meat, InlineSkate | Tests whether frequency composition, rather than local jumps, dominates |
| Synthetic controlled panel | Existing five families plus noise, blur, resampling, shift, and amplitude perturbations | Establishes mechanism and interprets outcomes |

The benchmark protocol should preserve official archive train/test splits. Hyperparameter selection must occur only within training data. For every dataset, repeat training with at least **10 fixed seeds**, report every run, and use the same seed schedule across models. Report balanced accuracy or macro F1 when classes are imbalanced, along with accuracy, parameter counts, wall-clock time, and GPU memory.

Modern time-series evaluation should not be limited to your own neural baselines. InceptionTime is a strong deep-learning reference baseline, while MiniROCKET is a fast, effective transformation baseline.[2] [3] Include simple tabular and distance-based references because recent evidence shows that simple non-temporal baselines can remain competitive on a meaningful portion of archive datasets.[4]

### Required baselines

| Family | Minimum implementation | Why it is necessary |
|---|---|---|
| Simple controls | Logistic regression or ridge on raw samples and on Fourier features; 1-NN Euclidean; 1-NN DTW where feasible | Tests whether a neural network is necessary at all |
| Your intended models | Raw MLP; Fourier MLP; Fourier + current jump descriptor | Preserves the original hypothesis fairly |
| Learned local baseline | Compact 1D CNN and InceptionTime | Tests whether learned temporal filters dominate engineered features |
| Strong non-neural baseline | MiniROCKET with a linear classifier | Provides a highly competitive, efficient reference |
| Optional ensemble ceiling | HIVE-COTE 2.0 on a small subset only | Contextualizes accuracy without making compute infeasible |

## 4. Improve the Feature Method Before Retesting the Claim

The current representation takes up to four peaks, converts them to location-magnitude pairs, sorts them, and zero-pads. This creates discontinuities in the representation itself: a slight change in signal or threshold can alter peak ordering and the number of retained peaks. It is plausible that this design, not concentration-factor theory, causes the observed degradation.

Test a small, predeclared feature family instead of repeatedly tuning one peak vector after inspecting results.

| Ablation | Design | Diagnostic value |
|---|---|---|
| Fourier-only | Real/imaginary coefficients, amplitude/phase alternative, and power spectrum alternative | Establishes the spectral baseline |
| Peak vector | Current top-$k$ location/magnitude design for $k \in \{0,2,4,8\}$ | Tests the original thesis mechanism directly |
| Edge map | Feed the entire concentration-factor response as a second channel to a 1D CNN | Avoids lossy peak selection and preserves spatial structure |
| Smooth pooled statistics | Quantiles, energy, total variation, peak count, and positive/negative edge mass | Tests whether stable summary information is useful |
| Multi-resolution features | Multiple Fourier cutoffs and multiple concentration factors | Tests resolution sensitivity |
| Oracle edge control, synthetic only | Ground-truth jump positions/magnitudes | Separates “edge information is useful” from “the estimator is too noisy” |

The **oracle edge control** is especially valuable. If oracle jump locations do not help on the synthetic classes, the project has learned that jumps are not discriminative for those labels. If oracle edges help but inferred edges do not, the paper has a precise mathematical-engineering finding: estimator error or vectorization destroys the potentially useful information.

## 5. Add an Analysis Layer, Not Only a Leaderboard

The paper will be more original if it explains outcomes rather than merely reporting them. Predeclare analyses relating model differences to measurable dataset properties: estimated total variation, spectral entropy, signal-to-noise ratio, class-wise jump-location separation, sample length, and training-set size.

For each dataset, calculate a paired effect size, such as the seed-wise accuracy difference between Fourier-plus-edge and Fourier-only models. Regress or stratify this difference by the signal properties. This directly answers the improved research question: **what type of signal, if any, benefits from edge information?**

Figures should include: representative signals with true and inferred edges for synthetic data; edge-map failure cases; per-dataset paired difference plots; a rank plot across the benchmark suite; and a compute-versus-accuracy scatter plot. This combination gives the paper a mathematical narrative and an empirical narrative.

## 6. Make the Statistics Match the Claims

The current three-seed Wilcoxon tests are correctly labelled descriptive. Keep that honesty. The confirmatory paper should use ten or more paired seeds per dataset, report the full seed-level table, and avoid treating a small p-value as the only scientific conclusion.

For a multi-dataset study, use paired within-dataset comparisons and aggregate ranks or signed effect sizes across datasets. Report uncertainty intervals, practical effect thresholds, and multiple-comparison handling for the ablation family. Do not use a global pooled accuracy across heterogeneous datasets as the main endpoint.

## 7. Improve Writing and Submission Positioning

The manuscript’s strongest current feature is its candor. Preserve that, but remove submission-blocking language from the final paper. Phrases such as “must not be submitted” and “the author must verify” belong in internal readiness documents, not in the submitted manuscript. Replace them with completed statements after the associated work is done.

Use the following contribution paragraph once the experiments support it:

> We provide a reproducible multi-benchmark evaluation of raw, Fourier, and concentration-factor-derived representations for univariate time-series classification. By separating oracle and inferred edge information and evaluating stable pooled and spatial edge representations, we identify the regimes in which spectral edge information is informative and the regimes in which peak-vector descriptors are unstable or unhelpful.

Avoid claiming novelty for the FFT, Fourier coefficients, or concentration factors themselves. The novelty must be the **controlled bridge from spectral edge detection to modern classification evaluation**, supported by a clean experimental protocol and a failure analysis.

## 8. Prioritized 8-Week Execution Plan

| Week | Deliverable | Submission gate advanced |
|---|---|---|
| 1 | Freeze hypotheses, dataset list, seeds, primary metric, and ablations in `PROTOCOL.md` | Eliminates post hoc design choices |
| 2 | Add `aeon` or equivalent reproducible loaders and baseline wrappers; test all datasets | Establishes a valid benchmark pipeline |
| 3 | Implement edge-map, pooled-statistic, and oracle-edge synthetic controls | Diagnoses representation failure rather than merely tuning |
| 4 | Execute a five-dataset pilot and inspect failures without altering the frozen primary protocol | Validates compute and reporting pipeline |
| 5–6 | Run 12–20 dataset, 10-seed confirmatory benchmark on Colab or staged compute | Produces publishable evidence |
| 7 | Produce tables, rank/effect plots, failure cases, compute metrics, and reproducibility archive | Converts runs into reviewable evidence |
| 8 | Rewrite manuscript, obtain external feedback, and select the journal only after results are known | Aligns claims, venue, and evidence |

## Bottom Line

Do not try to force the original positive hypothesis back into the paper. Improve the paper by making it more rigorous and more interesting: identify the conditions under which **Fourier representations** work, and determine whether **edge information** is genuinely useful, merely redundant, or degraded by the current extraction pipeline. A well-executed answer to that question is substantially more publishable than a retrofitted claim that jump features always improve classification.

## References

[1] [Time Series Classification Website: UCR/UEA archives, reference results, and aeon access](https://www.timeseriesclassification.com/)

[2] [Fawaz et al., InceptionTime: Finding AlexNet for Time Series Classification](https://arxiv.org/abs/1909.04939)

[3] [Dempster et al., MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series Classification](https://arxiv.org/abs/2012.08791)

[4] [Dhariyal et al., Back to Basics: A Sanity Check on Modern Time Series Classification Algorithms](https://arxiv.org/abs/2308.07886)
