# ECG200 Fixed-Split Screening Result

**Dataset source:** The UCR/UEA Time Series Classification website describes ECG200 as a two-class, univariate ECG dataset with 100 training series, 100 test series, and length 96. The repository stores no third-party dataset file; it records only the loader, source URL, and derived experiment outputs. The archive must be cited as Dau et al. (2019).

**Source pages:**

- https://www.timeseriesclassification.com/description.php?Dataset=ECG200
- https://www.cs.ucr.edu/~eamonn/time_series_data_2018/

## Protocol

The official train/test split was preserved. For each of three random seeds (`11`, `23`, `37`), the 100-series training partition was split stratified 80/20 for training/validation. Each series was z-normalized independently. The raw MLP, compact 1D CNN, Fourier MLP, and Fourier-plus-jump MLP were trained up to 50 epochs with validation early stopping. The Fourier and jump settings were 32 modes, trigonometric concentration factor, four peak pairs, and combined locations/magnitudes. The run used CPU.

## Results

| Model | Mean test accuracy | Sample SD | Mean macro F1 | Sample SD |
|---|---:|---:|---:|---:|
| Raw MLP | 89.33% | 0.58 pp | 88.22% | 0.71 pp |
| Fourier MLP | 91.67% | 1.15 pp | 90.86% | 1.30 pp |
| Fourier plus inferred jumps | 79.00% | 6.24 pp | 74.26% | 11.03 pp |
| Compact 1D CNN | 85.00% | 2.65 pp | 83.30% | 3.77 pp |

The jump-augmented model was lower than the raw MLP, Fourier MLP, and compact CNN by 10.33, 12.67, and 6.00 percentage points, respectively. The two-sided Wilcoxon p-value was 0.25 for each comparison because there were only three seeds; this is not a confirmatory statistical result.

## Interpretation

This is a **screening experiment**, not a state-of-the-art benchmark. It confirms the central reproducibility finding of the synthetic smoke study: the current peak-location/magnitude jump-feature implementation does not demonstrate a classification advantage. It also demonstrates that the evaluation code can preserve a public archive’s fixed train/test split and calculate metrics for a binary classifier correctly.

The dataset is small, only one archive task was evaluated, and the compact CNN is not equivalent to InceptionTime or ROCKET. The results must not be compared to published UCR leaderboards or used to claim general clinical performance. Before submission, the project still needs a prespecified multi-dataset benchmark suite, stronger baselines, and the full ablation matrix.
