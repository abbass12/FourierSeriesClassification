# Replication and Model-Architecture Audit v2.2.1

## Direct Answer

**Yes, I checked the exact modern models and configuration that were run. No, an RNN, LSTM, GRU, or systematically tuned feed-forward neural network has not yet been evaluated.**

More importantly, I did **not** reproduce every numerical result from the pre-modernization thesis notebook exactly. I reproduced the project’s revised, seed-controlled protocol and its key qualitative result: under the corrected implementation and the specified three-seed synthetic GPU screen, the current Fourier-plus-jump peak-vector representation did **not** beat Fourier-only features. That is a different and more defensible claim than matching every historical percentage in the legacy notebook.

## What Was Actually Run

The completed free-Google-Colab run used a Tesla T4 and a current PyTorch implementation. It completed in 77.935 seconds and generated versioned machine-readable outputs in `test_results/colab_screening/`.

| Component | Executed setting |
|---|---|
| Dataset | Five balanced synthetic 1D signal families |
| Samples | 300 examples per family, 1,500 samples per signal |
| Seeds | 11, 23, 37 |
| Noise | None in this screening run |
| Fourier representation | 50 modes; concatenated real and imaginary coefficients |
| Jump representation | Trigonometric concentration factor; up to four location-magnitude pairs; combined locations and magnitudes |
| Training | Up to 40 epochs; batch size 64; Adam with learning rate 0.001; ReduceLROnPlateau; early stopping on validation loss |
| Evaluation | Held-out test partition per seed; seed-wise accuracy; descriptive paired Wilcoxon tests |

### Executed architectures

| Label | Architecture | Input | Was run? |
|---|---|---|---|
| A: raw MLP | Fully connected 256 → 128 → 64, batch normalization, ReLU, dropout 0.2 | 1,500 raw samples | Yes |
| B: Fourier MLP | Same fully connected 256 → 128 → 64 architecture | 100 real-valued Fourier features | Yes |
| C: Fourier + jumps | 32-unit jump branch concatenated with Fourier features, then 256 → 128 → 64, batch normalization, ReLU, dropout 0.2 | Fourier features plus 8 jump-vector values | Yes |
| D: compact CNN | Conv1D channels 32 → 64 → 128, ReLU, batch normalization, max pooling, global average pooling | 1,500 raw samples | Yes |
| RNN / LSTM / GRU | Not implemented | Not applicable | **No** |
| InceptionTime / MiniROCKET | Not implemented | Not applicable | **No** |

## Completed T4 Results

| Model | Mean held-out accuracy | SD | 95% CI | Status of comparison |
|---|---:|---:|---:|---|
| Raw MLP | 91.78% | 0.84 pp | 90.83%–92.73% | Completed three-seed screen |
| Fourier MLP | **93.33%** | 2.00 pp | 91.07%–95.60% | Completed three-seed screen |
| Fourier + jump MLP | 92.56% | 1.17 pp | 91.23%–93.88% | Completed three-seed screen |
| Compact 1D CNN | 89.11% | 2.46 pp | 86.33%–91.89% | Completed three-seed screen |

The central paired test is Fourier-plus-jump versus Fourier-only. The estimated mean difference was **−0.78 percentage points** in favor of Fourier-only, with a two-sided Wilcoxon p-value of 0.50. With only three seeds, the test is underpowered. It cannot prove equivalence or infer a reliable ranking. It does, however, mean that the current evidence does not justify a claim that the jump-vector branch is beneficial.

## What Was and Was Not Replicated from the Original Work

### Preserved original materials

A local archival copy of the original repository was inspected. It includes legacy TensorFlow notebooks, result spreadsheets, saved CSV files, and a `Summary.txt` describing Models A, B, and C.

The legacy summary reports raw-signal results in approximately the 97%–99% range under several dataset-size configurations; it reports Fourier-model results around 93%–98% depending on coefficient count; and it reports Model C values including 54.64%, 40.84%, and 95.66% under different input combinations. These legacy values were not generated under a fully pinned protocol in the repository: the notebooks lack a clean, versioned experiment specification tying seed, split, data-generation parameters, exact model state, early stopping, and output file to each reported number.

### Exact replication status

| Question | Answer |
|---|---|
| Did we rerun the revised code end to end? | **Yes.** Unit-tested code, a repeated-seed smoke study, ECG200 screening/ablations, and a three-seed synthetic Tesla T4 screen were completed. |
| Did we reproduce every historical notebook percentage exactly? | **No.** That would require reconstructing the legacy TensorFlow environment, generation state, split logic, and model/training definitions precisely. The archival material does not yet provide a sufficiently pinned protocol. |
| Did we reproduce the old qualitative claim that explicit jumps improve the model? | **No.** The revised studies do not support that claim for the current extracted jump vector. |
| Does this show the original work was wrong? | **No.** It shows the old conclusion has not yet survived a controlled replication. Dataset-generation and experimental differences could explain the mismatch. |

## Could an RNN Change the Conclusion?

**Possibly, but it would answer a different hypothesis and should be treated as a fair baseline, not a rescue mechanism.** A recurrent network may exploit sequential ordering in raw 1D samples. It cannot automatically establish that concentration-factor jump descriptors are useful.

The correct sequence baseline is a small bidirectional GRU, followed only if justified by a bidirectional LSTM. A GRU is less expensive and has fewer parameters; it is a better first test for 1,500-point sequences. The RNN should use raw samples, normalized consistently with the other raw-input models. A second RNN experiment may concatenate a spatial edge map or a continuous concentration-factor response with the raw signal as an extra channel. It should **not** consume the current padded top-four-peak vector as though it were a sequence.

| RNN experiment | Input | Research question answered |
|---|---|---|
| BiGRU raw baseline | Raw 1D signal | Does sequential recurrence beat raw MLP/CNN? |
| BiGRU + continuous edge channel | Raw signal plus full concentration-factor response | Does spatially preserved spectral-edge information help a sequence model? |
| GRU on Fourier modes, optional | Ordered low-frequency amplitude/phase modes | Does spectral mode order provide recurrent structure? |
| GRU on padded peak vector | Do not prioritize | Artificial order and zero padding make this weakly motivated |

## Could Better FFNN Parameters Change the Conclusion?

**Yes. The current MLP was intentionally a reasonable fixed baseline, not an optimized architecture.** Better tuning could improve raw, Fourier, and jump-augmented models. However, tuning only the jump model after seeing it underperform would invalidate the comparison. Every representation must receive an equal validation-only tuning budget.

The highest-value MLP changes are:

1. Standardize each representation using training-set statistics only. The present raw and Fourier inputs deserve explicit comparable normalization checks.
2. Tune hidden widths, dropout, weight decay, learning rate, batch size, and Fourier mode count through validation only.
3. Add residual MLP blocks or LayerNorm as alternatives to the present batch-normalized stack, particularly for small datasets.
4. Tune the jump branch independently, including its hidden width and dropout, but only through the same frozen protocol as other models.
5. Replace the peak vector with stable pooled edge statistics or the full edge response before concluding that concentration-factor features fail.

### Fair tuning budget

Use a predeclared validation-only budget of 24–32 trials per representation per dataset, or an equivalent fixed random search. Do not test the final hold-out set during selection.

| Hyperparameter | Candidate values |
|---|---|
| Hidden layout | [128, 64], [256, 128, 64], [512, 256, 128] |
| Dropout | 0.0, 0.1, 0.2, 0.4 |
| Learning rate | 3×10⁻⁴, 1×10⁻³, 3×10⁻³ |
| Weight decay | 0, 1×10⁻⁵, 1×10⁻⁴, 1×10⁻³ |
| Fourier modes | 20, 50, 80, 160 |
| Jump branch width | 16, 32, 64, 128 |
| Peak count | 0, 2, 4, 8 |

After selection, freeze the winning configuration for every representation and evaluate it across at least ten paired seeds. Report the selected hyperparameters, complete trial logs, per-seed outputs, parameter counts, training time, and paired effect sizes.

## Recommended Next Experiment

Do **not** run every possible model at once. Start with a disciplined 1D model-comparison extension:

1. Keep the existing raw MLP, Fourier MLP, Fourier-plus-jump MLP, and compact CNN.
2. Add a compact bidirectional GRU raw baseline.
3. Add a BiGRU with raw signal plus full concentration-factor edge-response channel.
4. Add a Fourier MLP plus pooled edge-statistics condition.
5. Run a validation-only, equal-budget random search for every representation.
6. Freeze the protocol and execute ten paired seeds across clean and prespecified noisy conditions.
7. Repeat on a predeclared multi-dataset UCR panel.

This design determines whether the present negative jump result is caused by the **model family**, the **top-four-peak representation**, the **synthetic generator**, or simply inadequate tuning. It is far more informative than adding an RNN in isolation.

## Bottom Line

The completed result is a valid replication of the **revised, explicit protocol**, not a certified reproduction of every number in the legacy notebook. We have not tested RNNs or performed systematic hyperparameter optimization. An RNN or better MLP could improve absolute accuracy, but no evidence currently supports the assumption that it will reverse the Fourier-only versus Fourier-plus-jump comparison. The next scientifically defensible test is a matched, validation-only model and representation study.
