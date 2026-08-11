# Journal Submission and Outreach Plan

**Version 1.0.0**

## Target Journals (Ranked by Fit)

| # | Journal | Publisher | IF | APC | Review Time | Fit Rationale |
|---|---------|-----------|-----|-----|-------------|---------------|
| 1 | **Mathematics** | MDPI | 2.3 | ~$2,790 | 30-45 days | Open access, fast review, accepts applied math + ML |
| 2 | **Applied and Computational Harmonic Analysis** | Elsevier | 2.5 | Free (subscription) | 3-6 months | Perfect fit: Fourier analysis + computation |
| 3 | **Digital Signal Processing** | Elsevier | 2.9 | Free (subscription) | 2-4 months | Signal processing + ML focus |
| 4 | **Journal of Computational and Applied Mathematics** | Elsevier | 2.1 | Free (subscription) | 2-4 months | Computational math focus |
| 5 | **Signal Processing** | Elsevier | 4.4 | Free (subscription) | 3-6 months | Higher impact, more competitive |
| 6 | **IEEE Signal Processing Letters** | IEEE | 3.2 | Free (subscription) | 1-2 months | Short papers, fast turnaround |

### Recommended Strategy

Submit to **MDPI Mathematics** first for fast turnaround and open access visibility. If rejected, escalate to **Digital Signal Processing** or **JCAM**. The paper's interdisciplinary nature (math + ML) makes MDPI Mathematics the ideal first target.

## Researchers to Contact for Feedback

### Primary Contacts (Directly Relevant Work)

| Name | Affiliation | Relevance | Email Pattern |
|------|-------------|-----------|---------------|
| **Anne Gelb** | Dartmouth College | Originator of the edge detection method we use; her concentration factor work is the foundation of our Model C | anne.gelb@dartmouth.edu |
| **Eitan Tadmor** | University of Maryland (CSCAMM) | Co-author of the Gelb-Tadmor method; expert in spectral methods | tadmor@cscamm.umd.edu |
| **Rick Archibald** | Oak Ridge National Lab | Collaborated with Gelb on MRI Fourier reconstruction | archibaldrk@ornl.gov |

### Secondary Contacts (Related ML + Signal Processing)

| Name | Affiliation | Relevance |
|------|-------------|-----------|
| **Tim O'Shea** | Virginia Tech / DeepSig | Pioneer in deep learning for radio signal classification (RML datasets) |
| **Yue Zhao** | USC | Machine learning for time series and signal anomaly detection |
| **Dong Yu** | Tencent AI Lab | Deep learning for signal processing |

### University of Michigan Contacts

| Name | Department | Relevance |
|------|-----------|-----------|
| **Jeffrey Fessler** | EECS | MRI reconstruction, signal processing, Fourier methods |
| **Laura Balzano** | EECS | Signal processing, machine learning |
| **Anna Gilbert** | Mathematics | Compressed sensing, Fourier analysis |

## Outreach Email Template

### For Feedback Request (to Gelb/Tadmor)

```
Subject: Feedback Request: Using Your Edge Detection Method for ML Signal Classification

Dear Professor [Name],

I am writing to request your feedback on a manuscript that builds directly upon your foundational work on edge detection from spectral data (ACHA 1999, SIAM J. Numer. Anal. 2000).

In our paper, "Using Fourier Series and Machine Learning to Classify Signals," we demonstrate that explicitly incorporating jump discontinuity information (extracted via your concentration factor method) as features for neural network classification significantly improves accuracy compared to using raw signal data or Fourier coefficients alone.

Key findings:
- Model C (Fourier + jump features) achieves 95.4% accuracy vs 93.0% for raw data
- The improvement is most pronounced for signals with discontinuities
- Jump information compensates for limited frequency resolution at low mode counts

The manuscript and reproducible code are available at:
- Paper: [attached/link]
- Code: https://github.com/abbass12/FourierSeriesClassification

I would greatly appreciate any feedback on the mathematical rigor of our approach, particularly regarding the application of concentration factors in this ML context.

Thank you for your time and for the foundational work that made this research possible.

Best regards,
Abbass Srour
University of Michigan
abbasss@umich.edu
```

### For Journal Submission Cover Letter

```
Dear Editor,

We are pleased to submit our manuscript entitled "Using Fourier Series and Machine Learning to Classify Signals" for consideration in Mathematics.

This paper investigates whether Fourier series coefficients, combined with explicit jump discontinuity information extracted via concentration factors, can improve neural network-based signal classification. Our key contribution is the novel integration of the Gelb-Tadmor edge detection method as a feature extraction step for machine learning, demonstrating that mathematical properties of Fourier series can be leveraged to enhance classification performance.

The manuscript includes:
- Complete mathematical framework for the approach
- Reproducible experimental results (code available on GitHub)
- Interactive demonstrations via Google Colab

This work has not been published elsewhere and is not under consideration by another journal.

Sincerely,
Abbass Srour
```

## Submission Checklist

- [x] Codebase modernized and pushed to GitHub
- [x] Experiments replicated with reproducible results
- [x] Figures generated at publication quality
- [x] LaTeX paper written in MDPI format
- [x] References verified (DOIs added, years corrected)
- [x] Interactive notebook created for reviewers
- [ ] Run experiments on Google Colab with GPU for larger dataset (1000+ samples per type)
- [ ] Create MDPI account and submit manuscript
- [ ] Send feedback request emails to Gelb and Tadmor
- [ ] Contact UMich professors for potential collaboration/endorsement

## Timeline

| Week | Action |
|------|--------|
| Week 1 | Run full experiments on Colab GPU, finalize paper |
| Week 2 | Send outreach emails, incorporate any quick feedback |
| Week 3 | Submit to MDPI Mathematics |
| Week 4+ | Respond to reviewer comments (expected 30-45 day review) |

## Notes on Strengthening the Paper

1. **Increase dataset size**: Run with 2000+ samples per type on Colab GPU
2. **Add real-world signal test**: Consider ECG or seismic data as validation
3. **Statistical significance**: Add confidence intervals and p-values
4. **Ablation study**: Test different concentration factor types systematically
5. **Comparison with CNNs**: Add a CNN baseline for completeness
