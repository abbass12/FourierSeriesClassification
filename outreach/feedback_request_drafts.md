# Feedback Request Drafts

**Status:** Drafts only. Do not send without the author’s final review and explicit approval.

## Contact verification

| Recipient | Why contact them | Official page used for verification | Public contact shown on official page |
|---|---|---|---|
| Anne E. Gelb, Dartmouth College | Co-author of the foundational concentration-factor edge-detection papers used in the project. Her profile lists numerical analysis, data-driven numerical simulations, compressive sensing, and signal-processing applications. | https://faculty-directory.dartmouth.edu/anne-e-gelb | Anne.E.Gelb@dartmouth.edu |
| Eitan Tadmor, University of Maryland | Co-author of the foundational concentration-factor papers; official profile lists spectral viscosity and processing discontinuous spectral data among research materials. | https://www.math.umd.edu/~tadmor/ | tadmor@umd.edu |
| Jeffrey A. Fessler, University of Michigan | Potential local methodological feedback on statistical signal processing, machine learning, optimization, inverse problems, and MRI. | https://medschool.umich.edu/profile/1686/jeffrey-fessler | fessler@umich.edu |

## Draft 1: Anne E. Gelb

**Subject:** Request for brief methodological feedback on concentration-factor features in a reproducible pilot study

Dear Professor Gelb,

I am an independent researcher and University of Michigan alumnus preparing a reproducible pilot study that uses concentration-factor edge descriptors as machine-learning features for synthetic signal classification. The feature construction follows the generalized conjugate partial Fourier-sum framework in your work with Professor Tadmor, especially *Detection of Edges in Spectral Data* (1999) and its nonlinear-enhancement follow-up (2000).

The attached or linked draft does not claim that these descriptors improve classification. In a small repeated-seed smoke study, the current Fourier-plus-jump implementation did not outperform raw samples or Fourier coefficients alone. I am therefore revising the study around reproducibility and a planned confirmatory evaluation rather than a performance claim.

If you have time for a brief response, I would especially value your view on two questions:

1. Is it mathematically reasonable to use peak locations and signed concentration-factor responses as fixed-length features after truncation and zero padding?
2. Are there concentration-factor choices, regularization steps, or edge-recovery references that would be essential before testing this idea on noisy sampled data?

The code and transparent pilot outputs are available at:
https://github.com/abbass12/FourierSeriesClassification

I understand that you may not have time to review unsolicited work. Thank you for considering the request and for your foundational contributions to spectral edge detection.

Sincerely,

Abbass Srour
University of Michigan
abbasss@umich.edu

## Draft 2: Eitan Tadmor

**Subject:** Methodological question on concentration-factor edge descriptors for signal classification

Dear Professor Tadmor,

I am preparing a reproducible pilot study that evaluates raw samples, truncated Fourier coefficients, and concentration-factor edge descriptors for synthetic one-dimensional signal classification. The jump-descriptor construction is based on the generalized conjugate partial Fourier-sum framework developed in your work with Professor Gelb.

A small repeated-seed smoke experiment did not confirm an earlier single-run indication of a classification benefit from the inferred jump descriptors. Rather than overstate that result, I am revising the work to report it transparently and to define a confirmatory experiment with repeated seeds, ablations, and public benchmarks.

If you have a few minutes, I would be grateful for a short methodological comment on whether the following feature construction is defensible: detect high-magnitude peaks of the concentration-factor response; retain up to four locations and signed magnitudes; zero-pad to a fixed-length vector; and pass that vector to a classifier jointly with Fourier coefficients.

The pre-submission draft and code are available at:
https://github.com/abbass12/FourierSeriesClassification

Thank you for your time and for your influential work on spectral methods and discontinuous data.

Sincerely,

Abbass Srour
University of Michigan
abbasss@umich.edu

## Draft 3: Jeffrey A. Fessler

**Subject:** Request for brief methodological feedback on a reproducible Fourier-feature signal-classification pilot

Dear Professor Fessler,

I am seeking brief methodological feedback on a reproducible pilot project that compares raw signal samples, truncated Fourier coefficients, and concentration-factor edge descriptors for synthetic one-dimensional signal classification.

The project currently has a transparent but limited result: after correcting the split and checkpointing protocol, a three-seed smoke run does not support an accuracy benefit from the jump-feature model. I am designing the next stage as a repeated-seed study with stronger baselines and benchmark data, rather than submitting the preliminary model comparison as a general result.

Given your work in statistical signal processing, machine learning, optimization, inverse problems, and MRI, I would appreciate any concise advice on the most important methodological changes before external review. In particular, I would value guidance on benchmark selection and how to evaluate whether a hand-engineered spectral representation adds value beyond standard one-dimensional CNN or random-kernel baselines.

The repository is available at:
https://github.com/abbass12/FourierSeriesClassification

I recognize that you may not have time to review unsolicited work. Thank you for considering this request.

Sincerely,

Abbass Srour
University of Michigan
abbasss@umich.edu
