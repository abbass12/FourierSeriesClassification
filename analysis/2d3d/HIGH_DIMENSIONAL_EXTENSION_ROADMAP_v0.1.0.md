# High-Dimensional Extension Roadmap v0.1.0

## Recommendation

The extension is mathematically natural and potentially **more publishable** than the current one-dimensional feature-vector approach. It should be pursued as a staged project: establish a rigorous two-dimensional method first, then advance to three-dimensional volumes only after the two-dimensional evidence and software are sound.

The supplied graphic is best interpreted as a sampled scalar field

\[
z=f(x,y), \qquad (x,y)\in\Omega\subset\mathbb{R}^2,
\]

rendered as a three-dimensional height surface. Thus it is a **2.5D visualization of a 2D signal**, not yet a 3D volume. The highlighted red curve appears to mark a candidate discontinuity or ridge. This is exactly the right intermediate setting: the discontinuity set of a 2D field is a curve, whereas the discontinuity set of a 3D field is generally a surface.

> Do not carry the current one-dimensional “top four peaks plus zero padding” representation into 2D or 3D. In higher dimensions, the mathematically meaningful object is an **edge map or surface response field**, which should be preserved spatially and given to a CNN rather than collapsed into a small unordered vector.

## 1. Mathematical Extension

### From 1D jumps to 2D edge curves

For a periodic scalar image or height field \(f(x,y)\), define the two-dimensional Fourier coefficients

\[
\widehat f_{k,\ell}=
\frac{1}{(2\pi)^2}\int_{[-\pi,\pi]^2}f(x,y)e^{-i(kx+\ell y)}\,dx\,dy,
\]

and its truncated spectral reconstruction

\[
S_N f(x,y)=\sum_{|k|,|\ell|\le N}\widehat f_{k,\ell}e^{i(kx+\ell y)}.
\]

The 1D jump location is replaced by an edge set \(\Gamma\), typically a collection of curves. A practical initial concentration-factor implementation should construct directional edge responses along the coordinate axes,

\[
E_x^\sigma f(x,y)=i\sum_{k,\ell}\widehat f_{k,\ell}\,\operatorname{sgn}(k)\,\sigma\!\left(\frac{|k|}{N}\right)e^{i(kx+\ell y)},
\]

with an analogous \(E_y^\sigma\). Their magnitude \(\sqrt{|E_x^\sigma|^2+|E_y^\sigma|^2}\) is an initial edge-map channel. A subsequent directional or wavefront formulation can improve orientation sensitivity. Previous work has demonstrated image segmentation directly from Fourier spectral data through an edge map, avoiding an intervening pixel-space reconstruction for the segmentation step.[1]

### From 2D edge curves to 3D discontinuity surfaces

For a volume \(f(x,y,z)\), the 3D FFT coefficients are \(\widehat f_{k,\ell,m}\). The discontinuity set becomes a two-dimensional surface \(\Gamma\subset\mathbb{R}^3\). The first practical 3D method should compute separable concentration-factor responses \((E_x,E_y,E_z)\), retain all three components or their magnitude as volume channels, and classify the resulting volume with a small 3D CNN. It should not begin with full geometric surface reconstruction, which would add a difficult segmentation problem before the classification hypothesis is even tested.

## 2. The Research Question That Can Survive Negative Results

The proposed question is:

> **When does a spatially preserved concentration-factor edge map provide discriminative information beyond raw samples, FFT magnitude/phase, and ordinary gradient maps in 2D images or 3D volumes?**

This is stronger than claiming that Fourier edges will improve every classifier. It permits useful outcomes in every direction: an improvement in boundary-determined tasks, no gain where texture dominates, or evidence that standard spatial gradients outperform the spectral estimate under noise or limited resolution.

## 3. Required Comparisons

| Representation | 2D implementation | 3D implementation | Role |
|---|---|---|---|
| Raw input | Grayscale/RGB image to 2D CNN | Volume to 3D CNN | Primary learned baseline |
| Fourier-only | Low-frequency complex coefficient tile, or amplitude and phase channels | Low-frequency 3D coefficient block | Tests spectral representation independently |
| Gradient control | Sobel or Gaussian-gradient magnitude and orientation | 3D finite-difference gradient magnitude | Tests whether ordinary local edges explain results |
| Concentration-factor map | \(E_x,E_y,\|E\|\) as image channels | \(E_x,E_y,E_z,\|E\|\) as volume channels | Tests the mathematical construction |
| Raw + spectral edge map | Concatenate raw image and edge-map channels in one CNN | Concatenate raw volume and edge-map channels | Main practical method |
| Oracle edge map, synthetic only | Ground-truth boundary mask and normal direction | Ground-truth surface mask and normal direction | Separates information value from estimation error |

The most important comparison is **spectral edge map versus ordinary gradient map**. Without this control, an observed improvement cannot be attributed to the Fourier/concentration-factor method rather than the simple fact that boundaries matter.

## 4. 2D First: A Complete, Publishable Stage

### 4.1 Controlled synthetic data

Create labeled piecewise-smooth 2D fields on \([−\pi,\pi)^2\) with ground-truth boundaries. Examples should include disks, rectangles, rings, wedges, smoothly varying regions, oblique edges, curved boundaries, texture-only classes, and classes with identical textures but different boundary topology. Add controlled Gaussian noise, blur, missing Fourier modes, and spectral truncation.

This design yields three crucial tests.

| Test | Labels depend on | Expected diagnostic |
|---|---|---|
| Boundary-driven | Curve shape, orientation, number of regions | An edge map may add useful information |
| Texture-driven | Smooth frequency pattern, not boundaries | Edge map should not be expected to help |
| Oracle-versus-inferred | Same boundary labels, true mask available | Quantifies estimator loss from Fourier data to edge map |

Report classification accuracy and macro F1, but also report edge precision, recall, F1, and boundary localization error. The method needs to demonstrate that it computes a credible edge map before asking whether that map helps classification.

### 4.2 Public 2D benchmark

After the synthetic study, use a small, preselected panel from MedMNIST v2 or a non-medical vision benchmark. MedMNIST v2 provides standardized train-validation-test splits for 12 2D and 6 3D biomedical classification datasets, with the 2D images resized to 28×28 and 3D volumes to 28×28×28.[2]

For an initial 2D screen, use two grayscale datasets and one RGB dataset. The goal is methodological benchmarking, **not clinical prediction**, and claims must remain limited to the supplied benchmark labels.

Recommended 2D baseline set:

1. A compact 2D CNN trained on raw input.
2. A small ResNet-18, preferably trained from scratch on the standardized size.
3. The same CNN with Sobel-gradient channels.
4. The same CNN with concentration-factor edge-map channels.
5. The same CNN with both raw input and concentration-factor channels.
6. A Fourier-only classifier with a matched parameter budget.

## 5. Then 3D: A Feasible but Narrow First Stage

Start with **28×28×28 volumes**, not clinical-resolution CT or MRI. This keeps 3D FFTs and 3D CNNs feasible on Colab T4 GPUs and permits direct, reproducible comparisons. MedMNIST3D contains standardized volumes and fixed splits, including OrganMNIST3D, NoduleMNIST3D, AdrenalMNIST3D, FractureMNIST3D, VesselMNIST3D, and SynapseMNIST3D.[2]

The recommended initial 3D subset is one shape-oriented dataset, such as VesselMNIST3D or AdrenalMNIST3D, plus one intensity-volume dataset such as NoduleMNIST3D. This deliberately tests whether spectral surface information behaves differently when class structure is geometrical versus textural.

| Scope | Proposed architecture | Hardware practicality |
|---|---|---|
| 2D synthetic | 2D CNN with 1–4 input channels | Fast on CPU or Colab GPU |
| 2D public screen | Compact ResNet-18 and raw-plus-edge CNN | Fast on a T4 |
| 3D public pilot | Compact 3D CNN, 28³ inputs, batch size 8–32 | Practical on a T4 |
| 3D confirmatory study | 3D CNN/3D ResNet with 10 seeds across multiple volumes | Practical but staged GPU sessions required |
| Full-resolution volumetric imaging | Advanced 3D architectures and preprocessing | Do not begin here |

## 6. Recommended Code Architecture

Create a separate `src/highdim/` package rather than mixing the new work into the 1D signal modules.

```text
src/highdim/
├── fft_nd.py             # fft2/fftn feature extraction, masks, normalization
├── concentration_nd.py   # 2D/3D directional concentration-factor responses
├── synthetic_2d.py       # fields, boundary masks, known Fourier truncation
├── synthetic_3d.py       # optional volumes, surface masks, known geometry
├── datasets_medmnist.py  # controlled public-data loader with official splits
├── models_2d.py          # matched CNN/ResNet channel variants
├── models_3d.py          # compact 3D CNN and optional 3D ResNet
├── metrics_edges.py      # boundary F1, localization error, calibration
└── run_highdim.py        # frozen configuration and repeated-seed runner
```

The 1D code should remain as a reproducibility baseline. It is valuable as a testbed, not a burden.

## 7. Publication Strategy

Do **not** combine 1D, 2D, and 3D into a single first journal paper. It would likely become too broad and under-evaluated. Use a deliberate sequence.

| Paper or release | Scope | Strongest contribution |
|---|---|---|
| Current preprint/pilot | 1D synthetic plus narrow UCR screen | Reproducibility and falsification of the initial feature-vector claim |
| Main research paper | 2D controlled fields plus public image benchmarks | Spatial concentration-factor edge maps as classification channels; oracle controls and failure analysis |
| Follow-on paper | 3D volumes after the 2D method is validated | Spectral surface representations for volumetric classification |

A more mathematical venue becomes realistic only if you add a theorem or carefully controlled analysis of the 2D/3D estimator. An ML or scientific-computing venue is more appropriate for a multi-benchmark empirical paper with strong reproducibility but no new convergence theory.

## 8. First Concrete Milestone

The next implementation milestone should be a **2D synthetic edge-map benchmark**, not a 3D model:

1. Generate 64×64 and 128×128 fields with exact curve masks.
2. Compute `fft2` features and a separable trigonometric concentration-factor edge map.
3. Validate edge localization against the known masks.
4. Train the same compact 2D CNN in four conditions: raw, raw+Sobel, raw+spectral-edge, and raw+oracle-edge.
5. Repeat ten seeds across clean, noisy, blurred, and truncated-spectrum conditions.
6. Advance to a public 2D dataset only if the edge map is both numerically valid and scientifically interpretable.

This milestone gives a decisive answer: whether the spectral edge map carries useful classification information at all, and whether that information survives the move from a 1D fixed-length peak vector to a spatial representation.

## References

[1] Gelb, A.; Cates, D. *Segmentation of Images from Fourier Spectral Data*. Communications in Computational Physics **2009**, *5*, 326–349. https://doi.org/10.4208/cicp.2009.v5.p326

[2] Yang, J.; et al. *MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification*. Scientific Data **2023**, *10*, 41. https://pmc.ncbi.nlm.nih.gov/articles/PMC9852451/

[3] Cochran, D.; Gelb, A.; Wang, Y. *Edge Detection from Truncated Fourier Data Using Spectral Mollifiers*. Advances in Computational Mathematics **2013**, *38*, 737–762. https://doi.org/10.1007/s10444-011-9258-4

[4] Greengard, L.; Stucchio, C. *Spectral Edge Detection in Two Dimensions Using Wavefronts*. https://www.semanticscholar.org/paper/b39039c5b0e7f42125e91caa7eba8fb2f2b5683a
