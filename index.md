---
title: Home
layout: home
nav_order: 1
description: "EECS 351 project: AI-generated image detection with FFT, DCT, and DWT features."
permalink: /
---

# AI-Generated Image Detection Using DSP and Machine Learning
{: .fs-9 }

Can compact, interpretable frequency-domain features tell real photographs from AI-generated images?
{: .fs-6 .fw-300 }

[Read the method](pages/methods/index.md){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[See the results](pages/results/index.md){: .btn .fs-5 .mb-4 .mb-md-0 }

**University of Michigan · EECS 351 · Fall 2026**<br>
Ethan Kocheril, Jiayuan Zhu, Zhirui Wang, Ziyi Xie

---

## Project overview

Real photographs and generated images may look convincing while differing in frequency
and scale statistics. Our research question is: **can compact, interpretable DSP features
help distinguish real and AI-generated images across datasets and generators?** We examine
Fourier, cosine, and wavelet representations before testing their usefulness with machine
learning.

{% include figure.html src="/docs/figures/transforms.png" alt="CIFAKE REAL and FAKE grayscale examples with centered FFT, DCT, and two-level DWT visualizations" caption="**Figure 1.** One REAL and one FAKE CIFAKE training image with their centered FFT, DCT, and two-level DWT representations. Color limits are shared within each transform column." %}

{: .status }
**Completed:** reproducible DSP extraction, balanced sampling, CSV exports, and REAL/FAKE
comparisons on CIFAKE and RajuFace.<br>
**Planned:** Logistic Regression, a possible SVM comparison, and classifier generalization
tests. No trained model or classification performance is available yet.

## Grader cheat sheet

Where each course requirement is implemented, so it can be found without reading every page.

<!-- Team: confirm with the instructor how FFT/DCT/DWT count toward the in-class requirement
     before the final submission. We do not automatically count them as three separate tools. -->

| Requirement | What we use | Where |
| --- | --- | --- |
| Basis representations and frequency analysis | 2D FFT, 2D DCT, and 2D DWT energy features | [FFT](pages/methods/fft.md), [DCT](pages/methods/dct.md), [DWT](pages/methods/dwt.md) |
| Linear filtering | Separable 25-tap FIR low-pass filter and high-pass residual (synthetic demo) | [Filtering and sampling](pages/methods/filtering-sampling.md) |
| Sampling and aliasing | Downsampling by 2 with and without prefiltering (synthetic demo) | [Filtering and sampling](pages/methods/filtering-sampling.md) |
| Out-of-class technique | Logistic Regression classifier (planned) | [Classification](pages/classification.md) |
{: .cheat-sheet }

## On this site

| Page | Contents |
| --- | --- |
| [Background](pages/background.md) | Why frequency statistics may reveal generated images; DSP vs. machine learning |
| [Datasets](pages/datasets.md) | CIFAKE and RajuFace sources, sizes, and splits |
| [Methods](pages/methods/index.md) | Preprocessing, the FFT/DCT/DWT feature extractor, filtering and sampling |
| [Classification](pages/classification.md) | Planned classifiers and evaluation metrics |
| [Results](pages/results/index.md) | Feature statistics for both experiments and their interpretation |
| [Progress](pages/progress.md) | What is done, what changed, and what comes next |
| [Team](pages/team.md) | Members and contributions |
| [Reproducibility](pages/reproducibility.md) | Installation, commands, and technical guides |
| [References](pages/references.md) | Papers and datasets |
