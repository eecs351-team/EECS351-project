---
title: Home
layout: home
nav_order: 1
description: "EECS 351 project: detecting AI-generated face images with Fourier, cosine, and wavelet features."
permalink: /
---

# AI-Generated Image Detection Using DSP and Machine Learning
{: .fs-9 }

Can simple, interpretable frequency features tell real photographs from AI-generated images?
{: .fs-6 .fw-300 }

[Read the method](pages/methods/index.md){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[See the results](pages/results/index.md){: .btn .fs-5 .mb-4 .mb-md-0 }

**University of Michigan · EECS 351 · Fall 2026**<br>
Ethan Kocheril, Jiayuan Zhu, Zhirui Wang, Ziyi Xie

---

## Project overview

AI-generated images can look real to people, but they may still differ from photographs in how
their brightness changes from pixel to pixel: their **frequency content**. Our question is:
**can compact, interpretable digital signal processing (DSP) features help tell real images
from AI-generated ones?** We describe each image with three standard transforms, the Fast
Fourier Transform (FFT), the Discrete Cosine Transform (DCT), and the Discrete Wavelet Transform
(DWT), and then test with machine-learning classifiers whether those descriptions are enough to
detect generated images.

Our main dataset is **140k Real and Fake Faces**: real face photographs and faces made by
StyleGAN, an image generator.

{% include figure.html src="/docs/figures/faces/face_examples.png" alt="A real and a generated face in grayscale, each next to its Fourier power spectrum" caption="**Figure 1.** A real face (top) and a generated face (bottom), each shown in grayscale next to its Fourier power spectrum. The center of each spectrum is low frequency (smooth, slow changes); the edges are high frequency (fine detail). The two spectra look very similar, so the differences have to be measured, not seen." %}

{: .status }
**Done:** feature extraction with all three transforms; Logistic Regression and five other
classifiers on 10,000 training and 2,000 held-out test face images. The best result so far is
**74% test accuracy** (50% would be guessing).<br>
**Next:** set aside a final untouched test set, measure how compression and resizing affect
detection, try a noise-residual feature, and explain which frequencies the model relies on.
See [Progress](pages/progress.md).

## Grader cheat sheet

Where each course requirement is implemented, so it can be found without reading every page.

<!-- Team: confirm with the instructor how FFT/DCT/DWT count toward the in-class requirement
     before the final submission. We do not automatically count them as three separate tools. -->

| Requirement | What we use | Where |
| --- | --- | --- |
| Basis representations and frequency analysis | 2D Fast Fourier Transform, Discrete Cosine Transform, and Discrete Wavelet Transform features; a Hann window before the Fourier transform to reduce spectral leakage | [Methods](pages/methods/index.md), [Fourier](pages/methods/fft.md), [Cosine](pages/methods/dct.md), [Wavelet](pages/methods/dwt.md) |
| Linear filtering | Finite impulse response (FIR) low-pass filter and high-pass residual (synthetic demo); noise residual on face images (planned) | [Filtering and sampling](pages/methods/filtering-sampling.md) |
| Sampling and aliasing | Downsampling by 2 with and without an anti-aliasing filter (synthetic demo); resizing experiment on face images (planned) | [Filtering and sampling](pages/methods/filtering-sampling.md) |
| Out-of-class technique | Logistic Regression, Support Vector Machines, K-Nearest Neighbors, Naive Bayes, and Decision Tree classifiers | [Classification](pages/classification.md) |
{: .cheat-sheet }

## On this site

| Page | Contents |
| --- | --- |
| [Background](pages/background.md) | Why frequency content may reveal generated images; signal processing vs. machine learning |
| [Datasets](pages/datasets.md) | The face dataset we focus on, and CIFAKE, used to check the method |
| [Methods](pages/methods/index.md) | Preprocessing, the three transforms, and the features we extract |
| [Classification](pages/classification.md) | Classifiers, how we evaluate them, and their accuracy |
| [Results](pages/results/index.md) | What the frequency analysis shows for real and generated faces |
| [Progress](pages/progress.md) | What is done and what comes next |
| [Team](pages/team.md) | Members and contributions |
| [Reproducibility](pages/reproducibility.md) | How to get the datasets |
| [References](pages/references.md) | Papers and datasets |
