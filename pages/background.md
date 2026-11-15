---
title: Background
nav_order: 2
---

# Background
{: .no_toc }

Why look for generated images in the frequency domain, and where machine learning fits.
{: .lead }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

## Motivation

Generated images can look convincing to a human viewer while still differing from
photographs in how their energy is spread across spatial frequencies and scales.
Prior work has reported that CNN-based generators often fail to reproduce the spectral
distributions of real images, largely because of their up-sampling layers [[2]](references.md).
Rather than focusing on one specific frequency, we look at how image energy is distributed
across frequency ranges, orientations, and scales, and whether those distributions differ
consistently between real and AI-generated images.

We use three DSP representations:

- **Fast Fourier Transform (FFT):** energy at each spatial frequency over the whole image.
- **Discrete Cosine Transform (DCT):** a real cosine basis with strong energy compaction, the basis used by JPEG.
- **Discrete Wavelet Transform (DWT):** detail separated by orientation and by scale.

Comparing these representations gives a small set of statistical features. A simple
classifier then tests whether the features are actually useful for detection.

## DSP vs. machine learning

**Traditional DSP** lets us directly examine and understand the frequency characteristics of
images. The features are easy to interpret and to visualize, and domain knowledge guides
their choice, so they often need less labeled data and computation than complex
machine-learning models. However, FFT, DCT, and DWT are feature-extraction tools rather than
classifiers: the final decision still requires a rule or a classifier. Hand-chosen features
and thresholds can also be sensitive to noise, compression, and preprocessing.

**Machine learning** can learn how to combine many DSP-derived features, such as FFT band
energies, DCT coefficients, and wavelet statistics, instead of relying on one manually chosen
threshold. It also introduces risks. It needs representative labeled data and a careful
train/validation/test split; otherwise the model may overfit or exploit dataset-specific
shortcuts such as image size, compression, or object category. Its performance may drop on
unseen generators or after compression and resizing, and nonlinear models are often less
interpretable than the underlying features.

**Our approach** treats machine learning as a decision layer built on interpretable DSP
features, not as a replacement for signal analysis. Starting with a relatively simple
classifier keeps the focus on the signal-processing side while still testing whether the
features help in practice.
