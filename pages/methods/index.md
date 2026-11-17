---
title: Methods
nav_order: 4
has_children: true
has_toc: false
---

# Methods
{: .no_toc }

How each image becomes a set of numbers, and how those numbers become a decision.
{: .lead }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

## Pipeline

```text
Face image (JPEG)
  -> grayscale, values in [0, 1]
  -> Fourier / cosine / wavelet transforms
  -> features (11 basic numbers, plus finer frequency features)
  -> classifier trained on training images
  -> accuracy on held-out test images
```

## Preprocessing

Each image is converted to grayscale (brightness = 0.299 R + 0.587 G + 0.114 B) and scaled to
values between 0 and 1. Real and fake images are processed in exactly the same way. We do not
resize, sharpen, denoise, or crop, because any of these would change the frequency content we
want to measure.

## Basic features: 11 energy shares

For each transform we measure what **share** of the image's energy falls into each frequency
band. Shares, rather than raw amounts, keep the features comparable between bright and dark
images.

| Transform | What it measures | Features | Details |
| --- | --- | --- | --- |
| **Fast Fourier Transform (FFT)** | Energy at each spatial frequency | Share of energy at low, middle, and high frequencies (boundaries 0.15 and 0.35 cycles/pixel) | [Fourier](fft.md) |
| **Discrete Cosine Transform (DCT)** | Energy in a cosine basis, as in JPEG | Share of energy at low and high frequencies (boundary 0.15 cycles/pixel); how concentrated the energy is | [Cosine](dct.md) |
| **Discrete Wavelet Transform (DWT)** | Detail by direction and by scale | Share of detail energy that is horizontal, vertical, or diagonal; share at the fine and the coarse scale (Daubechies-4 wavelet, 2 levels) | [Wavelet](dwt.md) |

The zero-frequency term (the image's average brightness, called DC) is left out. Frequencies
are in cycles per pixel; 0.5 is the highest frequency a pixel grid can show.

## Finer frequency features (faces)

Three wide bands throw away the shape of the spectrum. For the face images we also keep
more detail:

| Feature | Size | What it is |
| --- | --- | --- |
| Radial Fourier spectrum | 127 numbers | Average Fourier power in thin rings around the center, one ring per frequency step [[2]](../references.md) |
| Cosine blocks | 1,024 numbers | The logarithm of cosine-transform magnitudes, averaged in 8x8 groups of coefficients [[8]](../references.md) |
| Region features | 32 numbers | The high-frequency shares above, computed separately on a 4x4 grid of face regions |
| Combined | 1,194 numbers | All of the above together with the 11 basic features |

**Hann window.** The Fourier transform treats an image as if it repeated forever, so a
mismatch between the left and right (or top and bottom) edges acts like a sharp jump and
spreads false energy along the frequency axes. This is called **spectral leakage**.
Multiplying the image by a Hann window, which fades smoothly to zero at the edges, removes it.
On the face data this raised the accuracy of the radial spectrum from 60.0% to 63.2%.

## Classifiers

The features go into a classifier that outputs REAL or FAKE. We start with **Logistic
Regression**, a linear model whose weights show which features push a decision toward FAKE,
and then compare **Support Vector Machines**, **K-Nearest Neighbors**, **Naive Bayes**, and a
**Decision Tree**. All settings are chosen using training images only; test images are used
once, at the end. See [Classification](../classification.md).

## Connection to EECS 351

| Course concept | Implemented work |
| --- | --- |
| Basis representations and frequency analysis | Fourier, cosine, and wavelet transforms and their energy features; Hann window against spectral leakage |
| Linear filtering | Finite impulse response (FIR) low-pass filter and high-pass residual on a synthetic signal ([demo](filtering-sampling.md)); noise residual on faces planned |
| Sampling and aliasing | Downsampling by 2 with and without an anti-aliasing filter on a synthetic signal ([demo](filtering-sampling.md)); resizing experiment on faces planned |
| Out-of-class technique | Logistic Regression, Support Vector Machines, K-Nearest Neighbors, Naive Bayes, Decision Tree |
