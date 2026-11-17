---
title: Results
nav_order: 6
has_children: true
has_toc: false
---

# Results and discussion
{: .no_toc }

What the frequency analysis shows for real and generated faces.
{: .lead }

| Experiment | Data | Page |
| --- | --- | --- |
| **Faces (main)** | 140k Real and Fake Faces, 256x256, StyleGAN; 5,000 training images per class | [Face frequency analysis](rajuface.md) |
| Method check | CIFAKE, 32x32, Stable Diffusion 1.4; 1,000 training images per class | [CIFAKE](cifake.md) |

Classifier accuracy is on the [Classification](../classification.md) page.

## Main findings so far

1. **Real and generated faces have very similar spectra.** Their average Fourier spectra show
   no obvious generator "fingerprint", such as a grid of bright spots. Generated faces have
   slightly less fine detail on average.
2. **The difference is in the shape of the spectrum.** Features that keep the shape of the
   spectrum detect generated faces much better than three wide frequency bands: 68.2% vs. 55.5%
   test accuracy with Logistic Regression, and up to **74.2%** with a radial basis function
   Support Vector Machine.
3. **Careful signal processing matters.** A Hann window before the Fourier transform removes
   edge artifacts (spectral leakage) and raised accuracy by about 3 points.
4. **The differences are concentrated in the middle of the face.** The eye row and the
   mouth-and-chin rows differ most; the background corners differ least.

## Why accuracy is limited

Published frequency-based detectors report much higher accuracy on full-resolution generated
faces [[2]](../references.md). In
this dataset, every image was shrunk to 256x256 and saved as JPEG before we received it.
Shrinking acts as a low-pass filter and removes the finest detail, which is where generator
traces are usually reported. This is our main explanation for the modest accuracy, and we
plan to test it with the resizing experiment described on the [Progress](../progress.md) page.
