---
title: Progress
nav_order: 7
---

# Progress
{: .no_toc }

What is done, what changed along the way, and what comes next.
{: .lead }

<!-- Add dates to each milestone as they are confirmed. -->

## Milestones

| Stage | Status | Notes |
| --- | --- | --- |
| Project topic finalized | Done | Detect AI-generated images with Fourier, cosine, and wavelet features and a simple classifier |
| Dataset: GenImage | Dropped | Its split ZIP archive could not be reassembled into a usable file |
| Dataset: CIFAKE | Done | Used to check the feature extraction on small 32x32 images |
| Dataset: 140k Real and Fake Faces | Done | Main dataset: 256x256 real and StyleGAN-generated faces |
| Feature extraction (Task 1) | Done | 11 basic energy-share features from the Fast Fourier, Discrete Cosine, and Discrete Wavelet Transforms |
| Logistic Regression (Task 2) | Done | Basic features by transform (up to 55.5%), finer frequency features (68.2%), Hann window |
| Other classifiers (Task 3) | Done | Support Vector Machines, K-Nearest Neighbors, Naive Bayes, Decision Tree; best 74.2% |
| Final untouched test set | Next | See below |
| Compression and resizing tests | Next | See below |
| Noise-residual feature | Next | See below |
| Explaining the model | Next | See below |

## What changed from the proposal

- **Dataset.** GenImage was replaced by CIFAKE, and then by 140k Real and Fake Faces as the main
  dataset. CIFAKE's 32x32 images are too small for fine-detail analysis, so it now serves only as
  a check of the method.
- **Finer features.** Three wide frequency bands per transform turned out to be too coarse. We
  added features that keep the shape of the spectrum (see [Methods](methods/index.md)).
- **Hann window.** We found that edge effects in the Fourier transform (spectral leakage) were
  hiding real differences, and added a window before the transform.
- **Wavelet transform.** The prototype's one-dimensional wavelet decomposition was replaced by a
  true two-dimensional one.

## Next steps

1. **Set aside a final test set.** We have looked at our current test images many times while
   making choices, so they are no longer truly unseen. We will keep a second, untouched set of
   10,000 test images from the same dataset for one final evaluation, and report every accuracy
   with a confidence range.
2. **Test compression and resizing.** We will re-save images with stronger JPEG compression and
   shrink them to 128x128 and 64x64, once with a low-pass (anti-aliasing) filter before
   shrinking and once without. This shows how robust the detector is, and tests our explanation
   that the dataset's own resizing removed most generator traces. It also applies the course's
   sampling and aliasing topic to real images.
3. **Try a noise-residual feature.** Subtracting a smoothed (low-pass filtered) copy of each
   image leaves its fine noise. Camera photographs contain sensor noise that generated images
   may lack, so the residual's frequency content could help. This applies the course's linear
   filtering topic.
4. **Explain the model.** Show which frequencies the Logistic Regression weights rely on, next
   to where the two classes differ most.
5. **Final evaluation and report.** Fix the final features and classifiers, evaluate them once
   on the untouched test set, and summarize what simple frequency features can and cannot do.

## Limitations

- The face images were shrunk to 256x256 and saved as JPEG by the dataset authors, which likely
  removed the finest generator traces.
- All results come from one dataset and one generator (StyleGAN), so they do not show how well
  the method works on other generators.
- Each test accuracy is uncertain by about ±2 percentage points.
- CIFAKE and the face dataset differ in content, resolution, and generator, so they cannot be
  compared directly.
