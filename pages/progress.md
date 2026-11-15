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
| Project topic finalized | Done | AI-generated image detection with FFT, DCT, and DWT features and a simple classifier |
| Dataset: GenImage | Dropped | The split ZIP archive could not be reassembled into a usable file |
| Dataset: CIFAKE | Done | Small-image baseline at native 32x32 |
| Prototype FFT / DCT / DWT plots | Done | First grayscale FFT magnitude plots; the early DWT used a 1D decomposition |
| Reproducible DSP pipeline | Done | Shared 11-feature extractor, balanced seeded sampling, CSV exports, figures, metadata |
| Experiment 1: CIFAKE | Done | 1,000 per class train, 200 per class test |
| Experiment 2: RajuFace | Done | 10 per class demo, then 200 per class test-split analysis at 256x256 |
| Validation tests | Done | 28 offline tests pass |
| Logistic Regression | Planned | Training-only scaling, held-out evaluation |
| SVM and other classifiers | Planned | If time permits |
| Generalization and robustness | Planned | Unseen generators, cross-dataset evaluation, compression and resizing |

## What changed from the proposal

- **Dataset.** GenImage was replaced by CIFAKE, and RajuFace (140k Real and Fake Faces)
  was added to test the same features at a higher resolution and on a GAN generator.
- **DWT.** The prototype's 1D `wavedec` at level 4 was replaced by a true 2D `wavedec2` at
  level 2, the maximum useful level for 32x32 images.
- **Image loading.** Pillow replaced OpenCV for loading and grayscale conversion, which avoids a
  larger dependency.
- **Display.** The FFT magnitude now uses a safe `log1p` for display instead of `log`, which
  fails on zero-magnitude coefficients.

## Next steps

1. Train Logistic Regression on the CIFAKE training features and evaluate on the held-out test
   features.
2. Compare feature groups (FFT, DCT, DWT) and add an SVM comparison if time permits.
3. Evaluate on RajuFace and other unseen data; study sensitivity to compression and resizing.
4. Run larger RajuFace samples (the optional 1,000-per-class experiment has not been run).

## Limitations

- CIFAKE's 32x32 resolution and wavelet boundary effects limit fine-detail analysis.
- Handcrafted features may capture compression, content, or generator-specific artifacts.
  Normalized frequency bands do not remove resolution dependence.
- RajuFace sampling can reflect source ordering; larger and more diverse samples are needed.
- Overlapping distributions do not establish detection accuracy.
- Downsampling RajuFace to 32x32 for CIFAKE model compatibility removes detail and belongs in a
  separate experiment from native-resolution analysis.
