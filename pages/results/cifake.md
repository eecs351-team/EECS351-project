---
title: "CIFAKE (method check)"
parent: Results
nav_order: 2
---

# CIFAKE (method check)
{: .no_toc }

1,000 REAL + 1,000 FAKE training images at native 32x32 resolution.
{: .lead }

{: .note }
This dataset was mainly used to check that our feature extraction works. **Our current focus is face images**; see [Faces: frequency analysis](rajuface.md).

We processed **1,000 REAL + 1,000 FAKE training images** and exported features for
**200 REAL + 200 FAKE test images**. The comparisons below use training data only. CIFAKE
was not used for classification.

## Transform examples

{% include figure.html src="/docs/figures/transforms.png" alt="CIFAKE REAL and FAKE grayscale examples with centered FFT, DCT, and two-level DWT visualizations" caption="**Figure 1.** First valid selected training image per class. Color limits are shared within each transform column. These images illustrate the representations; they were not chosen to show differences." %}

## Feature statistics

| Feature                         | REAL (n=1,000)      | FAKE (n=1,000)      |
| ------------------------------- | ------------------- | ------------------- |
| Fourier (FFT) high-frequency energy share | 0.03065 +/- 0.02089 | 0.04441 +/- 0.02464 |
| Cosine (DCT) high-frequency energy share | 0.16170 +/- 0.08762 | 0.21359 +/- 0.10611 |
| Wavelet (DWT) finest-scale detail share | 0.18446 +/- 0.06100 | 0.24996 +/- 0.08083 |

{% include figure.html src="/docs/figures/train_distributions.png" alt="CIFAKE training distributions of FFT high-frequency, DCT high-frequency, and fine-scale wavelet energy ratios" caption="**Figure 2.** Common bins and per-class normalization. FAKE means are higher for these three features in this training subset, with substantial overlap." %}

## Data files

- [All 11 feature summaries (CSV)](../../docs/results/train_summary.csv)
- [Counts and configuration (JSON)](../../docs/results/metadata.json)
- [CIFAKE technical guide](../../docs/task1.md)
