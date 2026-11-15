---
title: "Experiment 2: RajuFace"
parent: Results
nav_order: 2
---

# Experiment 2: RajuFace
{: .no_toc }

200 REAL + 200 FAKE test-split faces at native 256x256 resolution.
{: .lead }

A **10 REAL + 10 FAKE test-split demo** passed before the initial
**200 REAL + 200 FAKE test-split analysis**. Images remained 256x256. The demo and initial
samples overlap and are not independent experiments.

## Transform examples

{% include figure.html src="/docs/figures/rajuface_transforms.png" alt="RajuFace REAL and FAKE original RGB, grayscale, FFT, DCT, and DWT examples at native 256 by 256 resolution" caption="**Figure 9.** First valid image per class after seeded buffer sampling: test indices 28 (REAL) and 57 (FAKE). They were not selected to maximize visible differences." %}

## Feature statistics

| Feature                         | REAL (n=200)        | FAKE (n=200)        |
| ------------------------------- | ------------------- | ------------------- |
| FFT high-frequency energy ratio | 0.01181 +/- 0.00707 | 0.01085 +/- 0.00739 |
| DCT high-frequency energy ratio | 0.03140 +/- 0.01750 | 0.02915 +/- 0.01780 |
| DWT level-1 detail-energy ratio | 0.40652 +/- 0.06589 | 0.39717 +/- 0.06579 |

{% include figure.html src="/docs/figures/rajuface_distributions.png" alt="RajuFace sampled test distributions of FFT high-frequency, DCT high-frequency, and fine-scale wavelet energy ratios with 200 images per class" caption="**Figure 10.** REAL means are slightly higher for these features in this subset, and the distributions overlap substantially. These are exploratory test-subset statistics; no model or thresholds were tuned on these data." %}

## Data files

- [All 11 feature summaries (CSV)](../../docs/results/rajuface_test_summary.csv)
- [Schema, commit, counts, and configuration (JSON)](../../docs/results/rajuface_metadata.json)
- [RajuFace technical guide](../../docs/rajuface.md)
