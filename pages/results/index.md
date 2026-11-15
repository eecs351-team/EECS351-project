---
title: Results
nav_order: 6
has_children: true
has_toc: false
---

# Results and discussion
{: .no_toc }

Descriptive feature statistics for REAL and FAKE images in two experiments.
{: .lead }

Both experiments used seed **42**, balanced class counts, and **no skipped images**. Tables
show **mean +/- sample standard deviation**; ratios are fractions.

| Experiment | Data | Samples | Page |
| --- | --- | --- | --- |
| 1 | CIFAKE, 32x32, Stable Diffusion 1.4 | 1,000 REAL + 1,000 FAKE training images | [CIFAKE](cifake.md) |
| 2 | RajuFace, 256x256, StyleGAN | 200 REAL + 200 FAKE test-split images | [RajuFace](rajuface.md) |

## Summary of the three headline features

| Feature | CIFAKE REAL | CIFAKE FAKE | RajuFace REAL | RajuFace FAKE |
| --- | --- | --- | --- | --- |
| FFT high-frequency energy ratio | 0.03065 +/- 0.02089 | 0.04441 +/- 0.02464 | 0.01181 +/- 0.00707 | 0.01085 +/- 0.00739 |
| DCT high-frequency energy ratio | 0.16170 +/- 0.08762 | 0.21359 +/- 0.10611 | 0.03140 +/- 0.01750 | 0.02915 +/- 0.01780 |
| DWT level-1 detail-energy ratio | 0.18446 +/- 0.06100 | 0.24996 +/- 0.08083 | 0.40652 +/- 0.06589 | 0.39717 +/- 0.06579 |

## Interpretation

Individual transform plots can look similar because they show aggregate frequency structure on
a logarithmic scale. Subtle numerical differences may still help a classifier, but visual
similarity alone neither proves nor rules out useful classification.

On CIFAKE, FAKE means are higher for all three features; on RajuFace, REAL means are slightly
higher. The direction of the mean differences changes between datasets, and the datasets also
differ in content, resolution, and sources, so this does not isolate a generator effect. In
both experiments the class distributions overlap substantially.

{: .status }
**No accuracy, precision, recall, F1, or confusion matrix is available yet.** Held-out
classification is needed to evaluate feature usefulness; see [Classification](../classification.md).
