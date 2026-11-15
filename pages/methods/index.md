---
title: Methods
nav_order: 4
has_children: true
has_toc: false
---

# Methods
{: .no_toc }

One shared preprocessing step and one shared feature extractor for both datasets.
{: .lead }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

## Pipeline

```text
Images -> Grayscale [0,1] -> FFT / DCT / DWT -> 11 numerical features
                                               v
                                  Class statistics and figures
                                               v
                              Classifier and evaluation (planned)
```

## Preprocessing

Pillow converts RGB to 8-bit `L` grayscale (ITU-R 601 luminance, approximately
0.299 R + 0.587 G + 0.114 B), followed by float64 conversion and division by 255.
Images retain their native dimensions, with identical preprocessing for REAL and FAKE.
There is no resizing, sharpening, denoising, or per-image standardization.

## Feature extractor

The shared extractor returns the same **11 finite, label-independent feature columns** for
both datasets. Features were defined before looking at test statistics. All energy features
use **squared** coefficient magnitudes; `log(1 + |coefficient|)` is used for visualization only.

| Transform | Purpose | Features | Details |
| --- | --- | --- | --- |
| **2D FFT** | Express spatial variation as complex sinusoids | Low / mid / high radial energy shares, excluding DC, at boundaries 0.15 and 0.35 cycles/pixel | [FFT](fft.md) |
| **2D DCT** | Express the image in an orthonormal type-II cosine basis | Low / high AC energy shares at 0.15 cycles/pixel; AC energy concentration | [DCT](dct.md) |
| **2D DWT** | Separate detail by orientation and scale | Horizontal / vertical / diagonal shares and two scale shares of total detail energy (`db4`, level 2, symmetric extension) | [DWT](dwt.md) |

Some columns are algebraically dependent (for example, the three FFT ratios sum to 1). This is
an interpretable summary schema, not feature selection for a classifier. Normalized frequency
bands keep the definitions consistent across resolutions but do not make the feature
distributions resolution-invariant.

A separate [filtering and sampling demonstration](filtering-sampling.md) covers linear
filtering and aliasing on a synthetic signal; it does not modify dataset images or add features.

## Connection to EECS 351

| Course concept | Implemented work |
| --- | --- |
| Basis representations and frequency analysis | FFT/DCT/DWT image analysis and energy features |
| Linear filtering | A separable 25-tap FIR low-pass filter and high-pass residual on a synthetic signal |
| Sampling and aliasing | The same synthetic signal downsampled by 2 with and without prefiltering |
| Out-of-class technique | Logistic Regression (planned; not yet trained or evaluated) |
