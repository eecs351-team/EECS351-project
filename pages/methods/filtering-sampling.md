---
title: Filtering and sampling
parent: Methods
nav_order: 4
---

# Filtering and sampling demonstration
{: .no_toc }

Linear filtering and aliasing shown on a synthetic signal.
{: .lead }

{: .note }
This experiment is explicitly **synthetic**. It does not modify the dataset images used for
feature extraction and does not add features.

## Setup

- **Signal:** gratings at 0.08 and 0.40 cycles/pixel.
- **Low-pass filter:** separable 25-tap Hamming-windowed finite impulse response (FIR) filter with cutoff 0.20 cycles/pixel.
- **High-pass residual:** $$I - \text{lowpass}(I)$$.
- **Sampling:** decimation by 2, with and without prefiltering.

## What it shows

**Linear filtering.** Spatial convolution corresponds to multiplying the spectrum by the filter's
frequency response, away from reflected boundaries. The low-pass output keeps the 0.08 grating;
the residual keeps the 0.40 grating.

**Sampling and aliasing.** Decimating by 2 lowers the Nyquist frequency to 0.25 cycles per
original pixel. Without prefiltering, the 0.40 component aliases to 0.20 cycles per new pixel.
The finite FIR filter attenuates content above the new Nyquist before decimation, with a nonzero
transition band and boundary effects.

{% include figure.html src="/docs/figures/filtering_sampling.png" alt="Synthetic grating filtered by an FIR low-pass filter, its high-pass residual, and decimation with and without prefiltering" caption="**Figure 1.** Synthetic filtering and sampling demonstration." %}
