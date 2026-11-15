---
title: FFT
parent: Methods
nav_order: 1
---

# 2D Fast Fourier Transform (FFT)
{: .no_toc }

How much of an image's energy sits at low, middle, and high spatial frequencies.
{: .lead }

## Definition

For an $$H \times W$$ grayscale image $$x[m,n]$$, the 2D discrete Fourier transform is

$$
X[k,\ell] = \sum_{m=0}^{H-1} \sum_{n=0}^{W-1} x[m,n]\, e^{-j 2\pi \left( \frac{km}{H} + \frac{\ell n}{W} \right)} .
$$

Each coefficient pairs a vertical frequency $$f_k$$ with a horizontal frequency $$f_\ell$$,
both in cycles/pixel (NumPy `fftfreq`, range $$[-0.5, 0.5)$$). We use NumPy's default
unnormalized forward FFT and `fftshift` to center the spectrum for display.

## Features

Each coefficient is assigned to a band by its radial frequency

$$
r = \sqrt{f_k^2 + f_\ell^2}\ \text{cycles/pixel},
$$

with bands **low** $$[0, 0.15)$$, **mid** $$[0.15, 0.35)$$, and **high** $$[0.35, \infty)$$.
The DC term is removed, and each band energy is divided by the total non-DC energy:

$$
\text{ratio}_{\text{band}} = \frac{\sum_{(k,\ell) \in \text{band}} \lvert X[k,\ell] \rvert^2}{\sum_{(k,\ell) \neq (0,0)} \lvert X[k,\ell] \rvert^2} .
$$

| Column | Meaning |
| --- | --- |
| `fft_low_energy_ratio` | Share of non-DC energy below 0.15 cycles/pixel |
| `fft_mid_energy_ratio` | Share between 0.15 and 0.35 cycles/pixel |
| `fft_high_energy_ratio` | Share at or above 0.35 cycles/pixel |

The three ratios sum to 1 whenever the non-DC energy exceeds `1e-24`; otherwise all are zero
(for example, a constant image).

## Example

{% include figure.html src="/finalization-task-1.png" alt="Early FFT magnitude example of a CIFAKE image" caption="**Figure 2.** Early prototype output: a CIFAKE input image and its centered FFT magnitude. Energy concentrates near the center (low frequencies)." %}

The reproducible pipeline figures for both datasets are on the [Results](../results/index.md) pages.
