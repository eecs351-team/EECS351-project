---
title: DCT
parent: Methods
nav_order: 2
---

# 2D Discrete Cosine Transform (DCT)
{: .no_toc }

A real cosine basis with strong energy compaction, the same basis family used by JPEG.
{: .lead }

## Definition

We use the orthonormal type-II DCT applied along both image axes:

$$
C[k,\ell] = \alpha_k \alpha_\ell \sum_{m=0}^{H-1} \sum_{n=0}^{W-1} x[m,n]
\cos\!\left( \frac{\pi k (2m+1)}{2H} \right) \cos\!\left( \frac{\pi \ell (2n+1)}{2W} \right),
$$

where $$\alpha_0 = \sqrt{1/N}$$ and $$\alpha_k = \sqrt{2/N}$$ for $$k \geq 1$$ along an axis of
length $$N$$. Coefficient $$(k,\ell)$$ corresponds to a cosine basis function with frequencies
$$k/(2H)$$ and $$\ell/(2W)$$ cycles/pixel. $$C[0,0]$$ is the DC term (the average level).

## Features

Each AC coefficient is assigned a basis radius

$$
r = \sqrt{\left(\frac{k}{2H}\right)^2 + \left(\frac{\ell}{2W}\right)^2}\ \text{cycles/pixel}.
$$

The DC coefficient is removed, and energies are divided by the total AC energy.

| Column | Meaning |
| --- | --- |
| `dct_low_energy_ratio` | Share of AC energy with $$r < 0.15$$ |
| `dct_high_energy_ratio` | Share of AC energy with $$r \geq 0.15$$ |
| `dct_ac_concentration` | $$\sum_i p_i^2$$, where $$p_i$$ are the normalized AC coefficient energies |

`dct_ac_concentration` is near 1 when the AC energy is concentrated in a few cosine
coefficients; smaller values indicate more distributed energy. Because the FFT is
unnormalized and the DCT is orthonormal, their raw energy totals should not be compared
without normalization.

## Example

{% include figure.html src="/docs/figures/transforms.png" alt="CIFAKE REAL and FAKE grayscale examples with centered FFT, DCT, and two-level DWT visualizations" caption="**Figure 3.** The DCT column shows `log(1 + |C[k,l]|)` for one REAL and one FAKE CIFAKE image. Low frequencies are in the upper-left corner." %}
