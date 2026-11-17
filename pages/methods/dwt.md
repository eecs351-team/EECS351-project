---
title: Wavelet transform (DWT)
parent: Methods
nav_order: 3
---

# 2D Discrete Wavelet Transform (DWT)
{: .no_toc }

Detail energy separated by orientation and by scale.
{: .lead }

## Definition

A one-level 2D DWT filters the image with a low-pass and a high-pass filter along each axis
and downsamples by 2. This produces an approximation band and three detail bands:

- `cH`: high-pass along array axis 0 (changes across rows)
- `cV`: high-pass along array axis 1 (changes across columns)
- `cD`: high-pass along both axes

Repeating the decomposition on the approximation band gives a coarser second level. We use
PyWavelets `wavedec2` with the **Daubechies `db4` wavelet, level 2, and symmetric extension**,
validated against `pywt.dwtn_max_level`. For 32x32 inputs, level 2 is the maximum useful level;
it is kept at level 2 for 256x256 faces so both datasets share one schema.

## Features

All features are shares of the **total detail energy** across both levels and all three
orientations; approximation coefficients are excluded.

| Column | Meaning |
| --- | --- |
| `dwt_horizontal_energy_ratio` | Energy in `cH` at both levels / total detail energy |
| `dwt_vertical_energy_ratio` | Energy in `cV` at both levels / total detail energy |
| `dwt_diagonal_energy_ratio` | Energy in `cD` at both levels / total detail energy |
| `dwt_level1_energy_ratio` | Energy in all three bands at the fine level 1 / total detail energy |
| `dwt_level2_energy_ratio` | Energy in all three bands at the coarse level 2 / total detail energy |

Level energies are coefficient-energy shares, not equal-weight averages of per-level ratios.
Symmetric boundary extension produces redundant boundary coefficients, so no energy-conservation
claim is made; border effects can be substantial at 32x32.

## Example

{% include figure.html src="/0000-10-real-dwt.png" alt="Early prototype multilevel DWT layout and coefficients for a REAL CIFAKE image" caption="**Figure 1.** Early prototype: PyWavelets coefficient layout for 1, 2, and 3 decomposition levels (top) and the packed coefficients of a REAL CIFAKE image (bottom). The pipeline uses 2 levels; a REAL/FAKE comparison is on the CIFAKE results page." %}
