---
title: "Faces: frequency analysis"
parent: Results
nav_order: 1
---

# Faces: frequency analysis
{: .no_toc }

140k Real and Fake Faces at 256x256: 5,000 real and 5,000 generated training images.
{: .lead }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

All figures on this page use the training images only; the test images are kept for
[Classification](../classification.md).

## One example per class

{% include figure.html src="/docs/figures/faces/face_examples.png" alt="A real and a generated face in grayscale, each next to its Fourier power spectrum" caption="**Figure 1.** One real (top) and one generated (bottom) training image in grayscale, each next to its Fourier power spectrum on a log scale. Low frequencies are at the center. Single spectra are noisy and look alike, so we compare averages over thousands of images." %}

## Average spectra

{% include figure.html src="/docs/figures/faces/mean_spectra.png" alt="Average Fourier spectra of real and generated faces and their difference, without a window" caption="**Figure 2.** Average log Fourier power of 5,000 real and 5,000 generated faces, and their difference (right; blue = less power in generated faces, red = more). No window was applied." %}

- There are **no periodic bright spots or grid patterns**, the classic trace of generator
  upsampling layers.
- Generated faces have **slightly less power above about 0.1 cycles/pixel**, meaning slightly
  less fine detail. The effect is small: the classes differ by about 0.3 standard deviations.
- The sharp dark horizontal line through the center of the difference map comes from the image
  edges, not from the faces (see below).

### With a Hann window

The Fourier transform treats each image as if it repeated forever. When the left and right
edges of an image do not match, the repetition creates a sharp jump, and that jump spreads
false energy along the frequency axes (**spectral leakage**). This produces the bright "+"
shape in single spectra. A Hann window fades the image smoothly to zero at its edges before
the transform and removes the leakage.

{% include figure.html src="/docs/figures/faces/mean_spectra_hann.png" alt="Average Fourier spectra of real and generated faces and their difference, after a Hann window" caption="**Figure 3.** The same average spectra after a 2D Hann window." %}

- The dark horizontal line disappears, so it was leakage from the image edges.
- A red band appears along the vertical frequency axis (about 0.3 to 0.45 cycles/pixel):
  generated faces have relatively more energy in fine horizontal structures.
- A fine ripple pattern near the axes is also visible; we have not identified its cause yet.

## Spectrum by frequency

{% include figure.html src="/docs/figures/faces/radial_spectrum.png" alt="Radial average of the Fourier spectrum for both classes, and the effect size at each frequency" caption="**Figure 4.** Left: power averaged over rings of equal frequency, mean ± one standard deviation per class. Right: effect size (Cohen's d, the difference in means divided by the spread) at each frequency; values near 0.2 count as a small effect." %}

The two curves almost overlap. The gap grows from low frequencies up to about 0.1 to 0.15
cycles/pixel and then stays near -0.3 standard deviations. A single frequency cannot
separate the classes, but the full curve shape helps a classifier (see
[Classification](../classification.md)).

## Differences by face region

Because all faces are aligned, each tile of a 4x4 grid covers roughly the same part of every
face. We computed two fine-detail measures on each tile: the share of Fourier energy at high
frequencies ("fft high energy ratio" in the figure) and the share of wavelet detail at the
finest scale ("dwt level1 energy ratio").

{% include figure.html src="/docs/figures/faces/grid_effects.png" alt="Average real face with a 4x4 grid, and the effect size of two fine-detail measures in each tile" caption="**Figure 5.** Left: the average real face with the 4x4 grid. Middle and right: effect size in each tile (blue = less fine detail in generated faces, red = more)." %}

- Generated faces have **less fine detail in the middle of the face**: the eye row for the
  Fourier measure and the mouth-and-chin rows for the wavelet measure (up to about -0.24).
- The **top corners**, mostly background, differ least.
- The top-middle tiles (hair) go the other way for the wavelet measure: slightly more fine
  detail in generated faces.
- All effects are small, which matches the modest classification accuracy.

<details markdown="block">
  <summary>Earlier first look (Task 1, 200 test images per class)</summary>

Before the larger experiment, we compared the 11 basic features on 200 real and 200
generated test images.

| Feature | REAL (n=200) | FAKE (n=200) |
| --- | --- | --- |
| Fourier high-frequency energy share | 0.01181 ± 0.00707 | 0.01085 ± 0.00739 |
| Cosine high-frequency energy share | 0.03140 ± 0.01750 | 0.02915 ± 0.01780 |
| Wavelet finest-scale detail share | 0.40652 ± 0.06589 | 0.39717 ± 0.06579 |

Real faces were slightly higher on all three, with heavily overlapping distributions, which the
larger experiment above confirms. Details: [summaries (CSV)](../../docs/results/rajuface_test_summary.csv),
[technical guide](../../docs/rajuface.md).
</details>
