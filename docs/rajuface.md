---
title: RajuFace technical guide
parent: Reproducibility
nav_order: 2
---

# RajuFace: Native-Resolution Technical Guide

The same DSP extractor now supports
[RajuFace/140k-Real-and-Fake-Faces](https://huggingface.co/datasets/RajuFace/140k-Real-and-Fake-Faces)
through Hugging Face `datasets`. Live schema inspection confirmed `image: Image`,
`label: ClassLabel(['real', 'fake'])`, and
`generator: ClassLabel(['Real', 'StyleGAN'])`, with train/validation/test sizes
100,000/20,000/20,000. **REAL = 0, FAKE = 1**. Generator names are exported as
metadata and never enter the numerical feature extractor.

## Reproduce

Run the small demo first and proceed only if it succeeds. Use the Python and
virtual environment setup in the [installation instructions](../pages/reproducibility.md#installation), including
the `datasets` dependency in `requirements.txt`.
The commands below pin the dataset commit used for the reported measurements:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements.txt
$env:HF_HOME = Join-Path (Get-Location) '.cache/huggingface'

# Quick demo: 10 REAL + 10 FAKE, native resolution
.venv/Scripts/python.exe -m src.rajuface --per-class 10 --seed 42 --revision 1f99c8e831601e3a300dd3de670df23384a9be6c --output-dir outputs/rajuface/demo

# After the demo passes: 200 REAL + 200 FAKE
.venv/Scripts/python.exe -m src.rajuface --per-class 200 --seed 42 --revision 1f99c8e831601e3a300dd3de670df23384a9be6c --output-dir outputs/rajuface/initial

# Offline checks for both datasets
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

On macOS/Linux use `.venv/bin/python` and optionally
`export HF_HOME="$PWD/.cache/huggingface"`. Existing output directories are
**rejected**: choose a fresh `--output-dir` when repeating a run. `--dataset-id`,
`--split` (default `test`), `--revision` (default `main`, resolved to a recorded
commit), `--per-class`, `--seed`, `--buffer-size`, `--max-scan`, and `--cache-dir`
are configurable. A compatible three-field schema and native 256x256 images are
required. An optional 1,000-per-class run uses `--per-class 1000` and a new output
directory; that larger experiment has not been executed.

## Sampling and preprocessing

Loading always uses `streaming=True`; there is no full-download fallback or random
indexing. A seeded, bounded shuffle keeps up to **128 encoded rows** by default,
assigns source indices before shuffling, and accepts valid candidates until both
class quotas are filled. At most **10,000 source rows** are requested by default;
exhaustion or the scan limit fails clearly with actual counts. This method samples
a source prefix, with possible shard/order bias; it is not a uniform sample of
the full split. The same revision, seed, buffer, and library versions reproduce
selection. Demo and initial selections overlap and are not independent samples.
Parquet streaming may transfer/cache whole chunks or shards, so network traffic
can exceed the bytes of the selected images; it does not load all 140,000 images.

Images are decoded one at a time, checked for **256x256**, and passed through the
shared CIFAKE preprocessing: RGB -> Pillow 8-bit `L` grayscale -> float64 / 255.
There is **no resizing or enhancement**. Only numeric features, the bounded encoded
buffer, and two representative RGB/grayscale images are retained. Corrupt or
wrong-sized candidates warn, are recorded, and are replaced; unrecoverable network
errors abort. Results are published only after balanced extraction and plotting
complete. Existing CIFAKE outputs and commands keep their original locations.

## Feature definitions and outputs

All **11 original features and their order are unchanged**; see the
[mathematical definitions](task1.md#features). FFT bands use cycles/pixel,
with DC removed and boundaries 0.15/0.35. The orthonormal type-II DCT uses basis
frequencies `k/(2N)`, excludes DC, and keeps the 0.15 boundary and AC concentration
statistic. DWT remains **db4, level 2, symmetric extension**: orientation and level
energies are divided by total detail energy, excluding approximation coefficients.
Level 2 is valid at 256x256 and has not been increased. All ratios use squared
coefficients; totals at or below `1e-24` give zero. Resolution-normalized bands
preserve definitions without making feature distributions resolution-invariant.

Each run writes:

- `features/test_features.csv`: `dataset_id`, `split`, stable source `sample_id`,
  binary `label`, decoded `generator` metadata, then the 11 `FEATURE_COLUMNS`.
  The sample identifier is interpreted with the dataset commit and split recorded
  in metadata. Other splits use their own filenames.
- `features/test_summary.csv`: class count, mean, and sample standard deviation
  (`ddof=1`; singleton standard deviation is zero) for every feature.
- `figures/transform_comparison.png`: original RGB, grayscale, FFT, DCT, and packed
  DWT coefficients for the first valid selected image of each class, with shared
  transform color scales and `log(1 + |coefficient|)` for display only.
- `figures/feature_distributions.png`: common-bin, class-normalized histograms of
  FFT/DCT high-frequency ratios and fine-scale DWT detail share.
- `metadata.json`: schema, split sizes, resolved commit, seed, sampling settings,
  actual counts/skips, preprocessing, DSP definitions, and package versions.

## Measured results and validation

The live test-split demo processed **10 REAL + 10 FAKE** (21 candidates examined).
After it passed, the initial experiment processed **200 REAL + 200 FAKE** (411
candidates examined). Both used seed 42 and skipped **zero** images. The initial
experiment produced these means +/- sample standard deviations:

| Feature | REAL (n=200) | FAKE (n=200) |
| --- | --- | --- |
| FFT high-frequency energy ratio | 0.01181 +/- 0.00707 | 0.01085 +/- 0.00739 |
| DCT high-frequency energy ratio | 0.03140 +/- 0.01750 | 0.02915 +/- 0.01780 |
| DWT fine-scale (level 1) detail-energy ratio | 0.40652 +/- 0.06589 | 0.39717 +/- 0.06579 |

REAL means are slightly higher for these three features in this selected subset;
the distributions overlap substantially. These are descriptive observations on
the sampled test data, not classification performance or evidence of a universal
generator signature. No thresholds or model settings were tuned on these data.
CIFAKE and RajuFace also differ in content, resolution, sources, compression, and
generators, so differences cannot be attributed solely to StyleGAN.

No trained CIFAKE Logistic Regression model, scaler, or saved model configuration
exists in this repository. External classification is therefore **skipped**;
no model was trained and no `metrics.json` was generated. Any future external
evaluation must reconstruct the exact training preprocessing, feature order, and
scaler. If 32x32 compatibility resizing is needed, it belongs in a separate,
explicitly antialiased experiment and removes high-frequency information.

All **28 tests passed** (16 existing CIFAKE checks + 12 RajuFace checks), including
real Hugging Face iterable fixtures, deterministic balanced sampling, schema/label
validation, bounded consumption, corruption and network failures, native 256x256
transform energy/reconstruction, finite feature exports, summary statistics,
figures, and overwrite protection. Both live RajuFace CSVs and PNGs were verified,
and both figure types were visually inspected. A separate live CIFAKE smoke run
passed with 5/class train and 2/class test:

```powershell
.venv/Scripts/python.exe demo.py --output-dir outputs/cifake-regression-rajuface
```

The original CIFAKE summary and metadata still match their archived `docs/results/`
copies. Generated outputs remain under the Git-ignored `outputs/` directory.

## Archived results

The following files are publication copies of the existing 200-per-class run;
original experiment outputs remain unchanged in `outputs/rajuface/initial/`.

- [Transform comparison](figures/rajuface_transforms.png).
- [Feature distributions](figures/rajuface_distributions.png).
- [Complete test-subset summary](results/rajuface_test_summary.csv).
- [Dataset schema, commit, settings, and counts](results/rajuface_metadata.json).

Return to the [project overview](../index.md).
