# EECS 351: AI-Generated Image Detection

Ethan Kocheril, Jiayuan Zhu, Zhirui Wang, Ziyi Xie

Task 1 investigates the frequency, scale, and orientation statistics of real and
AI-generated images. It extracts FFT, DCT, and DWT features from CIFAKE and exports
reproducible tables for a future Logistic Regression classifier. Classification
and Task 2/3 experiments are not implemented here.

## Setup and data

Python 3.10+; CPU only. Run commands from the repository root. On Windows PowerShell:

```powershell
py -3 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

On macOS/Linux, create the environment with `python3 -m venv .venv` and replace
`.venv/Scripts/python.exe` with `.venv/bin/python` below. An activated environment
can use `python` directly. No activation is required for the commands shown.

Download and extract the archive manually from the official
[CIFAKE dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images).
The program never downloads data. Point `--data-dir` to the directory containing
both original splits, including any enclosing folder created during extraction:

```text
datasets/CIFAKE/
  train/
    REAL/    # 50,000 JPGs; label 0
    FAKE/    # 50,000 JPGs; label 1
  test/
    REAL/    # 10,000 JPGs; label 0
    FAKE/    # 10,000 JPGs; label 1
```

Images retain native 32x32 resolution. Pillow converts RGB to its 8-bit `L`
grayscale using ITU-R 601 luminance (approximately 0.299 R + 0.587 G + 0.114 B),
then the pipeline converts to float64 and divides by 255. There is no resizing,
denoising, per-image standardization, or label-dependent preprocessing.

## Run

```powershell
# Demo: 5 REAL + 5 FAKE train; 2 REAL + 2 FAKE test
.venv/Scripts/python.exe demo.py --data-dir datasets/CIFAKE

# Initial experiment: 1,000 per class train; 200 per class test
.venv/Scripts/python.exe -m src.task1 --data-dir datasets/CIFAKE --output-dir outputs --train-per-class 1000 --test-per-class 200 --seed 42

# Entire original dataset (100,000 train and 20,000 test when intact)
.venv/Scripts/python.exe -m src.task1 --data-dir datasets/CIFAKE --output-dir outputs/full --all

# Synthetic mathematical, dataset, and end-to-end checks
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

`--train-per-class`, `--test-per-class`, `--seed`, and `--output-dir` work for the
demo too. Counts must be positive. Sorted filenames and a separate seeded shuffle
for each split/class produce deterministic row order; changing test sample counts
does not change training selection. This assumes unchanged data, code, and library
versions. Metadata records versions. Train/test folders are never merged.

Corrupt or wrong-sized images produce warnings and are recorded in metadata.
The loader tries the remaining deterministic candidates to replace them. If it
cannot reach the requested balanced counts, the run fails. `--all` requires equal
available class counts and enough valid images to process all of them. New results
are staged until both splits and all plots succeed; a failed extraction leaves a
previous successful run's outputs intact. Check the command's exit status.

## Structure

```text
src/
  dataset.py        # CIFAKE validation, balanced iteration, preprocessing
  dsp_features.py   # transforms and extract_features(image), no dataset logic
  analysis.py       # training summaries and headless figures
  task1.py          # CLI, CSV export, and pipeline orchestration
tests/test_task1.py # mathematical and data-integrity checks
demo.py             # same pipeline with smaller defaults
requirements.txt    # five runtime dependencies
```

There are no configuration frameworks or custom class hierarchies. Data are read
one image at a time. Only filenames, numeric training features, and two training
examples are retained; transform arrays are not saved for every image. With P
pixels, FFT and DCT cost approximately O(P log P), and fixed-filter DWT O(P).

The original repository had a README, four example PNGs, and two PDF documents,
with no standalone Python source. The prototype embedded in the finalized report
provided the `fft2` / `fftshift` approach, retained here with safe `log1p` display.
Its 1D `wavedec` level 4 is replaced by true 2D `wavedec2` level 2. Pillow provides
image loading without adding the larger OpenCV dependency. Original report PDFs
and example PNGs are preserved.

## Features

`from src.dsp_features import extract_features, FEATURE_COLUMNS` provides a fixed,
ordered dictionary of 11 finite floats from a 2D grayscale array in [0, 1].
All energies below are sums of **squared** coefficient magnitudes. Features are
defined before looking at test statistics.

| Feature columns | Definition |
| --- | --- |
| `fft_low_energy_ratio`, `fft_mid_energy_ratio`, `fft_high_energy_ratio` | Centered FFT radii `r = hypot(fftfreq(H), fftfreq(W))`, in cycles/pixel. Bands `[0, 0.15)`, `[0.15, 0.35)`, `[0.35, infinity)`. Divide each band energy by total non-DC energy. |
| `dct_low_energy_ratio`, `dct_high_energy_ratio` | Orthonormal type-II DCT. Basis radius `r = hypot(k/(2H), l/(2W))`. Low `r < 0.15`, high `r >= 0.15`; remove `(0,0)` DC before dividing by total AC energy. |
| `dct_ac_concentration` | With normalized AC coefficient energies `p_i`, compute `sum(p_i**2)`. Near 1 means energy concentrated in a few cosine coefficients; smaller values indicate more distributed AC energy. |
| `dwt_horizontal_energy_ratio`, `dwt_vertical_energy_ratio`, `dwt_diagonal_energy_ratio` | Sum squared `cH`, `cV`, or `cD` coefficients across both levels; divide by total detail energy across all levels/orientations. Approximation coefficients are excluded. |
| `dwt_level1_energy_ratio`, `dwt_level2_energy_ratio` | Sum all three detail orientations at the fine level 1 or coarse level 2; divide by the same total detail energy. |

FFT ratios sum to 1, DCT low/high ratios sum to 1, DWT orientation ratios sum to 1,
and DWT level ratios sum to 1 whenever the relevant total exceeds `1e-24`.
Otherwise all ratios in that group are zero, including constant-image cases;
the same threshold applies to DCT concentration. This convention ignores numerical
roundoff near zero. FFT uses NumPy's default unnormalized forward transform;
DCT is orthonormal. Do not compare their raw energy totals without normalization.

DWT uses **db4, level 2, symmetric extension**, validated against
`pywt.dwtn_max_level`. For 32x32 inputs, the maximum useful level is 2. Following
[PyWavelets' axis convention](https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html),
`cH` is high-pass along array axis 0 (changes across rows), `cV` along axis 1
(changes across columns), and `cD` along both. Level energies are coefficient
energy shares, not equal-weight averages of per-level ratios or per-pixel energy.
[Symmetric boundary extension](https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html)
produces redundant boundary coefficients: no energy-conservation claim is made.
Border effects can be substantial at 32x32.

The generic extractor accepts larger rectangular images with the same schema and
fixed two-level DWT. The CIFAKE loader deliberately checks 32x32. Normalized bands
do not make features equivalent across resolutions; resolution changes require
reevaluation. Some columns are algebraically dependent; this is an interpretable
summary schema, not feature selection for a classifier.

## Outputs

Normal runs write to `outputs/`; demo runs write to `outputs/demo/`:

```text
features/
  train_features.csv
  test_features.csv
  train_summary.csv
figures/
  transforms.png
  train_distributions.png
  filtering_sampling.png
metadata.json
```

Each feature CSV has `image_path` (relative to the dataset), `split`, `label`, then
`FEATURE_COLUMNS` in identical order. Summaries report each training class's count,
mean, and sample standard deviation (ddof=1; singleton convention 0). Test features
are exported for future evaluation; no test class distributions or summaries are
computed. Future code can load the CSV with pandas (optional, not a runtime
dependency) and use `X = df[list(FEATURE_COLUMNS)].to_numpy()` and
`y = df['label'].to_numpy()`. Any later scaling must be fitted on training data.

The transform figure uses the first valid selected training image of each class,
with common color limits per transform and nearest-neighbor display. Packed DWT
coefficients have approximation in the upper left, with detail regions at each
scale and zero padding where necessary. The distribution figure uses common
bins and per-class normalization for FFT high energy, DCT high energy, and DWT
fine-level detail share. Each panel includes the full observed training range,
with identical axis limits and bin edges for both classes.

The filtering figure is explicitly a **synthetic** demonstration: gratings at 0.08
and 0.40 cycles/pixel, a separable 25-tap Hamming-windowed FIR low-pass with cutoff
0.20 cycles/pixel, and a high-pass residual `I - lowpass(I)`. Spatial convolution
corresponds to multiplication by the filter frequency response away from reflected
boundaries. Decimating by 2 lowers Nyquist to 0.25 cycles/original pixel: without
prefiltering, the 0.40 component aliases to 0.20 cycles/new pixel. The finite FIR
attenuates above the new Nyquist, with a nonzero transition band and boundary
effects. This experiment does not modify extraction images or add features.

For the course requirements, connect Fourier/cosine/wavelet representations to
change of basis and frequency analysis, the FIR/residual to convolution and linear
filtering, and decimation to sampling and aliasing. Do not automatically count
FFT/DCT/DWT as three separate in-class tools. Confirm the coverage with the course
materials; Logistic Regression is the planned out-of-class technique for Task 2.

## Limitations and references

CIFAKE contains only 32x32 images; its synthetic class comes from Stable Diffusion
1.4. Resolution, compression, boundary treatment, and preprocessing can influence
all these features. Class differences may reflect dataset artifacts rather than
general signs of AI generation. No classification accuracy or generalization to
other generators is claimed. Cross-dataset and robustness studies belong to later
tasks. Data, environments, and generated outputs are ignored by Git; copy selected
report figures into `docs/figures/` when ready to commit them.

- Jordan J. Bird and Ahmad Lotfi, *CIFAKE: Image Classification and Explainable
  Identification of AI-Generated Synthetic Images*, 2023.
  [Original paper](https://arxiv.org/abs/2303.14126) and
  [dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images).
- Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009.
  [CIFAR-10 source and original report](https://www.cs.toronto.edu/~kriz/cifar.html).

## Original examples

These historical figures are retained independently of reproducible pipeline outputs:

![Original FFT example](finalization-task-1.png)
![Original REAL DWT example](0000-10-real-dwt.png)
![Original FAKE DWT example](1000-10-fake-dwt.png)
![Original REAL DCT example](0000-10-real-dct.png)
