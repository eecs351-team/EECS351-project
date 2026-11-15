---
title: Reproducibility
nav_order: 9
has_children: true
has_toc: false
---

# Reproducibility
{: .no_toc }

How to install the code and regenerate every number and figure on this site.
{: .lead }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

## Installation

Use **Python 3.10+**. Direct dependencies are NumPy, SciPy, PyWavelets, Matplotlib, Pillow, and
Hugging Face `datasets`; versions are in `requirements.txt`. Run from the repository root in
Windows PowerShell:

```powershell
py -3 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

On macOS/Linux, use `python3 -m venv .venv` and `.venv/bin/python` below. Manually extract CIFAKE
into `datasets/CIFAKE/{train,test}/{REAL,FAKE}/`. RajuFace requires network access and streams
through Hugging Face; the full dataset is not downloaded automatically.

## Usage

Each command performs feature extraction, descriptive analysis, and figure export. Choose fresh
output directories to preserve prior runs.

```powershell
# CIFAKE demo: 5/class train, 2/class test
.venv/Scripts/python.exe demo.py --output-dir outputs/cifake-demo-new

# CIFAKE experiment: 1,000/class train, 200/class test
.venv/Scripts/python.exe -m src.task1 --train-per-class 1000 --test-per-class 200 --seed 42 --output-dir outputs/cifake-run-new

# RajuFace: run the 10/class demo first; continue only after it passes
$env:HF_HOME = Join-Path (Get-Location) '.cache/huggingface'
$rajuRevision = '1f99c8e831601e3a300dd3de670df23384a9be6c'
.venv/Scripts/python.exe -m src.rajuface --per-class 10 --seed 42 --revision $rajuRevision --output-dir outputs/rajuface/demo-new
.venv/Scripts/python.exe -m src.rajuface --per-class 200 --seed 42 --revision $rajuRevision --output-dir outputs/rajuface/initial-new

# Offline mathematical, sampling, and export checks
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Use `--data-dir` for another CIFAKE location. RajuFace supports `--dataset-id`, `--split`,
`--revision`, `--buffer-size`, and `--max-scan`; it rejects existing output directories. Defaults
are a 128-row encoded shuffle buffer and a maximum scan of 10,000 source rows. Sampling has
prefix/order bias and is not uniform over the full split. Balance failures or unrecoverable
network errors stop the run without publishing partial results.

Runs save feature CSVs, class summaries, figures, and reproducibility metadata. Datasets and
`outputs/` are Git-ignored; selected publication copies are in `docs/`. **28 tests passed** in the
existing validation, alongside both live RajuFace runs and a CIFAKE smoke run. There is no
classification command yet.

## Project structure

```text
demo.py                  Small CIFAKE run
requirements.txt         Runtime dependencies
src/
  dataset.py             CIFAKE loading and shared grayscale preprocessing
  rajuface_dataset.py    Schema validation and balanced HF streaming
  dsp_features.py        Shared FFT, DCT, DWT and 11-feature extractor
  analysis.py            Summaries and saved figures
  task1.py               CIFAKE CLI
  rajuface.py            RajuFace CLI
tests/                   CIFAKE and RajuFace validation
docs/
  task1.md               CIFAKE technical guide
  rajuface.md            RajuFace technical guide
  figures/               Selected experiment figures for publication
  results/               Compact summaries and experiment metadata
  *.pdf                  Original course requirements and project proposal
pages/                   Website pages (this site)
datasets/                Local data (ignored)
outputs/                 Generated runs (ignored)
```

## Technical guides

- [CIFAKE technical guide](../docs/task1.md): setup, feature definitions, outputs, and the
  synthetic filtering demonstration.
- [RajuFace technical guide](../docs/rajuface.md): streaming, sampling, preprocessing, and
  validation at 256x256.
