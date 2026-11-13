"""Headless figures and training-only descriptive statistics."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.signal import firwin, freqz
from scipy.ndimage import convolve1d

from .dataset import LABELS
from .dsp_features import (
    FEATURE_COLUMNS, fft_transform, dct_transform, dwt_transform,
)


def plot_transforms(representatives: dict, destination: Path) -> None:
    """Use the same color scale for REAL and FAKE in each transform column."""
    panels = []
    for label in LABELS.values():
        _, image = representatives[label]
        wavelets, _ = pywt.coeffs_to_array(dwt_transform(image))
        panels.append([image, np.log1p(np.abs(fft_transform(image))),
                       np.log1p(np.abs(dct_transform(image))),
                       np.log1p(np.abs(wavelets))])
    titles = ["Grayscale [0, 1]", "FFT: log(1 + |F|)",
              "DCT: log(1 + |C|)", "DWT db4, L2: log(1 + |c|)"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), layout="constrained")
    for col, title in enumerate(titles):
        vmax = 1 if col == 0 else max(row[col].max() for row in panels)
        for row, (name, label) in enumerate(LABELS.items()):
            ax = axes[row, col]
            artist = ax.imshow(panels[row][col], cmap="gray" if col == 0 else "magma",
                               vmin=0, vmax=max(vmax, 1e-12), interpolation="nearest")
            ax.set_title(title if row == 0 else "")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(name)
                ax.set_xlabel(Path(representatives[label][0]).name, fontsize=8)
        fig.colorbar(artist, ax=axes[:, col], shrink=0.65)
    fig.suptitle("Training examples (first valid sample per class; no numerical resizing)")
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def summarize_training(groups: dict[int, np.ndarray], destination: Path) -> None:
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class", "label", "feature", "count", "mean", "std"])
        for name, label in LABELS.items():
            data = groups[label]
            for index, feature in enumerate(FEATURE_COLUMNS):
                # Sample standard deviation; a single observation gets 0 by convention.
                std = data[:, index].std(ddof=1) if len(data) > 1 else 0.0
                writer.writerow([name, label, feature, len(data), data[:, index].mean(), std])


def plot_distributions(groups: dict[int, np.ndarray], destination: Path) -> None:
    chosen = ["fft_high_energy_ratio", "dct_high_energy_ratio", "dwt_level1_energy_ratio"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, feature in zip(axes, chosen):
        index = FEATURE_COLUMNS.index(feature)
        pooled = np.concatenate([values[:, index] for values in groups.values()])
        upper = min(1.0, max(0.05, float(pooled.max()) * 1.05))
        bins = np.linspace(0, upper, 26)
        for (name, label), color in zip(LABELS.items(), ("#2667a0", "#c45932")):
            values = groups[label][:, index]
            ax.hist(values, bins=bins, weights=np.ones(len(values)) / len(values),
                    alpha=0.5, color=color, label=f"{name} (n={len(values)})")
        ax.set(xlabel=feature.replace("_", " "), ylabel="Fraction of class", xlim=(0, upper))
        ax.legend(fontsize=8)
    fig.suptitle("Training subset only; common bins and per-class normalization")
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def plot_sampling_experiment(destination: Path) -> None:
    """Synthetic gratings make aliasing visible without inventing dataset results."""
    y, x = np.mgrid[:96, :96]
    image = 0.5 + 0.25 * np.cos(2 * np.pi * 0.4 * x) + 0.2 * np.cos(2 * np.pi * 0.08 * y)
    kernel = firwin(25, 0.20, fs=1.0, window="hamming")
    low = convolve1d(convolve1d(image, kernel, axis=0, mode="reflect"),
                     kernel, axis=1, mode="reflect")
    residual = image - low
    panels = [image, low, residual, image[::2, ::2], low[::2, ::2]]
    titles = ["Synthetic: 0.08 and 0.40 cycles/pixel", "Separable FIR low-pass (cutoff 0.20)",
              "High-pass residual: original - low-pass", "2x downsample, no prefilter (aliasing)",
              "2x downsample after low-pass"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for ax, panel, title in zip(axes.flat, panels, titles):
        is_residual = panel is residual
        ax.imshow(panel, cmap="RdBu_r" if is_residual else "gray", interpolation="nearest",
                  vmin=-0.5 if is_residual else 0, vmax=0.5 if is_residual else 1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    frequency, response = freqz(kernel, worN=1024, fs=1.0)
    axes[1, 2].plot(frequency, 20 * np.log10(np.maximum(np.abs(response), 1e-6)))
    axes[1, 2].axvline(0.25, color="#c45932", linestyle="--", label="New Nyquist")
    axes[1, 2].set(xlabel="Cycles/original pixel", ylabel="Magnitude (dB)",
                   title="25-tap FIR frequency response (one axis)", ylim=(-100, 5))
    axes[1, 2].legend()
    fig.suptitle("Filtering and sampling demonstration (synthetic signal)")
    fig.savefig(destination, dpi=200)
    plt.close(fig)
