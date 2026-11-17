"""Headless figures for the 140k faces experiment (training data unless noted)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dataset import LABELS
from .spectral import GRID, GRID_FEATURES, log_power_spectrum

COLORS = ("#2667a0", "#c45932")


def cohens_d(real: np.ndarray, fake: np.ndarray) -> np.ndarray:
    """(mean FAKE - mean REAL) / pooled sample standard deviation, per column."""
    pooled = np.sqrt((real.var(axis=0, ddof=1) + fake.var(axis=0, ddof=1)) / 2)
    return (fake.mean(axis=0) - real.mean(axis=0)) / np.maximum(pooled, 1e-12)


def plot_face_examples(representatives: dict, destination: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7), layout="constrained")
    spectra = {label: log_power_spectrum(image) for label, (_, image) in representatives.items()}
    vmin = min(np.percentile(s, 1) for s in spectra.values())
    vmax = max(s.max() for s in spectra.values())
    for row, (name, label) in enumerate(LABELS.items()):
        record, image = representatives[label]
        axes[row, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(name)
        axes[row, 0].set_xlabel(record, fontsize=8)
        artist = axes[row, 1].imshow(spectra[label], cmap="magma", vmin=vmin, vmax=vmax)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_title("Grayscale 256x256")
    axes[0, 1].set_title("FFT: log10 power (mean removed)")
    fig.colorbar(artist, ax=axes[:, 1], shrink=0.6)
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def plot_mean_spectra(mean_spectra: dict[int, np.ndarray], counts: dict[int, int],
                      destination: Path,
                      title: str = "Class-average FFT spectra, training subset") -> None:
    """Class-average log spectra and their difference; fingerprints show as peaks/grids."""
    real, fake = mean_spectra[0], mean_spectra[1]
    size = real.shape[0]
    extent = (-0.5, 0.5 - 1 / size, 0.5 - 1 / size, -0.5)
    difference = fake - real
    center = size // 2
    difference[center, center] = 0  # DC is removed in both classes
    limit = np.percentile(np.abs(difference), 99.5)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    vmin, vmax = np.percentile(real, 1), max(real.max(), fake.max())
    for ax, data, panel in ((axes[0], real, f"REAL mean (n={counts[0]})"),
                            (axes[1], fake, f"FAKE mean (n={counts[1]})")):
        artist = ax.imshow(data, cmap="magma", vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(f"{panel}: log10 power")
    fig.colorbar(artist, ax=axes[:2], shrink=0.8)
    artist = axes[2].imshow(difference, cmap="RdBu_r", vmin=-limit, vmax=limit, extent=extent)
    axes[2].set_title("FAKE - REAL (log10 power ratio)")
    fig.colorbar(artist, ax=axes[2], shrink=0.8)
    for ax in axes:
        ax.set(xlabel="Horizontal frequency (cycles/pixel)",
               ylabel="Vertical frequency (cycles/pixel)")
    fig.suptitle(title)
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def plot_radial(radial: dict[int, np.ndarray], destination: Path) -> None:
    size = 2 * (radial[0].shape[1] + 1)
    frequency = np.arange(1, size // 2) / size
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for (name, label), color in zip(LABELS.items(), COLORS):
        values = radial[label]
        mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
        axes[0].plot(frequency, mean, color=color, label=f"{name} (n={len(values)})")
        axes[0].fill_between(frequency, mean - std, mean + std, color=color, alpha=0.2)
    axes[0].set(xlabel="Radial frequency (cycles/pixel)", ylabel="log10 mean ring power",
                title="Azimuthally averaged spectrum (mean +/- std)")
    axes[0].legend()
    axes[1].plot(frequency, cohens_d(radial[0], radial[1]), color="#444444")
    axes[1].axhline(0, color="#999999", linewidth=0.8)
    axes[1].set(xlabel="Radial frequency (cycles/pixel)", ylabel="Cohen's d (FAKE - REAL)",
                title="Per-frequency class separation")
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def plot_grid_effects(grid: dict[int, np.ndarray], mean_face: np.ndarray,
                      destination: Path) -> None:
    """Which aligned face regions carry the largest high-frequency differences."""
    effect = cohens_d(grid[0], grid[1]).reshape(GRID, GRID, len(GRID_FEATURES))
    limit = max(np.abs(effect).max(), 0.1)
    fig, axes = plt.subplots(1, 1 + len(GRID_FEATURES), figsize=(13, 4.4), layout="constrained")
    axes[0].imshow(mean_face, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Mean REAL training face")
    step = mean_face.shape[0] / GRID
    for line in np.arange(1, GRID) * step:
        axes[0].axhline(line - 0.5, color="#ffcc00", linewidth=0.8)
        axes[0].axvline(line - 0.5, color="#ffcc00", linewidth=0.8)
    for index, name in enumerate(GRID_FEATURES):
        ax = axes[index + 1]
        artist = ax.imshow(effect[:, :, index], cmap="RdBu_r", vmin=-limit, vmax=limit)
        for r in range(GRID):
            for c in range(GRID):
                ax.text(c, r, f"{effect[r, c, index]:+.2f}", ha="center", va="center", fontsize=9)
        ax.set_title(f"Cohen's d per tile: {name.replace('_', ' ')}", fontsize=10)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(artist, ax=axes[1:], shrink=0.8, label="FAKE - REAL (pooled std units)")
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def plot_classification(metrics: list[dict], confusion: np.ndarray, best: str,
                        destination: Path,
                        title: str = "Logistic Regression by feature group") -> None:
    """Test accuracy per feature group, plus the confusion matrix of one group."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained",
                             gridspec_kw={"width_ratios": [2, 1]})
    names = [row["feature_group"] for row in metrics]
    accuracy = [row["test_accuracy"] for row in metrics]
    bars = axes[0].barh(names, accuracy, color="#2667a0")
    axes[0].bar_label(bars, labels=[f"{value:.3f}" for value in accuracy], padding=3)
    axes[0].axvline(0.5, color="#999999", linestyle="--", label="Chance")
    axes[0].set(xlim=(0, 1.08), xlabel="Held-out test accuracy", title=title)
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right")
    # Fix the color scale at [0, row total] so shading and text color agree.
    axes[1].imshow(confusion, cmap="Blues", vmin=0, vmax=confusion.sum(axis=1).max())
    for (r, c), value in np.ndenumerate(confusion):
        axes[1].text(c, r, str(value), ha="center", va="center",
                     color="white" if value > 0.6 * confusion.sum(axis=1).max() else "black")
    axes[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=list(LABELS),
                yticklabels=list(LABELS), xlabel="Predicted", ylabel="True",
                title=f"Confusion matrix: {best}")
    fig.savefig(destination, dpi=200)
    plt.close(fig)


TRANSFORM_COLORS = {"fft": "#2a78d6", "dct": "#eb6834", "dwt": "#1baf7a"}


def plot_band_features(columns, band: dict[int, np.ndarray], coefficients: np.ndarray,
                       destination: Path) -> None:
    """Per-feature class separation (left) and standardized LR weights (right).

    Cohen's d reads one feature at a time. The weights are joint: several band ratios
    sum to one within a transform (FFT low+mid+high, DCT low+high, DWT H+V+D, DWT
    level 1+2), so a weight acts as a contrast between bands, not on its own.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained", sharey=True)
    names = [name.replace("_energy_ratio", "").replace("_", " ") for name in columns]
    colors = [TRANSFORM_COLORS[name.split("_")[0]] for name in columns]
    panels = ((cohens_d(band[0], band[1]), "Cohen's d (FAKE - REAL)",
               "Class separation per feature (training)"),
              (coefficients, "Standardized weight (positive pushes toward FAKE)",
               "Logistic Regression weights (all 11 jointly)"))
    for ax, (values, xlabel, title) in zip(axes, panels):
        ax.barh(names, values, height=0.6, color=colors)
        ax.axvline(0, color="#999999", linewidth=0.8)
        ax.set(xlabel=xlabel, title=title)
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in TRANSFORM_COLORS.values()]
    axes[1].legend(handles, [t.upper() for t in TRANSFORM_COLORS], loc="lower right",
                   frameon=False)
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def plot_classifier_comparison(metrics: list[dict], confusion: np.ndarray, best: str,
                               destination: Path) -> None:
    """Test accuracy for every classifier x feature set, plus the best confusion matrix."""
    sets = list(dict.fromkeys(row["feature_set"] for row in metrics))
    classifiers = list(dict.fromkeys(row["classifier"] for row in metrics))
    accuracy = np.full((len(classifiers), len(sets)), np.nan)
    for row in metrics:
        accuracy[classifiers.index(row["classifier"]), sets.index(row["feature_set"])] = \
            row["test_accuracy"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained",
                             gridspec_kw={"width_ratios": [1.6, 1]})
    top = max(0.75, float(np.nanmax(accuracy)))
    axes[0].imshow(accuracy, cmap="Blues", vmin=0.5, vmax=top, aspect="auto")
    for (r, c), value in np.ndenumerate(accuracy):
        axes[0].text(c, r, f"{value:.3f}", ha="center", va="center",
                     color="white" if value > 0.5 + 0.6 * (top - 0.5) else "black")
    axes[0].set(xticks=range(len(sets)), yticks=range(len(classifiers)),
                title="Held-out test accuracy (0.5 = chance)")
    axes[0].set_xticklabels(sets, fontsize=9)
    axes[0].set_yticklabels(classifiers)
    total = confusion.sum(axis=1).max()
    axes[1].imshow(confusion, cmap="Blues", vmin=0, vmax=total)
    for (r, c), value in np.ndenumerate(confusion):
        axes[1].text(c, r, str(value), ha="center", va="center",
                     color="white" if value > 0.6 * total else "black")
    axes[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=list(LABELS),
                yticklabels=list(LABELS), xlabel="Predicted", ylabel="True",
                title=f"Confusion matrix: {best}")
    axes[1].title.set_fontsize(10)
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def plot_real_vs_fake(shard: Path, destination: Path, per_class: int = 4, seed: int = 42,
                      split: str = "train") -> None:
    """Color examples of each class, chosen with the same seeded order as iter_faces."""
    from io import BytesIO
    import random

    from PIL import Image
    import pyarrow.parquet as pq

    table = pq.read_table(shard, columns=["image", "label"])
    labels = table.column("label").to_numpy()
    images = table.column("image")
    captions = {"REAL": "REAL: FFHQ photographs", "FAKE": "FAKE: StyleGAN faces"}
    fig, axes = plt.subplots(2, per_class, figsize=(2.4 * per_class, 5.2), layout="constrained")
    for row, (name, label) in enumerate(LABELS.items()):
        rows = np.flatnonzero(labels == label).tolist()
        random.Random(f"{seed}:{split}:{name}").shuffle(rows)
        for col, index in enumerate(rows[:per_class]):
            with Image.open(BytesIO(images[index].as_py()["bytes"])) as image:
                axes[row, col].imshow(np.asarray(image.convert("RGB")))
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(captions[name], fontsize=11)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
