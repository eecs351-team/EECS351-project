"""Task 2: DSP features + Logistic Regression on 140k Real and Fake Faces.

Training images come from one parquet shard and held-out test images from a
different shard. Every preprocessing and model choice (feature scaling and the
regularization strength chosen by 5-fold cross-validation) uses training data only.

Feature groups come in two stages. The basic stage follows the project proposal:
the 11 Task 1 band ratios, split by transform (FFT, DCT, DWT) and all together.
The extended stage adds finer spectral features that keep the spectrum's shape.
"""

import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import LABELS
from .dsp_features import FEATURE_COLUMNS, extract_features
from .face_analysis import (plot_band_features, plot_classification, plot_face_examples,
                            plot_grid_effects, plot_mean_spectra, plot_radial,
                            plot_real_vs_fake)
from .faces import iter_faces
from .spectral import (DCT_BLOCK, GRID, dct_log_blocks, grid_columns, grid_features,
                       log_power_spectrum, radial_spectrum)

# Feature arrays stored per image; evaluation groups below select from these.
GROUPS = ("band", "radial", "radial_hann", "dct_blocks", "grid")
# Columns of the 11 band ratios that belong to each transform.
TRANSFORM_COLUMNS = {
    prefix: [i for i, name in enumerate(FEATURE_COLUMNS) if name.startswith(f"{prefix}_")]
    for prefix in ("fft", "dct", "dwt")
}
# Evaluation groups in report order: (key, display name, stage).
EVAL_GROUPS = (
    ("fft", "FFT band ratios (3)", "basic"),
    ("dct", "DCT band ratios (3)", "basic"),
    ("dwt", "DWT band ratios (5)", "basic"),
    ("band", "All 11 band ratios", "basic"),
    ("radial", "Radial FFT spectrum (127)", "extended"),
    ("dct_blocks", "log-DCT blocks (1024)", "extended"),
    ("grid", "Tile high-freq (32)", "extended"),
    ("combined", "Combined spectral (1194)", "extended"),
    # Diagnostics: does windowing matter, and is it the spectrum's shape or level?
    ("radial_hann", "Radial FFT, Hann window (127)", "diagnostic"),
    ("radial_shape", "Radial shape only, Hann (127)", "diagnostic"),
    ("radial_level", "Radial level only, Hann (1)", "diagnostic"),
    ("jpeg", "JPEG table only (control)", "control"),
)
# Groups concatenated into "combined" (fixed so results stay comparable).
COMBINED = ("band", "radial", "dct_blocks", "grid")
GROUP_NAMES = {key: name for key, name, _ in EVAL_GROUPS}
GROUP_STAGES = {key: stage for key, _, stage in EVAL_GROUPS}
C_GRID = np.logspace(-4, 2, 7)


def extract_all(image: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "band": np.fromiter(extract_features(image).values(), dtype=np.float64),
        "radial": radial_spectrum(image),
        "radial_hann": radial_spectrum(image, window=True),
        "dct_blocks": dct_log_blocks(image),
        "grid": grid_features(image),
    }


def load_split(shard: Path, split: str, per_class: int, seed: int, report: dict,
               recompress_quality: int | None, keep_spectra: bool) -> dict:
    data = {group: [] for group in GROUPS}
    data.update(records=[], labels=[], table_sums=[])
    # Running sums of log spectra per class: without and with a Hann window.
    spectra = {window: {label: 0.0 for label in LABELS.values()} for window in (False, True)}
    images = {label: None for label in LABELS.values()}
    examples = {}
    for record, label, image, table_sum in iter_faces(shard, split, per_class, seed, report,
                                                      recompress_quality):
        for group, values in extract_all(image).items():
            data[group].append(values)
        data["records"].append(record)
        data["labels"].append(label)
        data["table_sums"].append(-1 if table_sum is None else table_sum)
        if keep_spectra:
            for window in (False, True):
                spectra[window][label] = spectra[window][label] + log_power_spectrum(image, window)
            images[label] = image.copy() if images[label] is None else images[label] + image
            examples.setdefault(label, (record, image))
    data = {key: np.asarray(value) for key, value in data.items()}
    counts = {label: int(np.sum(data["labels"] == label)) for label in LABELS.values()}
    if keep_spectra:
        for window, key in ((False, "mean_spectra"), (True, "mean_spectra_hann")):
            data[key] = {label: spectra[window][label] / counts[label] for label in counts}
        data["mean_images"] = {label: images[label] / counts[label] for label in images}
        data["examples"] = examples
    print(f"{split}: processed {sum(counts.values())} balanced images from {Path(shard).name}")
    return data


def feature_matrix(data, group: str) -> np.ndarray:
    """Build one evaluation group's (images x features) matrix from stored arrays."""
    if group in TRANSFORM_COLUMNS:
        return data["band"][:, TRANSFORM_COLUMNS[group]]
    if group == "combined":
        return np.hstack([data[name] for name in COMBINED])
    # Split each log spectrum into its overall level (mean over rings) and the
    # shape left after subtracting that level (a per-image gain normalization).
    if group == "radial_level":
        return data["radial_hann"].mean(axis=1, keepdims=True)
    if group == "radial_shape":
        return data["radial_hann"] - data["radial_hann"].mean(axis=1, keepdims=True)
    if group == "jpeg":
        return data["table_sums"][:, None].astype(np.float64)
    return data[group]


def evaluate(train: dict, test: dict, group: str, seed: int):
    """Fit one group's model on training data; return (metrics row, confusion, model)."""
    x_train, x_test = feature_matrix(train, group), feature_matrix(test, group)
    model = make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=C_GRID, cv=5, scoring="accuracy", max_iter=5000, random_state=seed))
    with warnings.catch_warnings():
        # Newer scikit-learn versions announce default/attribute changes here.
        warnings.simplefilter("ignore", FutureWarning)
        model.fit(x_train, train["labels"])
    classifier = model[-1]
    folds = classifier.scores_
    folds = np.asarray(folds[1] if isinstance(folds, dict) else folds).reshape(5, -1)
    predicted = model.predict(x_test)
    score = model.predict_proba(x_test)[:, 1]
    y = test["labels"]
    confusion = confusion_matrix(y, predicted, labels=[0, 1])
    (tn, fp), (fn, tp) = confusion
    row = {
        "feature_group": GROUP_NAMES[group], "key": group, "stage": GROUP_STAGES[group],
        "n_features": x_train.shape[1],
        "C": float(classifier.C_[0]),
        "cv_accuracy": float(folds.mean(axis=0).max()),
        "train_accuracy": float(accuracy_score(train["labels"], model.predict(x_train))),
        "test_accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return row, confusion, model


def table_summary(data: dict) -> dict:
    """Describe JPEG luminance quantization-table sums per class (-1 = not JPEG)."""
    summary = {}
    for name, label in LABELS.items():
        sums = data["table_sums"][data["labels"] == label]
        values, counts = np.unique(sums, return_counts=True)
        top = np.argsort(counts)[::-1][:5]
        summary[name] = {"distinct_tables": int(len(values)),
                         "min": int(sums.min()), "median": float(np.median(sums)),
                         "max": int(sums.max()),
                         "most_common": {str(int(values[i])): int(counts[i]) for i in top}}
    return summary


def run(train_file: Path, test_file: Path, output_dir: Path, train_per_class: int = 2500,
        test_per_class: int = 1000, seed: int = 42, recompress_quality: int | None = None) -> dict:
    train_file, test_file = Path(train_file).resolve(), Path(test_file).resolve()
    if train_file == test_file:
        raise ValueError("Training and test shards must be different files.")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"train": {}, "test": {}}
    with TemporaryDirectory(prefix="task2-", dir=output_dir) as temporary:
        stage = Path(temporary)
        (stage / "features").mkdir()
        (stage / "figures").mkdir()
        train = load_split(train_file, "train", train_per_class, seed, report["train"],
                           recompress_quality, keep_spectra=True)
        test = load_split(test_file, "test", test_per_class, seed, report["test"],
                          recompress_quality, keep_spectra=False)
        for split, data in (("train", train), ("test", test)):
            np.savez_compressed(stage / "features" / f"{split}_features.npz",
                                records=data["records"], labels=data["labels"],
                                table_sums=data["table_sums"],
                                **{group: data[group] for group in GROUPS})

        # The JPEG control is meaningless after re-encoding (one table for every image).
        groups = [key for key, *_ in EVAL_GROUPS
                  if not (key == "jpeg" and recompress_quality is not None)]
        metrics, confusions, models = [], {}, {}
        for group in groups:
            row, confusion, model = evaluate(train, test, group, seed)
            metrics.append(row)
            confusions[group] = confusion
            models[group] = model
            print(f"{row['feature_group']:<28} test accuracy {row['test_accuracy']:.4f} "
                  f"(AUC {row['roc_auc']:.4f}, C={row['C']:g})")
        with (stage / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)

        by_label = lambda key: {label: train[key][train["labels"] == label]
                                for label in LABELS.values()}
        counts = {label: int(np.sum(train["labels"] == label)) for label in LABELS.values()}
        basic = [row for row in metrics if row["stage"] == "basic"]
        spectral = [row for row in metrics if row["stage"] != "control"]
        best = max(spectral, key=lambda row: row["test_accuracy"])
        figures = stage / "figures"
        plot_real_vs_fake(train_file, figures / "real_vs_fake.png", seed=seed)
        plot_face_examples(train["examples"], figures / "face_examples.png")
        # Keep the class-average spectra so figures can be redrawn without re-extraction.
        np.savez_compressed(stage / "features" / "mean_spectra.npz",
                            **{f"{key}_{name}": train[key][label]
                               for key in ("mean_spectra", "mean_spectra_hann")
                               for name, label in LABELS.items()})
        plot_mean_spectra(train["mean_spectra"], counts, figures / "mean_spectra.png")
        plot_mean_spectra(train["mean_spectra_hann"], counts, figures / "mean_spectra_hann.png",
                          "Class-average FFT spectra with a 2D Hann window, training subset")
        plot_radial(by_label("radial"), figures / "radial_spectrum.png")
        plot_grid_effects(by_label("grid"), train["mean_images"][0], figures / "grid_effects.png")
        plot_classification(basic, confusions["band"], GROUP_NAMES["band"],
                            figures / "classification_basic.png",
                            "Logistic Regression: Task 1 features by transform")
        plot_classification(spectral, confusions[best["key"]], best["feature_group"],
                            figures / "classification.png",
                            "Logistic Regression: all feature groups")
        plot_band_features(FEATURE_COLUMNS, by_label("band"), models["band"][-1].coef_[0],
                           figures / "band_features.png")

        metadata = {
            "dataset": "140k Real and Fake Faces (FFHQ vs StyleGAN), Hugging Face parquet",
            "train_file": train_file.name, "test_file": test_file.name, "seed": seed,
            "labels": LABELS, "counts": report,
            "preprocessing": ("JPEG -> RGB" + (f" -> re-encode JPEG q={recompress_quality}"
                                               if recompress_quality else "")
                              + " -> Pillow L (ITU-R 601) -> float64 / 255; native 256x256"),
            "features": {
                "band": list(FEATURE_COLUMNS),
                "fft": [FEATURE_COLUMNS[i] for i in TRANSFORM_COLUMNS["fft"]],
                "dct": [FEATURE_COLUMNS[i] for i in TRANSFORM_COLUMNS["dct"]],
                "dwt": [FEATURE_COLUMNS[i] for i in TRANSFORM_COLUMNS["dwt"]],
                "radial": "log10 mean FFT power in 1-cycle/image rings, k = 1..127",
                "dct_blocks": (f"log10 |orthonormal DCT-II| averaged in "
                               f"{DCT_BLOCK}x{DCT_BLOCK} blocks"),
                "grid": grid_columns(),
                "combined": " + ".join(COMBINED),
                "radial_hann": "radial, after a unit-power 2D Hann window",
                "radial_shape": "radial_hann minus its own mean over rings",
                "radial_level": "mean of radial_hann over rings (1 value)",
                "jpeg": "luminance quantization-table sum only (compression control)",
            },
            "classifier": ("StandardScaler + LogisticRegressionCV (L2, lbfgs), 5-fold CV "
                           f"accuracy over C in {[float(c) for c in C_GRID]}; training only"),
            "jpeg_tables": {"train": table_summary(train), "test": table_summary(test)},
            "grid": GRID,
            "versions": {name: version(name) for name in
                         ("numpy", "scipy", "PyWavelets", "matplotlib", "Pillow",
                          "pyarrow", "scikit-learn")},
        }
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                             encoding="utf-8")
        for source in stage.rglob("*"):
            if source.is_file():
                target = output_dir / source.relative_to(stage)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    print(f"Saved features, metrics, nine figures, and metadata to {output_dir}")
    return {"metrics": metrics, "metadata": metadata}


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path,
                        default=Path("datasets/140k/validation-00000-of-00002.parquet"))
    parser.add_argument("--test-file", type=Path,
                        default=Path("datasets/140k/test-00000-of-00002.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/faces"))
    parser.add_argument("--train-per-class", type=positive_int, default=2500)
    parser.add_argument("--test-per-class", type=positive_int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recompress-quality", type=positive_int, default=None,
                        help="Re-encode every image as JPEG at this quality (compression control)")
    args = parser.parse_args(argv)
    try:
        run(args.train_file, args.test_file, args.output_dir, args.train_per_class,
            args.test_per_class, args.seed, args.recompress_quality)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
