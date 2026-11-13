"""Run balanced CIFAKE extraction and training-only DSP analysis."""

import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np

from .analysis import (plot_transforms, plot_distributions, plot_sampling_experiment,
                       summarize_training)
from .dataset import LABELS, iter_images, validate_dataset
from .dsp_features import (FEATURE_COLUMNS, DWT_LEVEL, DWT_MODE, WAVELET, extract_features)

CSV_COLUMNS = ("image_path", "split", "label", *FEATURE_COLUMNS)


def run(data_dir: Path, output_dir: Path, train_per_class: int | None = 1000,
        test_per_class: int | None = 200, seed: int = 42) -> dict:
    root = validate_dataset(data_dir)
    output_dir = Path(output_dir).resolve()
    # Outputs must never overwrite or be mixed into the input dataset.
    if output_dir == root or root in output_dir.parents:
        raise ValueError("Output directory must be outside the CIFAKE dataset root.")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    groups = {0: [], 1: []}
    representatives = {}
    seen = set()
    # Publish only after both balanced splits and figures have completed.
    with TemporaryDirectory(prefix="task1-", dir=output_dir) as temporary:
        stage = Path(temporary)
        (stage / "features").mkdir()
        (stage / "figures").mkdir()
        for split, count in (("train", train_per_class), ("test", test_per_class)):
            report[split] = {}
            with (stage / "features" / f"{split}_features.csv").open(
                    "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                for relative, label, image in iter_images(root, split, count, seed, report[split]):
                    identity = (root / relative).resolve()
                    if identity in seen:
                        raise ValueError(f"Duplicate/overlapping image record: {relative}")
                    seen.add(identity)
                    features = extract_features(image)
                    writer.writerow(dict(image_path=relative, split=split, label=label, **features))
                    if split == "train":
                        groups[label].append(list(features.values()))
                        representatives.setdefault(label, (relative, image))
            processed = sum(item["processed"] for item in report[split].values())
            skipped = sum(len(item["skipped"]) for item in report[split].values())
            print(f"{split}: processed {processed} balanced images; skipped {skipped}")

        groups = {label: np.asarray(values) for label, values in groups.items()}
        summarize_training(groups, stage / "features" / "train_summary.csv")
        plot_transforms(representatives, stage / "figures" / "transforms.png")
        plot_distributions(groups, stage / "figures" / "train_distributions.png")
        plot_sampling_experiment(stage / "figures" / "filtering_sampling.png")
        metadata = {
            "dataset": "CIFAKE", "seed": seed, "labels": LABELS, "counts": report,
            "preprocessing": "RGB -> Pillow L (ITU-R 601, 8-bit) -> float64 / 255; native 32x32",
            "fft": "squared magnitude; no DC; radial bands [0,.15), [.15,.35), [.35,inf) cycles/pixel",
            "dct": "orthonormal type II; AC radius k/(2N); low <.15, high >=.15; concentration sum(p_AC**2)",
            "dwt": {"wavelet": WAVELET, "level": DWT_LEVEL, "mode": DWT_MODE,
                    "normalization": "sum of all squared detail coefficients; approximation excluded",
                    "orientations": "PyWavelets cH, cV, cD; level1=fine, level2=coarse"},
            "zero_energy": "total <= 1e-24: all corresponding ratios/concentration are zero",
            "feature_columns": FEATURE_COLUMNS,
            "summary": "training only; sample std (ddof=1); singleton std=0",
            "representatives": {str(k): v[0] for k, v in representatives.items()},
            "versions": {name: version(name) for name in
                         ("numpy", "scipy", "PyWavelets", "matplotlib", "Pillow")},
        }
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        for source in stage.rglob("*"):
            if source.is_file():
                target = output_dir / source.relative_to(stage)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    print(f"Saved CSVs, training statistics, three figures, and metadata to {output_dir}")
    return metadata


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("sample counts must be positive")
    return number


def main(argv=None, *, demo: bool = False) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/CIFAKE"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo" if demo else "outputs"))
    parser.add_argument("--train-per-class", type=positive_int, default=5 if demo else 1000)
    parser.add_argument("--test-per-class", type=positive_int, default=2 if demo else 200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="Process every image; require balanced classes")
    args = parser.parse_args(argv)
    try:
        run(args.data_dir, args.output_dir,
            None if args.all else args.train_per_class,
            None if args.all else args.test_per_class, args.seed)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
