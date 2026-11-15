"""Native-resolution RajuFace DSP analysis; streaming only, no classifier training."""

import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .analysis import plot_distributions, plot_transforms, summarize_training
from .dataset import LABELS
from .dsp_features import FEATURE_COLUMNS, DWT_LEVEL, DWT_MODE, WAVELET, extract_features
from .rajuface_dataset import DATASET_ID, iter_balanced, open_stream
from .task1 import positive_int

CSV_COLUMNS = ("dataset_id", "split", "sample_id", "label", "generator", *FEATURE_COLUMNS)


def run(output_dir: Path, per_class: int = 10, seed: int = 42, *,
        dataset_id: str = DATASET_ID, split: str = "test", revision: str = "main",
        buffer_size: int = 128, max_scan: int = 10000,
        cache_dir: Path = Path(".cache/huggingface")) -> dict:
    """Stage a complete balanced experiment; refuse to replace any existing run."""
    if per_class < 1 or buffer_size < 1 or max_scan < 2 * per_class:
        raise ValueError("Counts/buffer must be positive and max-scan must be >= 2 * per-class")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError(f"Output already exists: {output_dir}. Choose a new --output-dir to preserve results.")
    # split is a schema key, never a filename: prevent user-supplied path traversal.
    if not split or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in split):
        raise ValueError("Split must be a simple name (letters, digits, underscore, hyphen)")
    stream, source_metadata = open_stream(dataset_id, split, revision, cache_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    report = {}
    groups = {0: [], 1: []}
    representatives, originals = {}, {}
    # Rename only after successful extraction/plotting; no partial CSV is published.
    with TemporaryDirectory(prefix="rajuface-", dir=output_dir.parent) as temporary:
        stage = Path(temporary) / "result"
        (stage / "features").mkdir(parents=True)
        (stage / "figures").mkdir()
        with (stage / "features" / f"{split}_features.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for index, label, generator, gray, rgb in iter_balanced(
                    stream, per_class, seed, buffer_size, max_scan,
                    source_metadata["generator_names"], report):
                features = extract_features(gray)
                writer.writerow(dict(dataset_id=dataset_id, split=split, sample_id=index,
                                     label=label, generator=generator, **features))
                groups[label].append([features[name] for name in FEATURE_COLUMNS])
                if label not in representatives:
                    representatives[label] = (f"{split}:{index}", gray)
                    originals[label] = rgb
        groups = {label: np.asarray(values) for label, values in groups.items()}
        summarize_training(groups, stage / "features" / f"{split}_summary.csv")
        plot_transforms(representatives, stage / "figures" / "transform_comparison.png",
                        originals=originals,
                        title=f"RajuFace {split}: first selected REAL / FAKE, native 256x256")
        plot_distributions(groups, stage / "figures" / "feature_distributions.png",
                           title=f"RajuFace {split} exploratory subset; common bins, per-class normalization")
        metadata = {
            **source_metadata, "labels": LABELS, "seed": seed, "streaming": True,
            "sampling": {"method": "seeded bounded shuffle of encoded source-prefix rows, then class quotas",
                         "buffer_size": buffer_size, "max_scan": max_scan,
                         "bias": "not a uniform random sample of the full split; bounded prefix/shard bias"},
            "counts": report,
            "preprocessing": {"mode": "native", "shape": [256, 256],
                              "grayscale": "RGB -> Pillow L (ITU-R 601, 8-bit) -> float64 / 255",
                              "resize": False},
            "feature_columns": FEATURE_COLUMNS,
            "fft": "non-DC squared magnitude; radial [0,.15), [.15,.35), [.35,inf) cycles/pixel",
            "dct": "orthonormal type II; AC radius hypot(k/(2H),l/(2W)); low <.15; high >=.15; concentration sum(p_AC**2)",
            "dwt": {"wavelet": WAVELET, "level": DWT_LEVEL, "mode": DWT_MODE,
                    "normalization": "total squared detail coefficients; approximation excluded",
                    "orientations": "PyWavelets cH, cV, cD; level1=fine, level2=coarse"},
            "zero_energy": "total <= 1e-24: corresponding ratios/concentration = 0",
            "summary": "exploratory selected split; sample std ddof=1, singleton std=0; no tuning",
            "representatives": {str(label): value[0] for label, value in representatives.items()},
            "classification": {"status": "skipped", "reason": "No trained CIFAKE classifier or saved model preprocessing exists in this repository."},
            "versions": {name: version(name) for name in
                         ("numpy", "scipy", "PyWavelets", "matplotlib", "Pillow", "datasets")},
        }
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        stage.rename(output_dir)
    print(f"Saved {report['processed']} native 256x256 images, CSVs, two figures, metadata to {output_dir}")
    print("Classifier evaluation skipped: no trained CIFAKE classifier/configuration found.")
    return metadata


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--split", default="test")
    parser.add_argument("--revision", default="main", help="Hub revision; resolved commit is recorded")
    parser.add_argument("--per-class", type=positive_int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=positive_int, default=128)
    parser.add_argument("--max-scan", type=positive_int, default=10000,
                        help="Maximum source rows fetched; limits scanning of class-ordered streams")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rajuface/demo"))
    args = parser.parse_args(argv)
    try:
        run(args.output_dir, args.per_class, args.seed, dataset_id=args.dataset_id,
            split=args.split, revision=args.revision, buffer_size=args.buffer_size,
            max_scan=args.max_scan, cache_dir=args.cache_dir)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
