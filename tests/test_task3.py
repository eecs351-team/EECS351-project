"""Classifier-comparison checks on small synthetic feature files (no images needed)."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from src.task3 import CLASSIFIER_NAMES, FEATURE_SETS, PCA_COMPONENTS, run

SHAPES = {"band": 11, "radial": 127, "radial_hann": 127, "dct_blocks": 1024, "grid": 32}


def write_features(path: Path, per_class: int, seed: int) -> None:
    """Gaussian features; FAKE is shifted in a few columns of every array."""
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1], per_class)
    arrays = {}
    for name, width in SHAPES.items():
        values = rng.standard_normal((labels.size, width))
        values[:, :3] += 1.5 * labels[:, None]
        arrays[name] = values
    np.savez_compressed(path, labels=labels, table_sums=np.full(labels.size, 369),
                        records=np.array([f"row{i}" for i in range(labels.size)]), **arrays)


class Task3Tests(unittest.TestCase):
    def test_all_classifiers_on_all_sets(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "features").mkdir()
            write_features(root / "features" / "train_features.npz", 150, 0)
            write_features(root / "features" / "test_features.npz", 50, 1)
            result = run(root / "features", root / "out", seed=0)
            metrics = result["metrics"]
            self.assertEqual(len(metrics), len(FEATURE_SETS) * len(CLASSIFIER_NAMES))
            for row in metrics:
                self.assertGreaterEqual(row["test_accuracy"], 0.8, row)
                self.assertEqual(row["tn"] + row["fp"] + row["fn"] + row["tp"], 100)
                json.loads(row["best_params"])
            inputs = {row["set_key"]: row["n_inputs"] for row in metrics}
            self.assertEqual(inputs, {"band": 11, "radial_hann": 127,
                                      "combined": PCA_COMPONENTS})
            with (root / "out" / "metrics.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), len(metrics))
            self.assertTrue((root / "out" / "figures" / "classifiers.png").is_file())
            self.assertTrue((root / "out" / "metadata.json").is_file())

    def test_missing_features_fail_clearly(self):
        with TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                run(Path(temporary), Path(temporary) / "out")


if __name__ == "__main__":
    unittest.main()
