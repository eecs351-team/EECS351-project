"""Spectral-feature and parquet-loader checks with synthetic data only."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from src.faces import decode_face, iter_faces
from src.spectral import (GRID, GRID_FEATURES, dct_log_blocks, grid_columns, grid_features,
                          hann_2d, power_spectrum, radial_spectrum)
from src.dsp_features import FEATURE_COLUMNS
from src.task2 import EVAL_GROUPS, TRANSFORM_COLUMNS, run


def jpeg_bytes(array: np.ndarray, quality: int) -> bytes:
    buffer = BytesIO()
    Image.fromarray(array.astype(np.uint8)).convert("RGB").save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()


def write_shard(path: Path, per_class: int, seed: int) -> None:
    """REAL: smooth noise, quality 95. FAKE: same plus a 0.25 cycle/pixel grid, quality 90."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:256, :256]
    grid = 12 * np.cos(np.pi * x / 2) * np.cos(np.pi * y / 2)
    rows = []
    for index in range(2 * per_class):
        label = index % 2
        base = 128 + 30 * rng.standard_normal((256, 256)).cumsum(axis=1) / 16
        image = np.clip(base + (grid if label else 0), 0, 255)
        rows.append({"image": {"bytes": jpeg_bytes(image, 90 if label else 95), "path": None},
                     "label": label})
    pq.write_table(pa.Table.from_pylist(rows), path)


class SpectralTests(unittest.TestCase):
    def test_radial_peak_matches_grating_frequency(self):
        x = np.arange(256)[None, :]
        image = np.repeat(0.5 + 0.4 * np.cos(2 * np.pi * 40 * x / 256), 256, axis=0)
        spectrum = radial_spectrum(image)
        self.assertEqual(spectrum.shape, (127,))
        self.assertEqual(int(np.argmax(spectrum)) + 1, 40)

    def test_dct_blocks_and_grid_shapes(self):
        image = np.random.default_rng(3).random((256, 256))
        self.assertEqual(dct_log_blocks(image).shape, (1024,))
        features = grid_features(image)
        self.assertEqual(features.shape, (GRID * GRID * len(GRID_FEATURES),))
        self.assertEqual(len(grid_columns()), features.size)
        self.assertTrue(np.all((features >= 0) & (features <= 1)))
        with self.assertRaises(ValueError):
            radial_spectrum(np.zeros((32, 64)))


    def test_hann_window_removes_edge_leakage(self):
        # A horizontal ramp jumps from 1 back to 0 at the periodic edge, which leaks
        # power along the horizontal frequency axis unless the edges are tapered.
        ramp = np.tile(np.linspace(0, 1, 256), (256, 1))
        raw, windowed = power_spectrum(ramp), power_spectrum(ramp, window=True)
        axis = (128, slice(128 + 64, 256))  # fy = 0, fx from 0.25 to 0.5 cycles/pixel
        self.assertLess(windowed[axis].sum(), 1e-3 * raw[axis].sum())
        self.assertAlmostEqual(float(np.mean(hann_2d((256, 256)) ** 2)), 1.0)
        # A pure grating still peaks in the right ring after windowing.
        x = np.arange(256)[None, :]
        grating = np.repeat(0.5 + 0.4 * np.cos(2 * np.pi * 40 * x / 256), 256, axis=0)
        self.assertEqual(int(np.argmax(radial_spectrum(grating, window=True))) + 1, 40)

    def test_transform_groups_partition_band_ratios(self):
        columns = sorted(i for indices in TRANSFORM_COLUMNS.values() for i in indices)
        self.assertEqual(columns, list(range(len(FEATURE_COLUMNS))))
        self.assertEqual({k: len(v) for k, v in TRANSFORM_COLUMNS.items()},
                         {"fft": 3, "dct": 3, "dwt": 5})


class FaceLoaderTests(unittest.TestCase):
    def test_decode_records_quantization_and_recompression(self):
        array = np.random.default_rng(4).integers(0, 255, (256, 256))
        image, high = decode_face(jpeg_bytes(array, 95))
        _, low = decode_face(jpeg_bytes(array, 50))
        self.assertEqual(image.shape, (256, 256))
        self.assertTrue(0 <= image.min() and image.max() <= 1)
        self.assertLess(high, low)
        with self.assertRaises(ValueError):
            decode_face(jpeg_bytes(array[:32, :32], 95))

    def test_balanced_deterministic_selection(self):
        with TemporaryDirectory() as temporary:
            shard = Path(temporary) / "shard.parquet"
            write_shard(shard, 6, 0)
            first = [r for r, *_ in iter_faces(shard, "train", 4, 7, {})]
            second = [r for r, *_ in iter_faces(shard, "train", 4, 7, {})]
            self.assertEqual(first, second)
            report = {}
            labels = [label for _, label, *_ in iter_faces(shard, "train", 4, 7, report)]
            self.assertEqual(labels.count(0), labels.count(1))
            self.assertEqual(report["REAL"]["processed"], 4)
            with self.assertRaises(ValueError):
                list(iter_faces(shard, "train", 7, 7, {}))

    def test_end_to_end_detects_synthetic_grid(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_shard(root / "train.parquet", 30, 1)
            write_shard(root / "test.parquet", 10, 2)
            result = run(root / "train.parquet", root / "test.parquet", root / "out", 30, 10, 0)
            scores = {row["key"]: row["test_accuracy"] for row in result["metrics"]}
            self.assertGreaterEqual(scores["dct_blocks"], 0.9)
            self.assertEqual(list(scores), [key for key, *_ in EVAL_GROUPS])
            self.assertEqual({row["n_features"] for row in result["metrics"]
                              if row["key"] in ("fft", "dct", "dwt", "band")}, {3, 5, 11})
            self.assertEqual(result["metadata"]["counts"]["test"]["FAKE"]["processed"], 10)
            for name in ("metrics.csv", "metadata.json", "features/train_features.npz",
                         "figures/mean_spectra.png", "figures/classification.png",
                         "figures/classification_basic.png", "figures/band_features.png",
                         "figures/mean_spectra_hann.png", "figures/real_vs_fake.png"):
                self.assertTrue((root / "out" / name).is_file(), name)
            recompressed = run(root / "train.parquet", root / "test.parquet", root / "re",
                               30, 10, 0, recompress_quality=85)
            self.assertNotIn("jpeg", [row["key"] for row in recompressed["metrics"]])
            tables = recompressed["metadata"]["jpeg_tables"]["train"]
            self.assertEqual(tables["REAL"]["distinct_tables"], 1)


if __name__ == "__main__":
    unittest.main()
