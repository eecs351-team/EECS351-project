"""Small mathematical and data-integrity checks; no CIFAKE download required."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import warnings

import numpy as np
from PIL import Image
import pywt

from src.dataset import LABELS, iter_images, load_image, validate_dataset
from src.dsp_features import (
    FEATURE_COLUMNS, extract_features, fft_transform, dct_transform, dwt_transform,
)
from src.task1 import CSV_COLUMNS, run


class DSPTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(12).random((32, 32))

    def test_shapes_and_energy_normalization(self):
        fourier = fft_transform(self.image)
        cosine = dct_transform(self.image)
        self.assertEqual(fourier.shape, self.image.shape)
        self.assertEqual(cosine.shape, self.image.shape)
        self.assertTrue(np.isfinite(cosine).all())
        np.testing.assert_allclose(np.sum(abs(fourier) ** 2) / self.image.size,
                                   np.sum(self.image ** 2))
        np.testing.assert_allclose(np.sum(cosine ** 2), np.sum(self.image ** 2))

    def test_radial_bands_match_sinusoidal_frequencies(self):
        x = np.arange(32)[None, :]
        for cycles, band in ((2, "low"), (8, "mid"), (14, "high")):
            with self.subTest(cycles=cycles):
                image = np.repeat(0.5 + 0.4 * np.cos(2 * np.pi * cycles * x / 32), 32, axis=0)
                features = extract_features(image)
                self.assertAlmostEqual(features[f"fft_{band}_energy_ratio"], 1)
                peaks = np.argpartition(abs(fft_transform(image))[16], -3)[-3:]
                self.assertEqual(set(peaks), {16, 16 - cycles, 16 + cycles})

    def test_single_dct_basis_concentration(self):
        x = np.arange(32)[None, :]
        for coefficient, band in ((4, "low"), (20, "high")):
            image = np.repeat(0.5 + 0.4 * np.cos(np.pi * (x + 0.5) * coefficient / 32), 32, axis=0)
            features = extract_features(image)
            self.assertAlmostEqual(features[f"dct_{band}_energy_ratio"], 1)
            self.assertAlmostEqual(features["dct_ac_concentration"], 1)

    def test_ratio_groups_and_determinism(self):
        features = extract_features(self.image)
        self.assertEqual(tuple(features), FEATURE_COLUMNS)
        self.assertEqual(features, extract_features(self.image))
        self.assertTrue(all(np.isfinite(value) and 0 <= value <= 1 for value in features.values()))
        for names in (FEATURE_COLUMNS[:3], FEATURE_COLUMNS[3:5],
                      FEATURE_COLUMNS[6:9], FEATURE_COLUMNS[9:]):
            self.assertAlmostEqual(sum(features[name] for name in names), 1)

    def test_constant_and_zero_signals(self):
        for value in (0, 0.5, 1):
            features = extract_features(np.full((32, 32), value))
            np.testing.assert_array_equal(list(features.values()), np.zeros(len(FEATURE_COLUMNS)))

    def test_dwt_structure_and_reconstruction(self):
        coefficients = dwt_transform(self.image)
        self.assertEqual(len(coefficients), 3)
        self.assertEqual(coefficients[0].shape, (13, 13))
        for detail, size in zip(coefficients[1:], (13, 19)):
            self.assertEqual(len(detail), 3)
            self.assertTrue(all(band.shape == (size, size) for band in detail))
        np.testing.assert_allclose(pywt.waverec2(coefficients, "db4", mode="symmetric"), self.image,
                                   atol=1e-12)

    def test_orientation_convention(self):
        # Variation only across columns is PyWavelets' cV detail.
        stripe = np.tile(np.arange(32) % 2, (32, 1)).astype(float)
        features = extract_features(stripe)
        self.assertAlmostEqual(features["dwt_vertical_energy_ratio"], 1)

    def test_larger_rectangular_input_keeps_schema(self):
        image = np.random.default_rng(0).random((64, 96))
        self.assertEqual(tuple(extract_features(image)), FEATURE_COLUMNS)

    def test_invalid_inputs_and_wavelet_level(self):
        invalid = [np.zeros((32, 32, 3)), np.full((32, 32), np.nan),
                   np.full((32, 32), np.inf), np.full((32, 32), -0.1),
                   np.full((32, 32), 2), np.zeros((8, 8))]
        for image in invalid:
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                extract_features(image)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "CIFAKE"
        rng = np.random.default_rng(3)
        for split in ("train", "test"):
            for name in LABELS:
                folder = self.root / split / name
                folder.mkdir(parents=True)
                for index in range(4):
                    pixels = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
                    Image.fromarray(pixels).save(folder / f"{index}.png")

    def collect(self, split="train", count=2, seed=42):
        report = {}
        images = list(iter_images(validate_dataset(self.root), split, count, seed, report))
        return images, report

    def test_labels_balance_determinism_and_partitions(self):
        self.assertEqual(LABELS, {"REAL": 0, "FAKE": 1})
        first, _ = self.collect()
        second, _ = self.collect()
        self.assertEqual([(p, label) for p, label, _ in first],
                         [(p, label) for p, label, _ in second])
        self.assertEqual([label for _, label, _ in first], [0, 0, 1, 1])
        test, _ = self.collect("test")
        self.assertTrue({p for p, _, _ in first}.isdisjoint(p for p, _, _ in test))
        self.assertTrue(all(f"/{'REAL' if label == 0 else 'FAKE'}/" in p for p, label, _ in first))

    def test_preprocessing_preserves_dimensions_and_grayscale(self):
        path = self.root / "train/REAL/red.png"
        Image.new("RGB", (32, 32), (255, 0, 0)).save(path)
        image = load_image(path)
        self.assertEqual(image.shape, (32, 32))
        self.assertEqual(image.dtype, np.float64)
        np.testing.assert_allclose(image, 76 / 255)
        Image.new("RGB", (64, 64)).save(path)
        with self.assertRaisesRegex(ValueError, "Expected"):
            load_image(path)

    def test_missing_insufficient_and_invalid_counts(self):
        with self.assertRaisesRegex(ValueError, "train/REAL"):
            validate_dataset(self.base / "absent")
        with self.assertRaisesRegex(ValueError, "requested 5, found 4"):
            self.collect(count=5)
        with self.assertRaisesRegex(ValueError, "positive"):
            self.collect(count=0)

    def test_corruption_replacement_and_accounting(self):
        selected, _ = self.collect()
        (self.root / selected[0][0]).write_bytes(b"corrupt")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            images, report = self.collect()
        self.assertEqual(len(images), 4)
        self.assertEqual(len(caught), 1)
        self.assertEqual(len(report["REAL"]["skipped"]), 1)
        self.assertEqual(report["REAL"]["processed"], 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(ValueError, "only 3 valid images"):
                self.collect(count=4)

    def test_all_images_and_unbalanced_all_error(self):
        images, _ = self.collect(count=None)
        self.assertEqual(len(images), 8)
        (self.root / "train/REAL/0.png").unlink()
        with self.assertRaisesRegex(ValueError, "equal available"):
            self.collect(count=None)

    def test_export_schema_finiteness_figures_and_training_summary(self):
        output = self.base / "outputs"
        metadata = run(self.root, output, 3, 1, 42)
        identifiers = []
        for split, count in (("train", 6), ("test", 2)):
            with (output / "features" / f"{split}_features.csv").open(newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(tuple(reader.fieldnames), CSV_COLUMNS)
                rows = list(reader)
            self.assertEqual(len(rows), count)
            self.assertEqual({row["split"] for row in rows}, {split})
            paths = [row["image_path"] for row in rows]
            self.assertEqual(len(set(paths)), count)
            identifiers.append(set(paths))
            for row in rows:
                self.assertTrue(np.isfinite([float(row[name]) for name in FEATURE_COLUMNS]).all())
        self.assertTrue(identifiers[0].isdisjoint(identifiers[1]))
        with (output / "features/train_summary.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2 * len(FEATURE_COLUMNS))
        self.assertEqual({row["count"] for row in rows}, {"3"})
        self.assertEqual(metadata["labels"], LABELS)
        self.assertEqual(len(list((output / "figures").glob("*.png"))), 3)

    def test_failure_does_not_publish_partial_outputs(self):
        output = self.base / "incomplete"
        with self.assertRaisesRegex(ValueError, "requested 5"):
            run(self.root, output, 2, 5)
        self.assertFalse((output / "features/train_features.csv").exists())


if __name__ == "__main__":
    unittest.main()
