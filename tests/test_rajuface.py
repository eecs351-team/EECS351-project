"""Offline tests using real HF IterableDataset objects and synthetic 256x256 images."""

import csv
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

from datasets import ClassLabel, Features, Image as HFImage, IterableDataset
import numpy as np
from PIL import Image
import pywt

from src.dataset import preprocess_image
from src.dsp_features import (FEATURE_COLUMNS, extract_features, fft_transform,
                              dct_transform, dwt_transform)
from src.rajuface import CSV_COLUMNS, run
from src.rajuface_dataset import buffered_samples, iter_balanced, open_stream

SCHEMA = Features({"image": HFImage(decode=False), "label": ClassLabel(names=["real", "fake"]),
                   "generator": ClassLabel(names=["Real", "StyleGAN"])})
SOURCE = {"dataset_id": "synthetic-test-only", "revision": "offline-fixture", "split": "test",
          "available_splits": {"test": 40}, "features": SCHEMA.to_dict(),
          "label_names": ["real", "fake"], "generator_names": ["Real", "StyleGAN"]}


def make_rows(count=40):
    rng = np.random.default_rng(17)
    for index in range(count):
        handle = BytesIO()
        Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(handle, format="PNG")
        yield {"image": {"bytes": handle.getvalue(), "path": None},
               "label": index % 2, "generator": index % 2}


def make_stream():
    # HF creates a builder cache even for an entirely synthetic iterable.
    with TemporaryDirectory() as cache, patch("datasets.config.HF_DATASETS_CACHE", Path(cache)):
        return IterableDataset.from_generator(make_rows, features=SCHEMA)


def collect(stream=None, count=3, seed=42, buffer=8, max_scan=40):
    report = {}
    records = list(iter_balanced(make_stream() if stream is None else stream, count,
                                seed, buffer, max_scan, SOURCE["generator_names"], report))
    return records, report


class NativeDSPTests(unittest.TestCase):
    def test_preprocessing_exact_grayscale_native_and_wrong_shape(self):
        gray = preprocess_image(Image.new("RGB", (256, 256), (255, 0, 0)), (256, 256))
        self.assertEqual(gray.shape, (256, 256))
        self.assertEqual(gray.dtype, np.float64)
        np.testing.assert_allclose(gray, 76 / 255)
        with self.assertRaisesRegex(ValueError, "Expected"):
            preprocess_image(Image.new("RGB", (32, 32)), (256, 256))

    def test_256_transform_shapes_energy_and_wavelet_reconstruction(self):
        image = np.random.default_rng(4).random((256, 256))
        fft, dct, dwt = fft_transform(image), dct_transform(image), dwt_transform(image)
        self.assertEqual(fft.shape, image.shape)
        self.assertEqual(dct.shape, image.shape)
        np.testing.assert_allclose(np.sum(abs(fft) ** 2) / image.size, np.sum(image ** 2))
        np.testing.assert_allclose(np.sum(dct ** 2), np.sum(image ** 2))
        self.assertEqual(dwt[0].shape, (69, 69))
        self.assertTrue(all(band.shape == (69, 69) for band in dwt[1]))
        self.assertTrue(all(band.shape == (131, 131) for band in dwt[2]))
        np.testing.assert_allclose(pywt.waverec2(dwt, "db4", mode="symmetric"), image, atol=1e-12)
        features = extract_features(image)
        self.assertEqual(tuple(features), FEATURE_COLUMNS)
        self.assertEqual(features, extract_features(image))
        self.assertTrue(all(np.isfinite(v) and 0 <= v <= 1 for v in features.values()))
        for names in (FEATURE_COLUMNS[:3], FEATURE_COLUMNS[3:5], FEATURE_COLUMNS[6:9], FEATURE_COLUMNS[9:]):
            self.assertAlmostEqual(sum(features[n] for n in names), 1)
        for value in (0, .5, 1):
            np.testing.assert_array_equal(list(extract_features(np.full((256, 256), value)).values()), 0)

    def test_frequency_bands_remain_in_cycles_per_pixel_at_256(self):
        x = np.arange(256)[None, :]
        for cycles, band in ((16, "low"), (64, "mid"), (112, "high")):
            image = np.repeat(.5 + .4 * np.cos(2 * np.pi * cycles * x / 256), 256, axis=0)
            self.assertAlmostEqual(extract_features(image)[f"fft_{band}_energy_ratio"], 1)
        for coefficient, band in ((32, "low"), (160, "high")):
            image = np.repeat(.5 + .4 * np.cos(np.pi * (x + .5) * coefficient / 256), 256, axis=0)
            features = extract_features(image)
            self.assertAlmostEqual(features[f"dct_{band}_energy_ratio"], 1)
            self.assertAlmostEqual(features["dct_ac_concentration"], 1)


class StreamingTests(unittest.TestCase):
    def test_actual_iterable_balance_reproducibility_and_source_identity(self):
        first, report = collect()
        second, _ = collect()
        self.assertEqual([(r[0], r[1]) for r in first], [(r[0], r[1]) for r in second])
        self.assertEqual(report["processed"], {"REAL": 3, "FAKE": 3})
        self.assertEqual(len({r[0] for r in first}), 6)
        alternate, _ = collect(seed=16, buffer=16)
        self.assertNotEqual([r[0] for r in first], [r[0] for r in alternate])
        for index, label, generator, gray, rgb in first + alternate:
            self.assertEqual(label, index % 2)
            self.assertEqual(generator, SOURCE["generator_names"][label])
            self.assertEqual(gray.shape, (256, 256))
            self.assertEqual(rgb.size, (256, 256))

    def test_bounded_consumption_and_shuffle_is_a_permutation(self):
        touched = []
        def source():
            for index in range(100):
                touched.append(index)
                yield {"value": index}
        shuffled = buffered_samples(source(), 42, 4, 12)
        selected = next(shuffled)
        self.assertEqual(len(touched), 4)  # no eager full-source collection
        rows = [selected, *shuffled]
        self.assertEqual(len(touched), 12)
        self.assertEqual(sorted(index for index, _ in rows), list(range(12)))
        self.assertTrue(all(index == row["value"] for index, row in rows))

    def test_corrupt_and_wrong_shape_are_replaced_and_counted(self):
        rows = list(make_rows(10))
        rows[0]["image"]["bytes"] = b"not an image"
        rows[1]["image"] = Image.new("RGB", (32, 32))
        with warnings.catch_warnings(record=True) as caught:
            records, report = collect(rows, count=2, buffer=1)
        self.assertEqual(len(caught), 2)
        self.assertEqual(len(report["skipped"]), 2)
        self.assertEqual(report["processed"], {"REAL": 2, "FAKE": 2})
        self.assertEqual(len(records), 4)

    def test_exhaustion_class_order_and_scan_limit_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "actual counts.*REAL.*2.*FAKE.*2"):
            collect(make_rows(4))
        rows = list(make_rows(12))
        for row in rows[:8]:
            row["label"] = 0
        with self.assertRaisesRegex(ValueError, "max-scan=6"):
            collect(rows, count=2, buffer=2, max_scan=6)
        for kwargs in ({"count": 0}, {"buffer": 0}, {"count": 3, "max_scan": 5}):
            with self.assertRaisesRegex(ValueError, "positive"):
                collect(**kwargs)

    def test_invalid_labels_and_generator_are_fatal(self):
        for column, value in (("label", 2), ("label", True), ("label", .0), ("generator", 8)):
            rows = list(make_rows(4))
            rows[0][column] = value
            with self.subTest(column=column, value=value), self.assertRaisesRegex(ValueError, "invalid"):
                collect(rows, count=1, buffer=1)

    def test_network_failure_does_not_restart_or_succeed_partially(self):
        def interrupted():
            yield from make_rows(2)
            raise ConnectionError("synthetic network interruption")
        with self.assertRaisesRegex(RuntimeError, "actual counts.*network interruption"):
            collect(interrupted(), buffer=1)

    def test_schema_mapping_splits_and_streaming_call(self):
        info = SimpleNamespace(features=SCHEMA, splits={"test": SimpleNamespace(num_examples=40)})
        with patch("huggingface_hub.HfApi") as api, \
             patch("datasets.load_dataset_builder") as builder, \
             patch("datasets.load_dataset") as loader:
            api.return_value.dataset_info.return_value.sha = "pinned-commit"
            builder.return_value.info = info
            loader.return_value = make_stream()
            stream, metadata = open_stream("test/source", "test", "main", Path(".cache"))
            self.assertIsInstance(stream, IterableDataset)
            self.assertEqual(metadata["revision"], "pinned-commit")
            self.assertEqual(metadata["available_splits"], {"test": 40})
            self.assertTrue(loader.call_args.kwargs["streaming"])
            self.assertEqual(loader.call_args.kwargs["revision"], "pinned-commit")
            with self.assertRaisesRegex(ValueError, "available splits"):
                open_stream("test/source", "missing", "main", Path(".cache"))
            bad = Features(dict(SCHEMA))
            bad["label"] = ClassLabel(names=["fake", "real"])
            info.features = bad
            with self.assertRaisesRegex(ValueError, "0=REAL"):
                open_stream("test/source", "test", "main", Path(".cache"))


class ExportTests(unittest.TestCase):
    def test_small_synthetic_run_csv_figures_summary_metadata_and_overwrite_guard(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "synthetic"
            with patch("src.rajuface.open_stream", return_value=(make_stream(), SOURCE)):
                metadata = run(output, per_class=3, buffer_size=8, max_scan=40)
            self.assertEqual(metadata["counts"]["processed"], {"REAL": 3, "FAKE": 3})
            with (output / "features/test_features.csv").open(newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames), CSV_COLUMNS)
                rows = list(reader)
            self.assertEqual(len(rows), 6)
            self.assertEqual(len({row["sample_id"] for row in rows}), 6)
            self.assertEqual({row["generator"] for row in rows}, {"Real", "StyleGAN"})
            self.assertTrue(all(np.isfinite([float(row[c]) for c in FEATURE_COLUMNS]).all() for row in rows))
            with (output / "features/test_summary.csv").open(newline="") as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(len(summary), 22)
            for record in summary:
                values = [float(r[record["feature"]]) for r in rows if r["label"] == record["label"]]
                self.assertAlmostEqual(float(record["mean"]), np.mean(values))
                self.assertAlmostEqual(float(record["std"]), np.std(values, ddof=1))
            figures = list((output / "figures").glob("*.png"))
            self.assertEqual(len(figures), 2)
            for figure in figures:
                with Image.open(figure) as image:
                    image.verify()
            self.assertFalse((output / "evaluation/metrics.json").exists())
            before = (output / "metadata.json").read_bytes()
            self.assertEqual(json.loads(before)["classification"]["status"], "skipped")
            with self.assertRaisesRegex(ValueError, "already exists"):
                run(output)
            self.assertEqual((output / "metadata.json").read_bytes(), before)

    def test_failed_stream_and_bad_split_publish_nothing(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "incomplete"
            with patch("src.rajuface.open_stream", return_value=(make_rows(2), SOURCE)):
                with self.assertRaisesRegex(ValueError, "actual counts"):
                    run(output, per_class=2, buffer_size=2, max_scan=4)
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, "simple name"):
                run(output, split="../escape")


if __name__ == "__main__":
    unittest.main()
