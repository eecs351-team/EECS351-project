"""Schema-checked Hugging Face streaming and bounded, balanced face sampling."""

from io import BytesIO
import random
import warnings

from PIL import Image

from .dataset import LABELS, preprocess_image

DATASET_ID = "RajuFace/140k-Real-and-Fake-Faces"
EXPECTED_SHAPE = (256, 256)


def open_stream(dataset_id: str, split: str, revision: str, cache_dir):
    """Inspect metadata before opening a stream; pin it to a resolved Hub commit.

    Only repository metadata is loaded eagerly. Image decoding is deferred until
    after sampling, so corrupt image bytes cannot terminate IterableDataset iteration.
    """
    from datasets import ClassLabel, Image as HFImage, load_dataset, load_dataset_builder
    from huggingface_hub import HfApi

    try:
        commit = HfApi().dataset_info(dataset_id, revision=revision, timeout=30).sha
        builder = load_dataset_builder(dataset_id, revision=commit, cache_dir=str(cache_dir))
        info = builder.info
        splits = {name: item.num_examples for name, item in (info.splits or {}).items()}
        if split not in splits:
            raise ValueError(f"Split {split!r} unavailable; available splits: {splits}")
        features = info.features
        if features is None or not {"image", "label", "generator"} <= set(features):
            raise ValueError(f"Expected image, label, generator fields; got {features}")
        if not isinstance(features["image"], HFImage):
            raise ValueError("The image field must be datasets.Image")
        labels = features["label"]
        if not isinstance(labels, ClassLabel) or [s.lower() for s in labels.names] != ["real", "fake"]:
            raise ValueError(f"Expected ClassLabel names ['real', 'fake'] (0=REAL, 1=FAKE); got {labels}")
        if not isinstance(features["generator"], ClassLabel):
            raise ValueError("The generator field must be datasets.ClassLabel metadata")
        metadata = {
            "dataset_id": dataset_id, "revision": commit, "split": split,
            "available_splits": splits, "features": features.to_dict(),
            "label_names": labels.names, "generator_names": features["generator"].names,
        }
        print(f"Dataset schema: {metadata}", flush=True)
        stream = load_dataset(dataset_id, split=split, revision=commit,
                              streaming=True, cache_dir=str(cache_dir))
        return stream.cast_column("image", HFImage(decode=False)), metadata
    except ValueError:
        raise
    except Exception as error:
        raise RuntimeError(
            f"Cannot inspect/open Hugging Face dataset {dataset_id!r}, split {split!r}: {error}. "
            "Check network access and dataset permissions; no full-download fallback is used."
        ) from error


def buffered_samples(stream, seed: int, buffer_size: int, max_scan: int):
    """Bounded shuffle over encoded rows, with source indices assigned BEFORE shuffle.

    Unlike IterableDataset.shuffle(), this keeps the original shard order so an
    index always identifies the same row for a fixed dataset revision and split.
    Only the first max_scan source rows can be read, and only buffer_size encoded
    rows are retained. This is a prefix-buffer sample, not a global uniform sample.
    """
    rng = random.Random(seed)
    source = iter(stream)
    buffer = []
    consumed = 0

    def read_one():
        nonlocal consumed
        if consumed >= max_scan:
            raise StopIteration
        row = next(source)
        indexed = (consumed, row)
        consumed += 1
        return indexed

    for _ in range(buffer_size):
        try:
            buffer.append(read_one())
        except StopIteration:
            break
    while buffer:
        position = rng.randrange(len(buffer))
        yield buffer[position]
        try:
            buffer[position] = read_one()
        except StopIteration:
            buffer.pop(position)


def decode_rgb(value) -> Image.Image:
    """Decode one selected row; keep only the two representative RGB images later."""
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    payload = value.get("bytes")
    if payload is None:
        raise ValueError("Streamed image has no embedded bytes; refusing an unverified path")
    with Image.open(BytesIO(payload)) as image:
        return image.convert("RGB")


def iter_balanced(stream, per_class: int, seed: int, buffer_size: int,
                  max_scan: int, generator_names: list[str], report: dict):
    """Yield (source index, label, generator name, gray array, original RGB).

    Invalid labels are fatal. Corrupt/wrong-sized images warn and are replaced by
    later candidates. Stream/network errors abort instead of silently restarting
    and risking duplicate rows. Exhaustion never counts as a successful partial run.
    """
    if per_class < 1 or buffer_size < 1 or max_scan < 2 * per_class:
        raise ValueError("Counts/buffer must be positive and max-scan must be >= 2 * per-class")
    report.update({"requested_per_class": per_class, "processed": {name: 0 for name in LABELS},
                   "candidates_examined": 0, "skipped": []})
    candidates = iter(buffered_samples(stream, seed, buffer_size, max_scan))
    while True:
        try:
            index, row = next(candidates)
        except StopIteration:
            break
        except Exception as error:
            raise RuntimeError(f"Dataset stream failed; actual counts {report['processed']}: {error}") from error
        report["candidates_examined"] += 1
        label = row.get("label")
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise ValueError(f"Source index {index}: invalid binary label {label!r}")
        name = "REAL" if label == 0 else "FAKE"
        if report["processed"][name] == per_class:
            continue
        generator = row.get("generator")
        if isinstance(generator, bool) or not isinstance(generator, int) or not 0 <= generator < len(generator_names):
            raise ValueError(f"Source index {index}: invalid generator metadata {generator!r}")
        try:
            rgb = decode_rgb(row["image"])
            gray = preprocess_image(rgb, EXPECTED_SHAPE)
        except (OSError, ValueError, TypeError, AttributeError, KeyError) as error:
            report["skipped"].append({"sample_id": index, "label": label, "reason": str(error)})
            warnings.warn(f"Skipping source index {index}: {error}", stacklevel=2)
            continue
        report["processed"][name] += 1
        yield index, label, generator_names[generator], gray, rgb
        if all(count == per_class for count in report["processed"].values()):
            return
    raise ValueError(
        f"Cannot collect {per_class} REAL + {per_class} FAKE; actual counts {report['processed']}, "
        f"skipped {len(report['skipped'])}, examined {report['candidates_examined']} candidates. "
        f"Stream exhausted or max-scan={max_scan} reached; no balanced result can be published."
    )
