"""140k Real and Fake Faces parquet shards: balanced selection and preprocessing.

REAL images are FFHQ photographs and FAKE images come from StyleGAN; both are
stored as 256x256 JPEG bytes inside Hugging Face parquet shards.
"""

from io import BytesIO
from pathlib import Path
import random
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError
import pyarrow.parquet as pq

from .dataset import LABELS

IMAGE_SIZE = (256, 256)


def decode_face(data: bytes, recompress_quality: int | None = None
                ) -> tuple[np.ndarray, int | None]:
    """JPEG bytes -> ITU-R 601 L grayscale in [0, 1], plus the luminance table sum.

    The quantization-table sum is a compression fingerprint: a smaller sum means
    finer quantization (higher JPEG quality). With ``recompress_quality`` both
    classes are re-encoded identically before conversion to remove that cue.
    """
    with Image.open(BytesIO(data)) as image:
        if (image.height, image.width) != IMAGE_SIZE:
            raise ValueError(f"Expected {IMAGE_SIZE}, got {(image.height, image.width)}")
        tables = getattr(image, "quantization", None) or {}
        table_sum = int(sum(tables[0])) if 0 in tables else None
        rgb = image.convert("RGB")
    if recompress_quality is not None:
        buffer = BytesIO()
        rgb.save(buffer, format="JPEG", quality=recompress_quality)
        buffer.seek(0)
        with Image.open(buffer) as image:
            table_sum = int(sum(image.quantization[0]))
            rgb = image.convert("RGB")
    return np.asarray(rgb.convert("L"), dtype=np.float64) / 255.0, table_sum


def iter_faces(shard: Path, split: str, per_class: int, seed: int, report: dict,
               recompress_quality: int | None = None):
    """Yield (record id, label, image, table sum) for a balanced, seeded subset.

    The shard's own ``label`` column uses real=0 and fake=1, matching LABELS.
    Corrupt or wrong-sized images are skipped and replaced deterministically.
    """
    shard = Path(shard)
    if per_class < 1:
        raise ValueError("per-class sample counts must be positive")
    table = pq.read_table(shard, columns=["image", "label"])
    labels = table.column("label").to_numpy()
    images = table.column("image")
    for name, label in LABELS.items():
        rows = np.flatnonzero(labels == label).tolist()
        if len(rows) < per_class:
            raise ValueError(f"{shard.name}/{name}: requested {per_class}, found {len(rows)}")
        random.Random(f"{seed}:{split}:{name}").shuffle(rows)
        counts = {"available": len(rows), "requested": per_class,
                  "processed": 0, "skipped": []}
        report[name] = counts
        for row in rows:
            if counts["processed"] == per_class:
                break
            record = f"{shard.name}#{row}"
            try:
                image, table_sum = decode_face(images[row].as_py()["bytes"], recompress_quality)
            except (OSError, UnidentifiedImageError, ValueError) as error:
                counts["skipped"].append({"image_path": record, "reason": str(error)})
                warnings.warn(f"Skipping {record}: {error}", stacklevel=2)
                continue
            counts["processed"] += 1
            yield record, label, image, table_sum
        if counts["processed"] < per_class:
            raise ValueError(f"{shard.name}/{name}: only {counts['processed']} valid images")
