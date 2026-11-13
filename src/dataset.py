"""CIFAKE paths, explicit labels, balanced selection, and preprocessing."""

from pathlib import Path
import random
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError

LABELS = {"REAL": 0, "FAKE": 1}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_dataset(root: Path) -> Path:
    root = Path(root).expanduser().resolve()
    missing = [f"{split}/{name}" for split in ("train", "test") for name in LABELS
               if not (root / split / name).is_dir()]
    if missing:
        raise ValueError(
            f"Invalid CIFAKE root: {root}. Missing: {', '.join(missing)}. "
            "Point --data-dir to the extracted folder containing "
            "train/REAL, train/FAKE, test/REAL, test/FAKE (check enclosing folders)."
        )
    folders = [(root / split / name).resolve()
               for split in ("train", "test") for name in LABELS]
    if len(set(folders)) != 4 or any(root not in folder.parents for folder in folders):
        raise ValueError("Split/class folders must be distinct and remain within the dataset root.")
    return root


def load_image(path: Path, expected_shape: tuple[int, int] = (32, 32)) -> np.ndarray:
    """Pillow RGB -> ITU-R 601 L grayscale, then float64 / 255."""
    with Image.open(path) as image:
        if (image.height, image.width) != expected_shape:
            raise ValueError(f"Expected {expected_shape}, got {(image.height, image.width)}")
        return np.asarray(image.convert("RGB").convert("L"), dtype=np.float64) / 255.0


def iter_images(root: Path, split: str, per_class: int | None, seed: int,
                report: dict):
    """Yield valid images incrementally; replace corrupt selections deterministically.

    None requests all images and requires equal available class counts. Exhaustion
    raises an error, so an unbalanced result is never published as a successful run.
    """
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    if per_class is not None and per_class < 1:
        raise ValueError("per-class sample counts must be positive")
    paths = {name: sorted(p for p in (root / split / name).iterdir()
                          if p.is_file() and p.suffix.lower() in EXTENSIONS)
             for name in LABELS}
    if per_class is None:
        if len(paths["REAL"]) != len(paths["FAKE"]):
            raise ValueError(f"{split}: --all requires equal available class counts")
        per_class = len(paths["REAL"])
    if per_class < 1:
        raise ValueError(f"{split}: no images available")
    for name, files in paths.items():
        if len(files) < per_class:
            raise ValueError(f"{split}/{name}: requested {per_class}, found {len(files)} images")
    for name, label in LABELS.items():
        files = paths[name]
        random.Random(f"{seed}:{split}:{name}").shuffle(files)
        counts = {"available": len(files), "requested": per_class,
                  "processed": 0, "skipped": []}
        report[name] = counts
        for path in files:
            # Reject links pointing to another split or outside this class.
            if path.resolve().parent != (root / split / name).resolve():
                raise ValueError(f"Image path escapes its split/class folder: {path}")
            try:
                image = load_image(path)
            except (OSError, UnidentifiedImageError, ValueError) as error:
                relative = path.relative_to(root).as_posix()
                counts["skipped"].append({"image_path": relative, "reason": str(error)})
                warnings.warn(f"Skipping {relative}: {error}", stacklevel=2)
                continue
            counts["processed"] += 1
            yield path.relative_to(root).as_posix(), label, image
            if counts["processed"] == per_class:
                break
        if counts["processed"] < per_class:
            raise ValueError(
                f"{split}/{name}: only {counts['processed']} valid images for "
                f"{per_class} requested; skipped {len(counts['skipped'])}. "
                "No balanced output can be completed."
            )
