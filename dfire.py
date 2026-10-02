"""DFire dataset utilities: audit a YOLO format dataset and write the Ultralytics data file.

DFire (Gaia, solutions on demand) holds more than 21,000 images labelled
for fire and smoke in YOLO format. Download it from the links in the
dataset repository (https://github.com/gaiasd/DFireDataset) or from the
Kaggle mirror, and unpack it so that the folder looks like:

    data/dfire/train/images, data/dfire/train/labels
    data/dfire/test/images,  data/dfire/test/labels

The audit is worth running before any training: it catches missing label
files, malformed rows and out of range boxes, and it reports the class
balance and box size distribution that drive the training choices.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_CLASSES = {0: "smoke", 1: "fire"}


def _label_path(image_path: Path) -> Path:
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def audit_split(split_dir: Path, classes: dict[int, str]) -> dict:
    images = sorted(p for p in (split_dir / "images").glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    names = {v: k for k, v in classes.items()}
    summary = {
        "images": len(images), "missing_label_files": 0, "background_images": 0, "malformed_rows": 0, "out_of_range_boxes": 0,
        "boxes": {name: 0 for name in names}, "images_by_content": {"only_fire": 0, "only_smoke": 0, "fire_and_smoke": 0, "none": 0},
        "box_area_fraction": {name: {"under_1_percent": 0, "1_to_10_percent": 0, "over_10_percent": 0} for name in names},
    }
    for image in images:
        label_file = _label_path(image)
        present: set[str] = set()
        if not label_file.exists():
            summary["missing_label_files"] += 1
        else:
            for row in label_file.read_text(encoding="utf-8").splitlines():
                parts = row.split()
                if not parts:
                    continue
                try:
                    cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
                except (ValueError, TypeError):
                    summary["malformed_rows"] += 1
                    continue
                if cls not in classes or len(parts) != 5:
                    summary["malformed_rows"] += 1
                    continue
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    summary["out_of_range_boxes"] += 1
                    continue
                name = classes[cls]
                present.add(name)
                summary["boxes"][name] += 1
                area = w * h
                bucket = "under_1_percent" if area < 0.01 else ("1_to_10_percent" if area < 0.10 else "over_10_percent")
                summary["box_area_fraction"][name][bucket] += 1
        if not present:
            summary["background_images"] += 1
            summary["images_by_content"]["none"] += 1
        elif present == {"fire"}:
            summary["images_by_content"]["only_fire"] += 1
        elif present == {"smoke"}:
            summary["images_by_content"]["only_smoke"] += 1
        else:
            summary["images_by_content"]["fire_and_smoke"] += 1
    return summary


def audit_dataset(root: str | Path, classes: dict[int, str] | None = None, out_path: str | Path | None = None) -> dict:
    root = Path(root)
    classes = classes or DEFAULT_CLASSES
    report = {"root": str(root), "classes": classes, "splits": {}}
    for split in ("train", "val", "test"):
        if (root / split / "images").is_dir():
            report["splits"][split] = audit_split(root / split, classes)
    if not report["splits"]:
        raise FileNotFoundError(f"No train, val or test folders with an images directory were found under {root}")
    report["total_images"] = sum(s["images"] for s in report["splits"].values())
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def make_validation_split(root: str | Path, fraction: float = 0.1, seed: int = 42) -> int:
    """Move a random fraction of the training set into ``val`` when the download has no validation split."""
    root = Path(root)
    if (root / "val" / "images").is_dir():
        return 0
    images = sorted(p for p in (root / "train" / "images").glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    random.Random(seed).shuffle(images)
    chosen = images[: int(len(images) * fraction)]
    (root / "val" / "images").mkdir(parents=True, exist_ok=True)
    (root / "val" / "labels").mkdir(parents=True, exist_ok=True)
    for image in chosen:
        label = _label_path(image)
        shutil.move(str(image), str(root / "val" / "images" / image.name))
        if label.exists():
            shutil.move(str(label), str(root / "val" / "labels" / label.name))
    return len(chosen)


def write_data_yaml(root: str | Path, out_path: str | Path, classes: dict[int, str] | None = None) -> Path:
    root = Path(root).resolve()
    classes = classes or DEFAULT_CLASSES
    data = {
        "path": str(root),
        "train": "train/images",
        "val": "val/images" if (root / "val" / "images").is_dir() else "test/images",
        "test": "test/images",
        "names": {int(k): v for k, v in classes.items()},
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out
