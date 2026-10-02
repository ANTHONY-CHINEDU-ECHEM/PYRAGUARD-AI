"""Fine tune, validate and export the learned detector with Ultralytics YOLO."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from pyraguard.config import Settings, load_settings
from pyraguard.data.dfire import audit_dataset, make_validation_split, write_data_yaml
from pyraguard.logging_utils import get_logger

log = get_logger(__name__)


def _require_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on the optional stack
        raise RuntimeError("Training needs the optional ML stack. Run: make install_ml") from exc
    return YOLO


def train(
    data_root: str | Path = "data/dfire",
    settings: Settings | None = None,
    epochs: int = 100,
    batch: int = 16,
    patience: int = 20,
    run_name: str = "pyraguard_dfire",
    workers: int = 8,
) -> dict:
    """Audit the dataset, fine tune from the configured base weights and install the best checkpoint."""
    settings = settings or load_settings()
    cfg = settings.vision.yolo
    root = settings.resolve(str(data_root))
    moved = make_validation_split(root)
    if moved:
        log.info("Created a validation split of %d images", moved)
    audit = audit_dataset(root, cfg.class_map, settings.resolve("reports/dfire_audit.json"))
    problems = sum(s["missing_label_files"] + s["malformed_rows"] + s["out_of_range_boxes"] for s in audit["splits"].values())
    log.info("Dataset audit: %d images, %d label problems", audit["total_images"], problems)
    data_yaml = write_data_yaml(root, settings.resolve("configs/dfire.yaml"), cfg.class_map)

    YOLO = _require_ultralytics()
    model = YOLO(cfg.base_weights)
    model.train(
        data=str(data_yaml), epochs=epochs, imgsz=cfg.image_size, batch=batch, patience=patience,
        device=cfg.device or None, workers=workers, project=str(settings.resolve("runs")), name=run_name, exist_ok=True,
        # fire has no canonical orientation or colour balance to protect, but hue shifts would turn
        # flame into something else, so hue augmentation stays small
        hsv_h=0.01, hsv_s=0.5, hsv_v=0.4, fliplr=0.5, mosaic=1.0, close_mosaic=10, cos_lr=True, seed=0,
    )
    best = settings.resolve("runs") / run_name / "weights" / "best.pt"
    target = settings.resolve(cfg.weights)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, target)
    log.info("Best checkpoint installed at %s", target)
    return {"weights": str(target), "audit": audit, "run_dir": str(best.parent.parent)}


def validate(settings: Settings | None = None, split: str = "test") -> dict:
    """Run Ultralytics validation on the held out split and write the metrics to reports."""
    settings = settings or load_settings()
    cfg = settings.vision.yolo
    YOLO = _require_ultralytics()
    model = YOLO(str(settings.resolve(cfg.weights)))
    metrics = model.val(data=str(settings.resolve("configs/dfire.yaml")), split=split, imgsz=cfg.image_size, device=cfg.device or None)
    names = model.names
    per_class = {names[int(c)]: {"ap50": round(float(metrics.box.ap50[i]), 4), "ap50_95": round(float(metrics.box.ap[i]), 4)} for i, c in enumerate(metrics.box.ap_class_index)}
    report = {
        "split": split, "weights": str(settings.resolve(cfg.weights)),
        "precision": round(float(metrics.box.mp), 4), "recall": round(float(metrics.box.mr), 4),
        "map50": round(float(metrics.box.map50), 4), "map50_95": round(float(metrics.box.map), 4), "per_class": per_class,
    }
    out = settings.resolve("reports/yolo_validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Validation report written to %s", out)
    return report


def export(settings: Settings | None = None, fmt: str = "onnx") -> str:
    """Export the trained detector for edge deployment (ONNX by default)."""
    settings = settings or load_settings()
    YOLO = _require_ultralytics()
    model = YOLO(str(settings.resolve(settings.vision.yolo.weights)))
    return str(model.export(format=fmt, imgsz=settings.vision.yolo.image_size))
