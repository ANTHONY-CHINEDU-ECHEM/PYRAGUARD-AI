"""Learned detector: an Ultralytics YOLO model fine tuned on fire and smoke.

The import is lazy so the rest of PyraGuard works on machines without torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyraguard.config import YoloConfig
from pyraguard.logging_utils import get_logger
from pyraguard.schemas import BBox, Detection
from pyraguard.vision.base import Detector

log = get_logger(__name__)


def ultralytics_available() -> bool:
    try:
        import ultralytics  # noqa: F401
    except Exception:
        return False
    return True


class YoloFireSmokeDetector(Detector):
    name = "yolo"

    def __init__(self, config: YoloConfig | None = None, weights: str | Path | None = None) -> None:
        self.cfg = config or YoloConfig()
        self.weights = Path(weights or self.cfg.weights)
        if not ultralytics_available():
            raise RuntimeError("ultralytics is not installed. Run: pip install ultralytics")
        if not self.weights.exists():
            raise FileNotFoundError(f"YOLO weights not found at {self.weights}. Train with: pyraguard train")
        from ultralytics import YOLO

        self.model = YOLO(str(self.weights))
        names = getattr(self.model, "names", None) or {}
        # trust the names stored in the checkpoint, fall back to the configured map
        self.class_map = {int(k): str(v).lower() for k, v in names.items()} or dict(self.cfg.class_map)
        log.info("Loaded YOLO weights %s with classes %s", self.weights, self.class_map)

    @staticmethod
    def is_ready(config: YoloConfig, root: Path | None = None) -> bool:
        weights = Path(config.weights)
        if root is not None and not weights.is_absolute():
            weights = root / weights
        return ultralytics_available() and weights.exists()

    def detect(self, frame: np.ndarray, timestamp: float | None = None) -> list[Detection]:
        results = self.model.predict(
            frame,
            conf=self.cfg.confidence,
            iou=self.cfg.iou,
            imgsz=self.cfg.image_size,
            device=self.cfg.device or None,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
                label = self.class_map.get(int(k), str(k))
                if label not in ("fire", "smoke"):
                    continue
                detections.append(Detection(label, float(c), BBox(float(x1), float(y1), float(x2), float(y2)), self.name))
        return detections
