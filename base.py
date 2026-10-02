"""Detector interface. Every detector turns one BGR frame into a list of detections."""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from pyraguard.schemas import Detection


class Detector(ABC):
    name: str = "detector"

    @abstractmethod
    def detect(self, frame: np.ndarray, timestamp: float | None = None) -> list[Detection]:
        """Return hazard candidates for one BGR ``uint8`` frame."""

    def reset(self) -> None:  # noqa: B027 - optional hook
        """Forget any temporal state (called when a stream restarts)."""


def resize_to_width(frame: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    """Downscale to ``width`` keeping aspect ratio. Returns the frame and the scale applied."""
    h, w = frame.shape[:2]
    if width <= 0 or w <= width:
        return frame, 1.0
    scale = width / float(w)
    return cv2.resize(frame, (width, int(round(h * scale))), interpolation=cv2.INTER_AREA), scale


def nms(detections: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
    """Per label non maximum suppression."""
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if all(not (det.label == k.label and det.bbox.iou(k.bbox) > iou_threshold) for k in kept):
            kept.append(det)
    return kept
