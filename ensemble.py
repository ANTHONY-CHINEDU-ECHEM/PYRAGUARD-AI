"""Fuse the learned detector with the classical one and build detectors from settings."""

from __future__ import annotations

import numpy as np

from pyraguard.config import PROJECT_ROOT, Settings
from pyraguard.logging_utils import get_logger
from pyraguard.schemas import Detection
from pyraguard.vision.base import Detector, nms
from pyraguard.vision.heuristic import HeuristicFireSmokeDetector

log = get_logger(__name__)
_ANNOUNCED = False


class EnsembleDetector(Detector):
    """Agreement raises confidence, disagreement keeps the candidate at a discount.

    When both detectors report the same label over the same area their
    confidences are fused with a noisy OR, ``1 - (1 - a)(1 - b)``. A box seen
    by the secondary detector alone is retained at a discount, because
    missing an early flame costs far more than one extra candidate that the
    temporal confirmation stage can still veto.
    """

    name = "ensemble"

    def __init__(self, primary: Detector, secondary: Detector, match_iou: float = 0.3, solo_discount: float = 0.8) -> None:
        self.primary, self.secondary = primary, secondary
        self.match_iou, self.solo_discount = match_iou, solo_discount

    def reset(self) -> None:
        self.primary.reset()
        self.secondary.reset()

    def detect(self, frame: np.ndarray, timestamp: float | None = None) -> list[Detection]:
        first = self.primary.detect(frame, timestamp)
        second = self.secondary.detect(frame, timestamp)
        fused: list[Detection] = []
        used: set[int] = set()
        for det in first:
            partner_idx, best = None, self.match_iou
            for j, other in enumerate(second):
                if j in used or other.label != det.label:
                    continue
                overlap = det.bbox.iou(other.bbox)
                if overlap >= best:
                    partner_idx, best = j, overlap
            if partner_idx is None:
                fused.append(det)
                continue
            used.add(partner_idx)
            other = second[partner_idx]
            confidence = 1.0 - (1.0 - det.confidence) * (1.0 - other.confidence)
            attrs = {**other.attributes, **det.attributes, "agreement_iou": best}
            fused.append(Detection(det.label, confidence, det.bbox, self.name, attrs))
        for j, other in enumerate(second):
            if j not in used:
                fused.append(Detection(other.label, other.confidence * self.solo_discount, other.bbox, other.source, other.attributes))
        return nms(fused, 0.5)


def build_detector(settings: Settings) -> Detector:
    """Create the detector named in the settings.

    ``auto`` prefers the ensemble when trained YOLO weights are present and
    otherwise falls back to the classical detector.
    """
    from pyraguard.vision.yolo import YoloFireSmokeDetector

    choice = settings.vision.detector.lower()
    heuristic = HeuristicFireSmokeDetector(settings.vision.heuristic)
    yolo_ready = YoloFireSmokeDetector.is_ready(settings.vision.yolo, PROJECT_ROOT)
    if choice == "heuristic" or (choice == "auto" and not yolo_ready):
        global _ANNOUNCED
        if choice == "auto" and not _ANNOUNCED:
            log.info("No trained YOLO weights found, using the classical detector.")
            _ANNOUNCED = True
        return heuristic
    weights = settings.resolve(settings.vision.yolo.weights)
    yolo = YoloFireSmokeDetector(settings.vision.yolo, weights)
    if choice == "yolo":
        return yolo
    return EnsembleDetector(yolo, heuristic)
