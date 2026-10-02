"""Drawing helpers for operator facing frames."""

from __future__ import annotations

import cv2
import numpy as np

from pyraguard.schemas import Detection, HazardAssessment, HazardLevel

LABEL_COLOURS = {"fire": (40, 60, 240), "smoke": (210, 150, 60), "hotspot": (200, 60, 200)}
LEVEL_COLOURS = {
    HazardLevel.CLEAR: (90, 160, 60),
    HazardLevel.WATCH: (60, 190, 220),
    HazardLevel.INCIPIENT: (30, 150, 245),
    HazardLevel.GROWING: (30, 90, 240),
    HazardLevel.CRITICAL: (40, 30, 200),
}


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        colour = LABEL_COLOURS.get(det.label, (255, 255, 255))
        x1, y1, x2, y2 = det.bbox.as_int()
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        tag = f"{det.label} {det.confidence:.2f}"
        if det.label == "hotspot" and "peak_celsius" in det.attributes:
            tag = f"hotspot {det.attributes['peak_celsius']:.0f}C"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y1 - 6 if y1 - th - 8 > 0 else y2 + th + 6
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 6, ty + 3), colour, -1)
        cv2.putText(out, tag, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_banner(frame: np.ndarray, assessment: HazardAssessment, confirmed: HazardLevel, camera_id: str = "") -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    colour = LEVEL_COLOURS[confirmed]
    cv2.rectangle(out, (0, 0), (w, 30), colour, -1)
    text = f"PyraGuard | {camera_id} | {confirmed.name} | score {assessment.score:.0f}"
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    bar = int((w - 16) * min(assessment.score, 100.0) / 100.0)
    cv2.rectangle(out, (8, h - 14), (w - 8, h - 6), (40, 40, 40), -1)
    cv2.rectangle(out, (8, h - 14), (8 + bar, h - 6), colour, -1)
    return out


def annotate(frame: np.ndarray, detections: list[Detection], assessment: HazardAssessment, confirmed: HazardLevel, camera_id: str = "") -> np.ndarray:
    return draw_banner(draw_detections(frame, detections), assessment, confirmed, camera_id)


def thermal_to_colour(temp_celsius: np.ndarray, low: float = 15.0, high: float = 150.0) -> np.ndarray:
    """Render a temperature map with the inferno palette for display."""
    norm = np.clip((temp_celsius - low) / max(high - low, 1e-6), 0.0, 1.0)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
