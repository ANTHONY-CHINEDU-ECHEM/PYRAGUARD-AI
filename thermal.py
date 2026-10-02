"""Hot spot analysis for thermal (long wave infrared) frames.

Visible light cameras see flame and smoke. A radiometric thermal camera sees
the stage before that: a bearing, a cable joint or a battery pack running far
above ambient. PyraGuard treats those as ``hotspot`` detections so the hazard
score can rise before anything is burning.
"""

from __future__ import annotations

import cv2
import numpy as np

from pyraguard.config import ThermalConfig
from pyraguard.schemas import HOTSPOT, BBox, Detection


class ThermalAnalyzer:
    def __init__(self, config: ThermalConfig | None = None) -> None:
        self.cfg = config or ThermalConfig()

    def to_celsius(self, raw: np.ndarray) -> np.ndarray:
        """Convert a raw thermal frame to degrees Celsius.

        Float input is assumed to be Celsius already. Integer input (8 or 16
        bit) is mapped linearly onto the configured radiometric range.
        """
        if raw.ndim == 3:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        if np.issubdtype(raw.dtype, np.floating):
            return raw.astype(np.float32)
        full_scale = float(np.iinfo(raw.dtype).max)
        span = self.cfg.range_max_celsius - self.cfg.range_min_celsius
        return (raw.astype(np.float32) / full_scale) * span + self.cfg.range_min_celsius

    def analyze(self, thermal: np.ndarray) -> list[Detection]:
        temp = self.to_celsius(thermal)
        ambient = float(np.median(temp))
        threshold = min(self.cfg.alarm_celsius, ambient + self.cfg.ambient_margin_celsius)
        hot = (temp >= threshold).astype(np.uint8)
        detections: list[Detection] = []
        count, labels, stats, _ = cv2.connectedComponentsWithStats(hot, connectivity=8)
        for idx in range(1, count):
            x, y, bw, bh, area = stats[idx]
            if area < self.cfg.min_area_pixels:
                continue
            region = temp[labels == idx]
            peak = float(region.max())
            excess = peak - threshold
            confidence = float(0.35 + 0.65 * np.clip(excess / 150.0, 0.0, 1.0))
            attrs = {
                "peak_celsius": peak,
                "mean_celsius": float(region.mean()),
                "ambient_celsius": ambient,
                "above_alarm": float(peak >= self.cfg.alarm_celsius),
                "extreme": float(peak >= 2.0 * self.cfg.alarm_celsius),
            }
            detections.append(Detection(HOTSPOT, confidence, BBox(float(x), float(y), float(x + bw), float(y + bh)), "thermal", attrs))
        return detections

    @staticmethod
    def rescale(detections: list[Detection], thermal_shape: tuple[int, int], frame_shape: tuple[int, int]) -> list[Detection]:
        """Map thermal boxes into visible frame coordinates (assumes aligned fields of view)."""
        th, tw = thermal_shape[:2]
        fh, fw = frame_shape[:2]
        sx, sy = fw / float(tw), fh / float(th)
        out = []
        for d in detections:
            b = d.bbox
            out.append(Detection(d.label, d.confidence, BBox(b.x1 * sx, b.y1 * sy, b.x2 * sx, b.y2 * sy), d.source, d.attributes))
        return out
