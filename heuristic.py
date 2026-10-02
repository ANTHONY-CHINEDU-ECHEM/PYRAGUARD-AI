"""Classical fire and smoke detector.

This detector needs no weights and no GPU. It has three jobs in PyraGuard:

* a baseline that the learned detector has to beat,
* an independent second opinion inside the ensemble (colour physics does not
  share failure modes with a neural network),
* a fallback so the full pipeline, the tests and the demo run anywhere.

Fire candidates come from chromatic rules in RGB and YCbCr space (in the
spirit of Chen et al. 2004 and Celik and Demirel 2009). Each candidate region
is then scored on properties that separate combustion from fire coloured
objects: a hot core surrounded by cooler flame, an irregular outline, and,
when previous frames exist, flicker.

Smoke candidates need a background model, so they are only produced in
stream mode: smoke is whatever turned a patch of the scene greyer, flatter
and less saturated than it was.
"""

from __future__ import annotations

import cv2
import numpy as np

from pyraguard.config import HeuristicConfig
from pyraguard.schemas import FIRE, SMOKE, BBox, Detection
from pyraguard.vision.base import Detector


class HeuristicFireSmokeDetector(Detector):
    name = "heuristic"

    def __init__(self, config: HeuristicConfig | None = None) -> None:
        self.cfg = config or HeuristicConfig()
        self._background: np.ndarray | None = None
        self._prev_gray: np.ndarray | None = None
        self._still: np.ndarray | None = None
        self._frames_seen = 0

    def reset(self) -> None:
        self._background = None
        self._prev_gray = None
        self._still = None
        self._frames_seen = 0

    # ------------------------------------------------------------------ fire
    def fire_mask(self, frame: np.ndarray) -> np.ndarray:
        """Boolean mask of pixels whose colour is consistent with flame."""
        cfg = self.cfg
        b = frame[..., 0].astype(np.int16)
        g = frame[..., 1].astype(np.int16)
        r = frame[..., 2].astype(np.int16)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        s = hsv[..., 1].astype(np.int16)
        y, cr, cb = (ycc[..., i].astype(np.int16) for i in range(3))

        rgb_rule = (r >= cfg.red_min) & (r >= g) & (g >= b)
        # saturation must rise as red falls, so dim red objects are rejected
        sat_rule = s >= ((255 - r) * cfg.saturation_min) // max(cfg.red_min, 1)
        ycc_rule = (y > cb) & (cr >= cb + cfg.cbcr_gap_min)
        mask = (rgb_rule & sat_rule & ycc_rule).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask.astype(bool)

    def _score_fire_region(
        self,
        frame: np.ndarray,
        component: np.ndarray,
        contour: np.ndarray,
        motion: np.ndarray | None,
    ) -> tuple[float, dict[str, float]]:
        g = frame[..., 1][component].astype(np.float32)
        r = frame[..., 2][component].astype(np.float32)
        area = float(component.sum())

        # 1. thermal gradient: green climbs from about 50 at the flame edge to about 250 in the core
        green_spread = float(np.clip(g.std() / 60.0, 0.0, 1.0))
        # 2. a hot core exists but does not fill the whole region
        core_ratio = float(((g > 190) & (r > 235)).mean())
        core_score = float(np.clip(1.0 - abs(core_ratio - 0.45) / 0.55, 0.0, 1.0)) if core_ratio > 0.02 else 0.0
        # 3. flames are ragged; signage, boxes and clothing are close to rectangles
        (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
        rectangularity = area / max(rw * rh, 1.0)
        shape_score = float(np.clip((1.0 - rectangularity) / 0.35, 0.0, 1.0))

        static_score = 0.50 * green_spread + 0.20 * core_score + 0.30 * shape_score
        features = {
            "green_spread": green_spread,
            "core_ratio": core_ratio,
            "rectangularity": float(min(rectangularity, 1.0)),
            "static_score": static_score,
        }
        if motion is None:
            return static_score, features
        flicker = float(np.clip(motion[component].mean() / 18.0, 0.0, 1.0))
        features["flicker"] = flicker
        # a perfectly still fire coloured region is discounted, never zeroed
        return static_score * (0.62 + 0.38 * flicker), features

    def _detect_fire(self, frame: np.ndarray, motion: np.ndarray | None) -> tuple[list[Detection], np.ndarray]:
        mask = self.fire_mask(frame)
        h, w = mask.shape
        min_area = max(12.0, self.cfg.min_area_fraction * h * w)
        detections: list[Detection] = []
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        for idx in range(1, count):
            x, y, bw, bh, area = stats[idx]
            if area < min_area:
                continue
            component = labels == idx
            contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            confidence, features = self._score_fire_region(frame, component, contour, motion)
            if confidence < self.cfg.min_confidence:
                continue
            features["mask_area"] = float(area)
            detections.append(
                Detection(FIRE, float(min(confidence, 0.99)), BBox(float(x), float(y), float(x + bw), float(y + bh)), self.name, features)
            )
        return detections, mask

    # ----------------------------------------------------------------- smoke
    def _detect_smoke(self, frame: np.ndarray, gray: np.ndarray, fire_mask: np.ndarray, activity: np.ndarray | None) -> list[Detection]:
        cfg = self.cfg
        if not cfg.smoke_enabled or self._background is None or activity is None or self._frames_seen < cfg.warmup_frames:
            return []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        s, v = hsv[..., 1], hsv[..., 2]
        diff = cv2.absdiff(gray.astype(np.float32), self._background)
        diff = cv2.GaussianBlur(diff, (0, 0), 3.0)
        candidate = (diff > cfg.smoke_diff_min) & (s < cfg.smoke_saturation_max) & (v > cfg.smoke_value_min)
        candidate &= ~cv2.dilate(fire_mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        mask = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

        h, w = mask.shape
        min_area = cfg.smoke_min_area_fraction * h * w
        # smoke hides detail: compare present texture with the texture of the remembered background
        texture_now = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)), (0, 0), 4.0)
        texture_bg = cv2.GaussianBlur(np.abs(cv2.Laplacian(self._background, cv2.CV_32F)), (0, 0), 4.0)

        detections: list[Detection] = []
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for idx in range(1, count):
            x, y, bw, bh, area = stats[idx]
            if area < min_area:
                continue
            component = labels == idx
            # smoke churns; a ghost left behind by a moved object is perfectly still
            churn = float(activity[component].mean())
            if churn < cfg.smoke_activity_ratio:
                continue
            strength = float(np.clip(diff[component].mean() / 45.0, 0.0, 1.0))
            fill = float(area) / float(bw * bh)
            softness = float(np.clip(1.0 - fill, 0.0, 0.6) / 0.6)  # plumes are blobby, not box shaped
            veil = float(np.clip((texture_bg[component].mean() - texture_now[component].mean()) / 3.0 + 0.5, 0.0, 1.0))
            confidence = min(0.85, 0.28 + 0.27 * strength + 0.20 * softness + 0.20 * veil)
            if confidence < cfg.min_confidence:
                continue
            attrs = {"strength": strength, "softness": softness, "veil": veil, "churn": churn, "mask_area": float(area)}
            detections.append(Detection(SMOKE, confidence, BBox(float(x), float(y), float(x + bw), float(y + bh)), self.name, attrs))
        return detections

    # ------------------------------------------------------------------ main
    def detect(self, frame: np.ndarray, timestamp: float | None = None) -> list[Detection]:
        if frame is None or frame.ndim != 3:
            raise ValueError("expected a BGR colour frame")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = None
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            motion = cv2.absdiff(gray, self._prev_gray).astype(np.float32)

        # motion relative to the sensor noise floor, so thresholds transfer between cameras
        activity = None
        if motion is not None:
            smoothed = cv2.GaussianBlur(motion, (0, 0), 3.0)
            activity = smoothed / max(float(np.median(smoothed)), 0.5)

        fire, fire_mask = self._detect_fire(frame, motion)
        smoke = self._detect_smoke(frame, gray, fire_mask, activity)

        self._update_background(gray, fire_mask, activity)
        self._prev_gray = gray
        self._frames_seen += 1
        return fire + smoke

    def _update_background(self, gray: np.ndarray, fire_mask: np.ndarray, activity: np.ndarray | None) -> None:
        current = gray.astype(np.float32)
        if self._background is None or self._background.shape != current.shape:
            self._background = current.copy()
            self._still = np.zeros(current.shape, np.uint16)
            return
        rate = np.full(current.shape, self.cfg.background_learning_rate, np.float32)
        changed = np.abs(current - self._background) > self.cfg.smoke_diff_min
        if self._frames_seen >= self.cfg.warmup_frames:
            rate[changed] *= 0.05  # moving foreground is absorbed very slowly
        if activity is not None and self._still is not None:
            # changed but motionless for several frames: a ghost or a moved object, absorb it now
            quiet = changed & (activity < self.cfg.ghost_quiet_ratio)
            self._still = np.where(quiet, self._still + 1, 0).astype(np.uint16)
            rate[self._still >= self.cfg.ghost_absorb_frames] = 1.0
        rate[fire_mask] = 0.0
        self._background += rate * (current - self._background)
