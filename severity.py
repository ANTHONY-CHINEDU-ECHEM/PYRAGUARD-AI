"""Hazard scoring.

The scorer converts detections and their temporal behaviour into a single
score from 0 to 100 and a named level. Each factor is normalised to the
range 0 to 1 first, so the weights in the configuration read as plain
importance shares.

Factors
    fire_extent   how much of the view is flame (log scaled, small flames still register)
    smoke_extent  how much of the view is smoke
    confidence    how sure the detectors are
    growth        how fast the tracked region is expanding
    persistence   how long the region has been continuously present
    thermal       peak temperature from the thermal channel, when one exists

Factors that cannot be measured (for example growth on a single still
image, or thermal on a site with no thermal camera) are left out and the
remaining weights are renormalised, so the score keeps the same meaning in
every deployment.
"""

from __future__ import annotations

import math

from pyraguard.config import HazardConfig
from pyraguard.schemas import FIRE, HOTSPOT, SMOKE, Detection, HazardAssessment, HazardLevel, TrackDynamics


def _log_scale(fraction: float, gain: float) -> float:
    return min(1.0, math.log10(1.0 + gain * max(fraction, 0.0)) / math.log10(1.0 + gain * 0.25))


class SeverityScorer:
    def __init__(self, config: HazardConfig | None = None) -> None:
        self.cfg = config or HazardConfig()

    def level_for(self, score: float) -> HazardLevel:
        t = self.cfg.thresholds
        if score >= t["critical"]:
            return HazardLevel.CRITICAL
        if score >= t["growing"]:
            return HazardLevel.GROWING
        if score >= t["incipient"]:
            return HazardLevel.INCIPIENT
        if score >= t["watch"]:
            return HazardLevel.WATCH
        return HazardLevel.CLEAR

    def assess(
        self,
        detections: list[Detection],
        dynamics: list[TrackDynamics] | None,
        frame_shape: tuple[int, int],
        zone_multiplier: float = 1.0,
    ) -> HazardAssessment:
        """Score one frame. Pass ``dynamics=None`` for a single still image."""
        if not detections:
            return HazardAssessment(HazardLevel.CLEAR, 0.0, {}, ["No hazard candidates in view."], [])

        h, w = frame_shape[:2]
        frame_area = float(h * w)
        fires = [d for d in detections if d.label == FIRE]
        smokes = [d for d in detections if d.label == SMOKE]
        hotspots = [d for d in detections if d.label == HOTSPOT]
        rationale: list[str] = []
        factors: dict[str, float] = {}

        fire_fraction = min(1.0, sum(d.bbox.area for d in fires) / frame_area)
        smoke_fraction = min(1.0, sum(d.bbox.area for d in smokes) / frame_area)
        factors["fire_extent"] = _log_scale(fire_fraction, 400.0) if fires else 0.0
        factors["smoke_extent"] = _log_scale(smoke_fraction, 60.0) if smokes else 0.0
        factors["confidence"] = max(
            [d.confidence for d in fires] + [0.7 * d.confidence for d in smokes] + [0.8 * d.confidence for d in hotspots]
        )
        if fires:
            rationale.append(f"Flame covers {fire_fraction * 100:.1f} percent of the view (confidence {max(d.confidence for d in fires):.2f}).")
        if smokes:
            rationale.append(f"Smoke covers {smoke_fraction * 100:.1f} percent of the view.")

        static_object = False
        if dynamics is not None:
            by_id = {t.track_id: t for t in dynamics}
            tracked = [by_id[d.track_id] for d in fires + smokes if d.track_id in by_id]
            growth_rate = max([t.growth_rate for t in tracked], default=0.0)
            age = max([t.age_seconds * t.persistence for t in tracked], default=0.0)
            factors["growth"] = min(1.0, max(growth_rate, 0.0) / self.cfg.growth_saturation_per_second)
            factors["persistence"] = min(1.0, age / self.cfg.persistence_saturation_seconds)
            if growth_rate > 0.02:
                rationale.append(f"Region is growing: area doubles in about {math.log(2) / growth_rate:.0f} seconds.")
            fire_tracks = [by_id[d.track_id] for d in fires if d.track_id in by_id]
            if fire_tracks and not smokes:
                # a fire coloured region that neither flickers nor grows is almost always an object
                mature = [t for t in fire_tracks if t.age_seconds >= 1.5]
                if mature and all(t.flicker < 0.08 and abs(t.growth_rate) < 0.01 for t in mature):
                    static_object = True
                    rationale.append("Fire coloured region is static (no flicker, no growth): treated as an object, kept under watch.")
                # a fire is anchored to its fuel; a region sliding sideways is a person, a vehicle or a reflection
                moving = [t for t in fire_tracks if t.age_seconds >= 0.5]
                if moving and all(t.drift_speed > 0.05 for t in moving):
                    static_object = True
                    rationale.append("Fire coloured region is travelling sideways: treated as a moving object, kept under watch.")

        if hotspots:
            peak = max(float(d.attributes.get("peak_celsius", 0.0)) for d in hotspots)
            factors["thermal"] = min(1.0, max(0.0, (peak - 60.0) / 240.0))
            rationale.append(f"Thermal hot spot at {peak:.0f} degrees Celsius.")

        if static_object:
            factors["fire_extent"] *= 0.25
            factors["confidence"] *= 0.4
            factors["persistence"] = factors.get("persistence", 0.0) * 0.25

        weights = {k: v for k, v in self.cfg.weights.items() if k in factors}
        total = sum(weights.values()) or 1.0
        score = 100.0 * sum(weights[k] * factors[k] for k in weights) / total
        score = min(100.0, score * max(zone_multiplier, 0.1))
        if zone_multiplier != 1.0:
            rationale.append(f"Zone risk multiplier {zone_multiplier:.2f} applied.")

        level = self.level_for(score)
        # guard rails that do not depend on the weighted score
        if not fires and not smokes and hotspots:
            # heat alone is a warning, not a fire: stay at watch unless the surface is far beyond its alarm point
            extreme = any(d.attributes.get("extreme") for d in hotspots)
            level = HazardLevel.INCIPIENT if extreme else HazardLevel.WATCH
            rationale.append(
                "No flame or smoke yet, but the surface is far above its alarm temperature: ignition may be imminent."
                if extreme
                else "No flame or smoke yet: overheating equipment, a pre ignition condition."
            )
        if fires and fire_fraction >= 0.20 and not static_object:
            level = max(level, HazardLevel.GROWING)
        if level == HazardLevel.CLEAR:
            level = HazardLevel.WATCH  # something was detected, so never report all clear
        labels = sorted({d.label for d in detections})
        return HazardAssessment(level, score, factors, rationale, labels)
