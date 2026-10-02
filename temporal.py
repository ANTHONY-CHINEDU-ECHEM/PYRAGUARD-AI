"""Temporal behaviour of tracked hazard regions.

Early stage identification is a question about time, not about one frame.
A small bright region is a fire when it persists, flickers and grows; it is
a reflection when it does none of those. This module turns a track history
into those three measurements plus rise speed (useful for smoke).
"""

from __future__ import annotations

import numpy as np

from pyraguard.config import TrackingConfig
from pyraguard.schemas import TrackDynamics
from pyraguard.vision.tracker import IoUTracker, Track


def _slope(t: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Least squares slope of y on t and the residuals of the fit."""
    t0 = t - t.mean()
    denom = float((t0**2).sum())
    if denom <= 1e-9:
        return 0.0, np.zeros_like(y)
    slope = float((t0 * (y - y.mean())).sum() / denom)
    return slope, (y - y.mean()) - slope * t0


def track_dynamics(
    track: Track, frame_area: float, frame_height: float, frame_width: float, frames_in_window: int, cfg: TrackingConfig
) -> TrackDynamics:
    obs = list(track.history)
    times = np.array([o.timestamp for o in obs], dtype=np.float64)
    areas = np.array([max(o.area, 1.0) for o in obs], dtype=np.float64)
    tops = np.array([o.bbox.y1 for o in obs], dtype=np.float64)
    centres = np.array([o.bbox.center[0] for o in obs], dtype=np.float64)
    span = float(times[-1] - times[0]) if len(obs) > 1 else 0.0

    growth, flicker, rise, drift = 0.0, 0.0, 0.0, 0.0
    if len(obs) >= 4 and span >= 0.4:
        growth, residuals = _slope(times, np.log(areas))
        # flames breathe: the log area oscillates around its trend by several percent
        flicker = float(np.clip(residuals.std() / 0.12, 0.0, 1.0))
        top_slope, _ = _slope(times, tops)
        rise = float(-top_slope / max(frame_height, 1.0))
        centre_slope, _ = _slope(times, centres)
        drift = float(abs(centre_slope) / max(frame_width, 1.0))

    persistence = float(np.clip(len(obs) / max(frames_in_window, 1), 0.0, 1.0)) if frames_in_window else 0.0
    recent = areas[-3:]
    return TrackDynamics(
        track_id=track.track_id,
        label=track.label,
        age_seconds=float(obs[-1].timestamp - track.first_seen),
        hits=track.hits,
        persistence=persistence,
        area_fraction=float(np.median(recent) / max(frame_area, 1.0)),
        growth_rate=float(growth),
        flicker=flicker,
        rise_speed=rise,
        drift_speed=drift,
        mean_confidence=float(np.mean([o.confidence for o in obs[-8:]])),
    )


def analyse_tracks(tracker: IoUTracker, timestamp: float, frame_shape: tuple[int, int], cfg: TrackingConfig) -> list[TrackDynamics]:
    h, w = frame_shape[:2]
    frames = len(tracker.frame_times)
    return [track_dynamics(t, float(h * w), float(h), float(w), frames, cfg) for t in tracker.active(timestamp)]
