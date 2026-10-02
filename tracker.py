"""Lightweight IoU tracker.

Fire regions change shape every frame, so appearance based trackers add
little. Greedy IoU association with a centre distance fallback is enough to
keep one identity per hazard region, which is all the temporal analysis
needs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from pyraguard.config import TrackingConfig
from pyraguard.schemas import BBox, Detection


@dataclass
class Observation:
    timestamp: float
    bbox: BBox
    confidence: float
    area: float  # mask area when the detector reports it, box area otherwise


@dataclass
class Track:
    track_id: int
    label: str
    first_seen: float
    history: deque[Observation] = field(default_factory=deque)
    hits: int = 0

    @property
    def last(self) -> Observation:
        return self.history[-1]


class IoUTracker:
    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.cfg = config or TrackingConfig()
        self.tracks: dict[int, Track] = {}
        self.frame_times: deque[float] = deque()
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self.frame_times.clear()
        self._next_id = 1

    def _affinity(self, track: Track, det: Detection) -> float:
        box = track.last.bbox
        overlap = box.iou(det.bbox)
        if overlap >= self.cfg.iou_match:
            return overlap
        (ax, ay), (bx, by) = box.center, det.bbox.center
        reach = max(box.width, box.height, det.bbox.width, det.bbox.height)
        distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        # small flickering regions often miss on IoU but stay within one box size of themselves
        return self.cfg.iou_match * 0.5 if reach > 0 and distance <= reach else 0.0

    def update(self, detections: list[Detection], timestamp: float) -> list[Detection]:
        """Assign ``track_id`` to each detection and age out stale tracks."""
        self.frame_times.append(timestamp)
        horizon = timestamp - self.cfg.window_seconds
        while self.frame_times and self.frame_times[0] < horizon:
            self.frame_times.popleft()

        pairs = []
        for tid, track in self.tracks.items():
            for j, det in enumerate(detections):
                if det.label != track.label:
                    continue
                score = self._affinity(track, det)
                if score > 0:
                    pairs.append((score, tid, j))
        pairs.sort(reverse=True)
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _, tid, j in pairs:
            if tid in used_tracks or j in used_dets:
                continue
            used_tracks.add(tid)
            used_dets.add(j)
            self._append(self.tracks[tid], detections[j], timestamp)
        for j, det in enumerate(detections):
            if j in used_dets:
                continue
            track = Track(self._next_id, det.label, timestamp)
            self._next_id += 1
            self.tracks[track.track_id] = track
            self._append(track, det, timestamp)

        stale = [tid for tid, t in self.tracks.items() if timestamp - t.last.timestamp > self.cfg.max_age_seconds]
        for tid in stale:
            del self.tracks[tid]
        for track in self.tracks.values():
            while track.history and track.history[0].timestamp < horizon:
                track.history.popleft()
        return detections

    def _append(self, track: Track, det: Detection, timestamp: float) -> None:
        area = float(det.attributes.get("mask_area", det.bbox.area))
        track.history.append(Observation(timestamp, det.bbox, det.confidence, area))
        track.hits += 1
        det.track_id = track.track_id

    def active(self, timestamp: float) -> list[Track]:
        """Tracks observed in the most recent frame."""
        return [t for t in self.tracks.values() if t.history and abs(t.last.timestamp - timestamp) < 1e-6]
