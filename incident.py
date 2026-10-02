"""Incident lifecycle with temporal confirmation.

A single hot frame must never page a fire warden, and a real fire must not
flap between levels as flames flicker. The manager therefore applies:

* time based confirmation: a level is confirmed when at least
  ``confirm_ratio`` of the frames in the last ``confirm_seconds`` reached
  it, which makes the behaviour independent of the camera frame rate,
* a ratchet: an open incident only escalates (stepping down is a human
  decision in an emergency),
* a quiet period: the incident closes after ``clear_after_seconds`` with no
  confirmed hazard.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field

from pyraguard.config import HazardConfig
from pyraguard.schemas import HazardAssessment, HazardLevel, IncidentEvent


@dataclass
class Incident:
    incident_id: str
    camera_id: str
    zone_id: str | None
    opened_at: float
    level: HazardLevel
    peak_score: float
    last_hazard_at: float
    closed_at: float | None = None
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
            "opened_at": round(self.opened_at, 3),
            "closed_at": None if self.closed_at is None else round(self.closed_at, 3),
            "level": self.level.name,
            "peak_score": round(self.peak_score, 1),
            "history": self.history,
        }


class IncidentManager:
    def __init__(self, config: HazardConfig | None = None) -> None:
        self.cfg = config or HazardConfig()
        self._recent: dict[str, deque[tuple[float, HazardLevel]]] = {}
        self._first_seen: dict[str, float] = {}
        self.open: dict[str, Incident] = {}
        self.closed: list[Incident] = []

    def reset(self) -> None:
        self._recent.clear()
        self._first_seen.clear()
        self.open.clear()
        self.closed.clear()

    def confirmed_level(self, camera_id: str) -> HazardLevel:
        recent = self._recent.get(camera_id)
        if not recent:
            return HazardLevel.CLEAR
        newest = recent[-1][0]
        # the camera must have been observed for most of a full window before anything is confirmed
        if newest - self._first_seen.get(camera_id, newest) < 0.75 * self.cfg.confirm_seconds:
            return HazardLevel.CLEAR
        needed = max(2.0, self.cfg.confirm_ratio * len(recent))
        for level in sorted(HazardLevel, reverse=True):
            if level == HazardLevel.CLEAR:
                break
            if sum(1 for _, lv in recent if lv >= level) >= needed:
                return level
        return HazardLevel.CLEAR

    def update(self, camera_id: str, zone_id: str | None, assessment: HazardAssessment, timestamp: float) -> tuple[HazardLevel, IncidentEvent | None]:
        recent = self._recent.setdefault(camera_id, deque())
        self._first_seen.setdefault(camera_id, timestamp)
        recent.append((timestamp, assessment.level))
        while recent and recent[0][0] < timestamp - self.cfg.confirm_seconds:
            recent.popleft()
        confirmed = self.confirmed_level(camera_id)
        incident = self.open.get(camera_id)
        event: IncidentEvent | None = None

        if incident is None:
            if confirmed >= HazardLevel.INCIPIENT:
                incident = Incident(f"INC{uuid.uuid4().hex[:8].upper()}", camera_id, zone_id, timestamp, confirmed, assessment.score, timestamp)
                self.open[camera_id] = incident
                event = IncidentEvent("opened", incident.incident_id, camera_id, zone_id, confirmed, HazardLevel.CLEAR, timestamp, assessment.score)
        else:
            incident.peak_score = max(incident.peak_score, assessment.score)
            if confirmed >= HazardLevel.INCIPIENT:
                incident.last_hazard_at = timestamp
            if confirmed > incident.level:
                previous, incident.level = incident.level, confirmed
                event = IncidentEvent("escalated", incident.incident_id, camera_id, zone_id, confirmed, previous, timestamp, assessment.score)
            elif timestamp - incident.last_hazard_at >= self.cfg.clear_after_seconds:
                incident.closed_at = timestamp
                self.closed.append(self.open.pop(camera_id))
                event = IncidentEvent("closed", incident.incident_id, camera_id, zone_id, HazardLevel.CLEAR, incident.level, timestamp, assessment.score)
        if event is not None and incident is not None:
            incident.history.append({"t": round(timestamp, 3), "kind": event.kind, "level": event.level.name, "score": round(assessment.score, 1)})
        # while an incident is open the operator sees its ratcheted level
        current = self.open.get(camera_id)
        return (max(confirmed, current.level) if current else confirmed), event

    def all_incidents(self) -> list[Incident]:
        return list(self.open.values()) + self.closed
