"""Domain objects shared by every layer of PyraGuard.

Plain dataclasses are used on purpose: they are cheap to create inside the
per frame loop and serialise cleanly to JSON for the API, the alert payloads
and the incident log.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any

FIRE = "fire"
SMOKE = "smoke"
HOTSPOT = "hotspot"
HAZARD_LABELS = (FIRE, SMOKE, HOTSPOT)


@dataclass(frozen=True)
class BBox:
    """Axis aligned box in pixel coordinates (x1, y1) top left, (x2, y2) bottom right."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def iou(self, other: BBox) -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def clip(self, width: int, height: int) -> BBox:
        return BBox(
            min(max(self.x1, 0.0), width),
            min(max(self.y1, 0.0), height),
            min(max(self.x2, 0.0), width),
            min(max(self.y2, 0.0), height),
        )

    def as_int(self) -> tuple[int, int, int, int]:
        return int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))

    def as_list(self) -> list[float]:
        return [round(self.x1, 1), round(self.y1, 1), round(self.x2, 1), round(self.y2, 1)]

    @staticmethod
    def from_yolo(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> BBox:
        """Build a box from normalised YOLO centre format."""
        return BBox((cx - w / 2) * img_w, (cy - h / 2) * img_h, (cx + w / 2) * img_w, (cy + h / 2) * img_h)

    def to_yolo(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        cx, cy = self.center
        return cx / img_w, cy / img_h, self.width / img_w, self.height / img_h


@dataclass
class Detection:
    """A single hazard candidate reported by a detector for one frame."""

    label: str
    confidence: float
    bbox: BBox
    source: str = "unknown"
    attributes: dict[str, Any] = field(default_factory=dict)
    track_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "bbox": self.bbox.as_list(),
            "source": self.source,
            "track_id": self.track_id,
            "attributes": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.attributes.items()},
        }


class HazardLevel(IntEnum):
    """Ordered hazard scale. Levels map onto the escalation matrix in the knowledge base."""

    CLEAR = 0
    WATCH = 1
    INCIPIENT = 2
    GROWING = 3
    CRITICAL = 4

    @property
    def title(self) -> str:
        return self.name.capitalize()


@dataclass
class TrackDynamics:
    """Temporal behaviour of one tracked hazard region."""

    track_id: int
    label: str
    age_seconds: float
    hits: int
    persistence: float
    area_fraction: float
    growth_rate: float  # relative area change per second (0.07 doubles in about ten seconds)
    flicker: float  # 0 to 1, how much the region shimmers frame to frame
    rise_speed: float  # fraction of frame height per second, positive is upward
    drift_speed: float  # sideways speed of the region centre, fraction of frame width per second
    mean_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


@dataclass
class HazardAssessment:
    level: HazardLevel
    score: float  # 0 to 100
    factors: dict[str, float] = field(default_factory=dict)
    rationale: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.name,
            "score": round(self.score, 1),
            "factors": {k: round(v, 4) for k, v in self.factors.items()},
            "rationale": self.rationale,
            "labels": self.labels,
        }


@dataclass
class IncidentEvent:
    """Emitted by the incident manager whenever an incident changes state."""

    kind: str  # opened | escalated | deescalated | closed
    incident_id: str
    camera_id: str
    zone_id: str | None
    level: HazardLevel
    previous_level: HazardLevel
    timestamp: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["level"] = self.level.name
        d["previous_level"] = self.previous_level.name
        return d


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "section": self.section,
            "score": round(float(self.score), 4),
            "source": self.metadata.get("source", ""),
            "text": self.text,
        }


@dataclass
class PlanItem:
    text: str
    citations: list[str]
    support: float = 0.0  # lexical support of the statement by its cited chunks, 0 to 1
    kind: str = "action"  # action | prohibition

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "citations": self.citations, "support": round(self.support, 3), "kind": self.kind}


@dataclass
class EvacuationRoute:
    origin: str
    path: list[str]
    exit_zone: str
    assembly_point: str
    distance_m: float
    eta_seconds: float
    instructions: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["distance_m"] = round(self.distance_m, 1)
        d["eta_seconds"] = round(self.eta_seconds, 1)
        return d


@dataclass
class ResponsePlan:
    """Grounded emergency response advice for one incident state."""

    summary: str
    actions: list[PlanItem]
    prohibitions: list[PlanItem]
    sources: list[RetrievedChunk]
    groundedness: float
    citation_coverage: float
    provider: str
    query: str
    routes: list[EvacuationRoute] = field(default_factory=list)
    blocked_zones: list[str] = field(default_factory=list)
    trapped_zones: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "actions": [a.to_dict() for a in self.actions],
            "prohibitions": [p.to_dict() for p in self.prohibitions],
            "sources": [s.to_dict() for s in self.sources],
            "groundedness": round(self.groundedness, 3),
            "citation_coverage": round(self.citation_coverage, 3),
            "provider": self.provider,
            "query": self.query,
            "routes": [r.to_dict() for r in self.routes],
            "blocked_zones": self.blocked_zones,
            "trapped_zones": self.trapped_zones,
            "warnings": self.warnings,
        }


@dataclass
class FrameResult:
    """Everything PyraGuard concluded about one frame."""

    camera_id: str
    timestamp: float
    frame_index: int
    detections: list[Detection]
    dynamics: list[TrackDynamics]
    assessment: HazardAssessment
    confirmed_level: HazardLevel
    event: IncidentEvent | None = None
    plan: ResponsePlan | None = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "timestamp": round(self.timestamp, 3),
            "frame_index": self.frame_index,
            "detections": [d.to_dict() for d in self.detections],
            "dynamics": [t.to_dict() for t in self.dynamics],
            "assessment": self.assessment.to_dict(),
            "confirmed_level": self.confirmed_level.name,
            "event": self.event.to_dict() if self.event else None,
            "plan": self.plan.to_dict() if self.plan else None,
            "latency_ms": round(self.latency_ms, 2),
        }
