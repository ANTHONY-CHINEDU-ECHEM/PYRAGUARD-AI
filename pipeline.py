"""The PyraGuard engine: perception, assessment, knowledge and action in one loop.

    frame -> detect -> track -> temporal analysis -> severity -> incident state
          -> (on state change) scene context -> retrieval -> grounded plan
          -> evacuation routes -> alerts
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable

import numpy as np

from pyraguard.alerts.dispatcher import AlertDispatcher
from pyraguard.config import Settings, load_settings
from pyraguard.evacuation.router import EvacuationRouter
from pyraguard.evacuation.site import Site
from pyraguard.hazard.incident import IncidentManager
from pyraguard.hazard.severity import SeverityScorer
from pyraguard.logging_utils import get_logger
from pyraguard.rag.scene import SceneContext
from pyraguard.rag.service import RagService
from pyraguard.schemas import BBox, Detection, FrameResult, HazardAssessment, HazardLevel, ResponsePlan
from pyraguard.vision.annotate import annotate
from pyraguard.vision.base import Detector, resize_to_width
from pyraguard.vision.ensemble import build_detector
from pyraguard.vision.temporal import analyse_tracks
from pyraguard.vision.thermal import ThermalAnalyzer
from pyraguard.vision.tracker import IoUTracker

log = get_logger(__name__)


class PyraGuardEngine:
    def __init__(
        self,
        settings: Settings | None = None,
        detector_factory: Callable[[], Detector] | None = None,
        rag: RagService | None = None,
        dispatcher: AlertDispatcher | None = None,
        site: Site | None = None,
        enable_rag: bool = True,
        enable_alerts: bool = True,
    ) -> None:
        self.settings = settings or load_settings()
        self._detector_factory = detector_factory or (lambda: build_detector(self.settings))
        self._detectors: dict[str, Detector] = {}
        self._trackers: dict[str, IoUTracker] = {}
        self._frame_index: dict[str, int] = {}
        self.thermal = ThermalAnalyzer(self.settings.vision.thermal)
        self.scorer = SeverityScorer(self.settings.hazard)
        self.incidents = IncidentManager(self.settings.hazard)
        site_path = self.settings.resolve(self.settings.evacuation.site_file)
        self.site = site or (Site.from_yaml(site_path) if site_path.exists() else None)
        self.router = EvacuationRouter(self.site, self.settings.evacuation) if self.site else None
        self.rag = rag if rag is not None else (RagService(self.settings) if enable_rag else None)
        self.dispatcher = dispatcher if dispatcher is not None else (AlertDispatcher(self.settings.alerts) if enable_alerts else None)
        self.levels: dict[str, HazardLevel] = {}
        self.last_plans: dict[str, ResponsePlan] = {}

    # ------------------------------------------------------------- utilities
    def reset(self) -> None:
        self._detectors.clear()
        self._trackers.clear()
        self._frame_index.clear()
        self.incidents.reset()
        self.levels.clear()
        self.last_plans.clear()

    def _zone(self, camera_id: str):
        return self.site.zone_for_camera(camera_id) if self.site else None

    def zone_hazards(self) -> dict[str, HazardLevel]:
        """Current confirmed level per zone, taken from the cameras that watch it."""
        hazards: dict[str, HazardLevel] = {}
        for camera_id, level in self.levels.items():
            zone = self._zone(camera_id)
            if zone is not None and level > HazardLevel.CLEAR:
                hazards[zone.zone_id] = max(level, hazards.get(zone.zone_id, HazardLevel.CLEAR))
        return hazards

    def scene_for(self, camera_id: str, assessment: HazardAssessment, level: HazardLevel) -> SceneContext:
        zone = self._zone(camera_id)
        scene = SceneContext(camera_id=camera_id, labels=list(assessment.labels), level=level, score=assessment.score, rationale=list(assessment.rationale))
        if zone is not None:
            scene.zone_id, scene.zone_name, scene.zone_type = zone.zone_id, zone.name, zone.type
            scene.materials, scene.occupancy = list(zone.materials), zone.occupancy or None
            scene.assisted_occupants = any(z.assisted for z in self.site.zones.values()) if self.site else False
        return scene

    def build_plan(self, camera_id: str, assessment: HazardAssessment, level: HazardLevel, frame: np.ndarray | None) -> ResponsePlan | None:
        if self.rag is None:
            return None
        plan = self.rag.advise(self.scene_for(camera_id, assessment, level), frame)
        if self.router is not None and level >= HazardLevel.INCIPIENT:
            zone = self._zone(camera_id)
            hazards = self.zone_hazards()
            if zone is not None:
                hazards[zone.zone_id] = max(level, hazards.get(zone.zone_id, HazardLevel.CLEAR))
            evacuation = self.router.plan(hazards)
            plan.routes, plan.blocked_zones, plan.trapped_zones = evacuation.routes, evacuation.blocked_zones, evacuation.trapped_zones
            plan.warnings.extend(evacuation.notes)
        self.last_plans[camera_id] = plan
        return plan

    def _detect(self, camera_id: str, frame: np.ndarray, timestamp: float, thermal: np.ndarray | None, detector: Detector) -> tuple[list[Detection], np.ndarray, float]:
        small, scale = resize_to_width(frame, self.settings.vision.process_width)
        detections = detector.detect(small, timestamp)
        if thermal is not None and self.settings.vision.thermal.enabled:
            hot = self.thermal.analyze(thermal)
            detections.extend(self.thermal.rescale(hot, thermal.shape[:2], small.shape[:2]))
        return detections, small, scale

    @staticmethod
    def _to_original(detections: list[Detection], scale: float) -> None:
        if scale == 1.0:
            return
        for det in detections:
            b = det.bbox
            det.bbox = BBox(b.x1 / scale, b.y1 / scale, b.x2 / scale, b.y2 / scale)

    # ---------------------------------------------------------------- stream
    def process_frame(self, frame: np.ndarray, camera_id: str = "camera", timestamp: float | None = None, thermal: np.ndarray | None = None) -> FrameResult:
        """Process one frame of a continuous stream."""
        started = time.perf_counter()
        timestamp = time.time() if timestamp is None else float(timestamp)
        if camera_id not in self._detectors:
            self._detectors[camera_id] = self._detector_factory()
            self._trackers[camera_id] = IoUTracker(self.settings.tracking)
        detector, tracker = self._detectors[camera_id], self._trackers[camera_id]
        index = self._frame_index.get(camera_id, 0)
        self._frame_index[camera_id] = index + 1

        detections, small, scale = self._detect(camera_id, frame, timestamp, thermal, detector)
        tracker.update(detections, timestamp)
        dynamics = analyse_tracks(tracker, timestamp, small.shape, self.settings.tracking)
        zone = self._zone(camera_id)
        assessment = self.scorer.assess(detections, dynamics, small.shape, zone.risk_multiplier if zone else 1.0)
        confirmed, event = self.incidents.update(camera_id, zone.zone_id if zone else None, assessment, timestamp)
        self.levels[camera_id] = confirmed
        self._to_original(detections, scale)

        result = FrameResult(camera_id, timestamp, index, detections, dynamics, assessment, confirmed, event)
        if event is not None and event.kind in ("opened", "escalated"):
            result.plan = self.build_plan(camera_id, assessment, confirmed, frame)
        if event is not None and self.dispatcher is not None:
            self.dispatcher.dispatch(event, result)
        result.latency_ms = (time.perf_counter() - started) * 1000.0
        return result

    def run(self, frames: Iterable[tuple[np.ndarray, float]], camera_id: str = "camera", max_frames: int | None = None, on_result: Callable[[np.ndarray, FrameResult], None] | None = None) -> list[FrameResult]:
        results: list[FrameResult] = []
        for i, (frame, timestamp) in enumerate(frames):
            if max_frames is not None and i >= max_frames:
                break
            result = self.process_frame(frame, camera_id, timestamp)
            results.append(result)
            if on_result is not None:
                on_result(frame, result)
        return results

    # ----------------------------------------------------------------- still
    def analyze_image(self, image: np.ndarray, camera_id: str = "camera", thermal: np.ndarray | None = None, with_plan: bool = True) -> FrameResult:
        """Analyse a single still. There is no temporal evidence, so the frame level stands as confirmed."""
        started = time.perf_counter()
        detector = self._detector_factory()
        detections, small, scale = self._detect(camera_id, image, 0.0, thermal, detector)
        zone = self._zone(camera_id)
        assessment = self.scorer.assess(detections, None, small.shape, zone.risk_multiplier if zone else 1.0)
        self._to_original(detections, scale)
        result = FrameResult(camera_id, time.time(), 0, detections, [], assessment, assessment.level)
        if with_plan and assessment.level > HazardLevel.CLEAR:
            result.plan = self.build_plan(camera_id, assessment, assessment.level, image)
        result.latency_ms = (time.perf_counter() - started) * 1000.0
        return result

    def render(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        return annotate(frame, result.detections, result.assessment, result.confirmed_level, result.camera_id)
