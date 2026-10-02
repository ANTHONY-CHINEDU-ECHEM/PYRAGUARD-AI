"""Reproducible pipeline benchmark on generated scenes.

Three questions are answered, each with exact ground truth:

1. Stills: how well does the detector localise fire and smoke in one image?
2. Clips: how quickly is a real ignition confirmed, and how many false
   incidents do hard negatives cause, compared with alarming on any single
   frame?
3. Thermal: at what surface temperature does the hot spot channel respond?

The benchmark measures the pipeline that is configured. With no trained
weights that is the classical detector; after training, the same command
measures the YOLO ensemble.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from pyraguard.config import Settings, load_settings
from pyraguard.data.synthetic import (
    SCENARIOS,
    generate_image,
    generate_sequence,
    generate_thermal_frame,
    sequence_spec,
)
from pyraguard.evaluation.detection_metrics import ClassStats, binary_summary
from pyraguard.logging_utils import get_logger
from pyraguard.pipeline import PyraGuardEngine
from pyraguard.schemas import FIRE, SMOKE, HazardLevel
from pyraguard.vision.base import resize_to_width
from pyraguard.vision.ensemble import build_detector
from pyraguard.vision.thermal import ThermalAnalyzer

log = get_logger(__name__)

SIZE_BUCKETS = (("tiny (under 0.3 percent of frame)", 0.0, 0.003), ("small (0.3 to 1.5 percent)", 0.003, 0.015), ("medium and large (over 1.5 percent)", 0.015, 1.0))


def run_still_benchmark(settings: Settings, count: int = 1500, seed: int = 2026, iou_threshold: float = 0.3) -> dict:
    stats = {FIRE: ClassStats(), SMOKE: ClassStats()}
    by_size = {name: [0, 0] for name, _, _ in SIZE_BUCKETS}  # found, total
    tp = fp = tn = fn = 0
    false_boxes_on_negatives = 0
    categories: dict[str, int] = {}
    latencies: list[float] = []
    detector_name = ""
    for i in range(count):
        sample = generate_image(seed * 1_000_003 + i)
        categories[sample.meta["category"]] = categories.get(sample.meta["category"], 0) + 1
        detector = build_detector(settings)  # fresh state: every still is independent
        detector_name = detector.name
        frame, _ = resize_to_width(sample.image, settings.vision.process_width)
        started = time.perf_counter()
        detections = detector.detect(frame)
        latencies.append((time.perf_counter() - started) * 1000.0)
        h, w = frame.shape[:2]
        for label in (FIRE, SMOKE):
            truths = [box for name, box in sample.labels if name == label]
            matched = stats[label].add_image([d for d in detections if d.label == label], truths, iou_threshold)
            if label == FIRE:
                for box, found in zip(truths, matched):
                    fraction = box.area / float(h * w)
                    for name, low, high in SIZE_BUCKETS:
                        if low <= fraction < high:
                            by_size[name][1] += 1
                            by_size[name][0] += int(found)
        has_fire = any(name == FIRE for name, _ in sample.labels)
        flagged = any(d.label == FIRE for d in detections)
        tp += int(has_fire and flagged)
        fn += int(has_fire and not flagged)
        fp += int(not has_fire and flagged)
        tn += int(not has_fire and not flagged)
        if not sample.labels:
            false_boxes_on_negatives += len(detections)
    fire, smoke = stats[FIRE].summary(), stats[SMOKE].summary()
    return {
        "images": count, "seed": seed, "iou_threshold": iou_threshold, "detector": detector_name, "categories": categories,
        "fire": fire, "smoke": smoke,
        "map": round((fire["average_precision"] + smoke["average_precision"]) / 2.0, 4),
        "fire_recall_by_size": {k: {"found": v[0], "total": v[1], "recall": round(v[0] / v[1], 4) if v[1] else None} for k, v in by_size.items()},
        "image_level_fire": binary_summary(tp, fp, tn, fn),
        "false_boxes_per_negative_image": round(false_boxes_on_negatives / max(categories.get("none", 0), 1), 4),
        "latency_ms": {"median": round(float(np.median(latencies)), 2), "p95": round(float(np.percentile(latencies, 95)), 2)},
    }


def run_sequence_benchmark(settings: Settings, clips: int = 60, seed: int = 2026) -> dict:
    per_scenario: dict[str, dict] = {s: {"clips": 0, "incidents": 0, "raw_alarms": 0, "delays": []} for s in SCENARIOS}
    growth_errors: list[float] = []
    latencies: list[float] = []
    peak_levels: dict[str, int] = {}
    for i in range(clips):
        spec = sequence_spec(seed * 7_919 + i, SCENARIOS[i % len(SCENARIOS)] if i < 2 * len(SCENARIOS) else None)
        engine = PyraGuardEngine(settings, enable_rag=False, enable_alerts=False)
        bucket = per_scenario[spec.scenario]
        bucket["clips"] += 1
        opened_at: int | None = None
        raw_alarm = False
        peak = HazardLevel.CLEAR
        estimates: list[float] = []
        for frame in generate_sequence(spec):
            result = engine.process_frame(frame.image, "bench", frame.timestamp)
            latencies.append(result.latency_ms)
            raw_alarm = raw_alarm or bool(result.detections)
            peak = max(peak, result.confirmed_level)
            if result.event is not None and result.event.kind == "opened" and opened_at is None:
                opened_at = frame.index
            if spec.scenario == "ignition" and spec.ignition_frame is not None:
                elapsed = (frame.index - spec.ignition_frame) / spec.fps
                fire_tracks = [t for t in result.dynamics if t.label == FIRE and t.growth_rate > 0]
                if 2.0 <= elapsed <= 4.5 and fire_tracks:
                    estimates.append(max(t.growth_rate for t in fire_tracks))
        bucket["raw_alarms"] += int(raw_alarm)
        if opened_at is not None:
            bucket["incidents"] += 1
            if spec.ignition_frame is not None:
                bucket["delays"].append((opened_at - spec.ignition_frame) / spec.fps)
        if spec.scenario == "ignition":
            peak_levels[peak.name] = peak_levels.get(peak.name, 0) + 1
            if estimates and spec.doubling_seconds:
                estimated_doubling = math.log(2) / float(np.median(estimates))
                growth_errors.append(abs(estimated_doubling - spec.doubling_seconds) / spec.doubling_seconds)

    def positive(name: str) -> dict:
        b = per_scenario[name]
        delays = b["delays"]
        return {
            "clips": b["clips"], "detected": b["incidents"], "detection_rate": round(b["incidents"] / max(b["clips"], 1), 4),
            "median_seconds_to_confirm": round(float(np.median(delays)), 2) if delays else None,
            "p90_seconds_to_confirm": round(float(np.percentile(delays, 90)), 2) if delays else None,
            "max_seconds_to_confirm": round(float(np.max(delays)), 2) if delays else None,
        }

    def negative(name: str) -> dict:
        b = per_scenario[name]
        return {
            "clips": b["clips"], "false_incidents": b["incidents"], "false_incident_rate": round(b["incidents"] / max(b["clips"], 1), 4),
            "clips_with_any_raw_detection": b["raw_alarms"], "raw_alarm_rate": round(b["raw_alarms"] / max(b["clips"], 1), 4),
        }

    neg_clips = per_scenario["negative"]["clips"] + per_scenario["transient"]["clips"]
    neg_incidents = per_scenario["negative"]["incidents"] + per_scenario["transient"]["incidents"]
    neg_raw = per_scenario["negative"]["raw_alarms"] + per_scenario["transient"]["raw_alarms"]
    pos_clips = per_scenario["ignition"]["clips"] + per_scenario["smoulder"]["clips"]
    pos_found = per_scenario["ignition"]["incidents"] + per_scenario["smoulder"]["incidents"]
    return {
        "clips": clips, "seed": seed, "frames_per_clip": 90, "fps": 10.0,
        "ignition": positive("ignition"), "smoulder": positive("smoulder"),
        "moving_hi_vis_negative": negative("negative"), "light_flash_negative": negative("transient"),
        "overall": {
            "positive_clips": pos_clips, "detection_rate": round(pos_found / max(pos_clips, 1), 4),
            "negative_clips": neg_clips, "false_incident_rate_confirmed": round(neg_incidents / max(neg_clips, 1), 4),
            "false_alarm_rate_single_frame": round(neg_raw / max(neg_clips, 1), 4),
        },
        "ignition_peak_levels": peak_levels,
        "growth_estimate_median_relative_error": round(float(np.median(growth_errors)), 4) if growth_errors else None,
        "latency_ms": {"median": round(float(np.median(latencies)), 2), "p95": round(float(np.percentile(latencies, 95)), 2)},
    }


def run_thermal_benchmark(settings: Settings, per_temperature: int = 60, seed: int = 2026) -> dict:
    analyzer = ThermalAnalyzer(settings.vision.thermal)
    rows = []
    for temperature in (None, 45.0, 60.0, 75.0, 100.0, 180.0, 300.0):
        flagged = alarm = 0
        for i in range(per_temperature):
            temp_map, _ = generate_thermal_frame(seed + i * 13 + int(temperature or 0), temperature)
            detections = analyzer.analyze(temp_map)
            flagged += int(bool(detections))
            alarm += int(any(d.attributes.get("above_alarm") for d in detections))
        rows.append({
            "hot_spot_celsius": temperature, "frames": per_temperature,
            "flagged_rate": round(flagged / per_temperature, 4), "above_alarm_rate": round(alarm / per_temperature, 4),
        })
    return {"alarm_celsius": settings.vision.thermal.alarm_celsius, "ambient_margin_celsius": settings.vision.thermal.ambient_margin_celsius, "rows": rows}


def run_benchmark(settings: Settings | None = None, stills: int = 1500, clips: int = 60, seed: int = 2026, out_path: str | Path | None = None) -> dict:
    settings = settings or load_settings()
    started = time.time()
    report = {
        "note": "Measured on procedurally generated scenes with exact ground truth. Not a substitute for validation on real footage.",
        "stills": run_still_benchmark(settings, stills, seed),
        "clips": run_sequence_benchmark(settings, clips, seed),
        "thermal": run_thermal_benchmark(settings, seed=seed),
    }
    report["runtime_seconds"] = round(time.time() - started, 1)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("Benchmark report written to %s", out_path)
    return report
