"""Typed configuration.

Settings are read from ``configs/default.yaml`` (or the file named by the
``PYRAGUARD_CONFIG`` environment variable) and validated with pydantic.
Secrets such as API keys never live in YAML; they are read from the
environment at the point of use.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]


class HeuristicConfig(BaseModel):
    red_min: int = 170
    saturation_min: int = 55
    cbcr_gap_min: int = 22
    min_area_fraction: float = 0.00035
    min_confidence: float = 0.60
    smoke_enabled: bool = True
    smoke_saturation_max: int = 70
    smoke_value_min: int = 70
    smoke_diff_min: int = 14
    smoke_min_area_fraction: float = 0.004
    smoke_activity_ratio: float = 3.0
    ghost_quiet_ratio: float = 1.6
    background_learning_rate: float = 0.02
    ghost_absorb_frames: int = 8
    warmup_frames: int = 5


class YoloConfig(BaseModel):
    weights: str = "models/pyraguard_yolo.pt"
    base_weights: str = "yolo26s.pt"
    confidence: float = 0.30
    iou: float = 0.50
    image_size: int = 640
    device: str = ""
    class_map: dict[int, str] = Field(default_factory=lambda: {0: "smoke", 1: "fire"})


class ThermalConfig(BaseModel):
    enabled: bool = True
    alarm_celsius: float = 80.0
    ambient_margin_celsius: float = 35.0
    min_area_pixels: int = 9
    range_min_celsius: float = 0.0
    range_max_celsius: float = 400.0


class VisionConfig(BaseModel):
    detector: str = "auto"  # auto | heuristic | yolo | ensemble
    process_width: int = 640
    heuristic: HeuristicConfig = Field(default_factory=HeuristicConfig)
    yolo: YoloConfig = Field(default_factory=YoloConfig)
    thermal: ThermalConfig = Field(default_factory=ThermalConfig)


class TrackingConfig(BaseModel):
    iou_match: float = 0.15
    max_age_seconds: float = 1.5
    window_seconds: float = 6.0
    min_hits: int = 3


class HazardConfig(BaseModel):
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "fire_extent": 0.30,
            "smoke_extent": 0.14,
            "confidence": 0.14,
            "growth": 0.18,
            "persistence": 0.14,
            "thermal": 0.10,
        }
    )
    thresholds: dict[str, float] = Field(
        default_factory=lambda: {"watch": 8.0, "incipient": 24.0, "growing": 48.0, "critical": 72.0}
    )
    growth_saturation_per_second: float = 0.15
    persistence_saturation_seconds: float = 3.0
    confirm_seconds: float = 1.2
    confirm_ratio: float = 0.6
    clear_after_seconds: float = 20.0


class RagConfig(BaseModel):
    knowledge_dir: str = "knowledge_base/documents"
    reference_image_dir: str = "knowledge_base/reference_images"
    index_dir: str = "storage/index"
    text_embedder: str = "hashing"  # hashing | sentence_transformers | open_clip
    text_model: str = "BAAI/bge-small-en-v1.5"
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    embedding_dim: int = 2048
    chunk_words: int = 170
    chunk_overlap_words: int = 30
    top_k: int = 6
    plan_top_k: int = 10
    reference_min_similarity: float = 0.80
    candidate_k: int = 24
    rrf_k: int = 60
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    tag_boost: float = 0.25
    max_chunks_per_document: int = 2
    min_support: float = 0.55
    max_actions: int = 10
    max_prohibitions: int = 6


class LLMConfig(BaseModel):
    provider: str = "extractive"  # extractive | anthropic | openai | ollama
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 900
    timeout_seconds: float = 30.0
    vision_captions: bool = False


class AlertConfig(BaseModel):
    channels: list[str] = Field(default_factory=lambda: ["console", "jsonl"])
    min_level: str = "INCIPIENT"
    cooldown_seconds: float = 30.0
    jsonl_path: str = "reports/incidents.jsonl"


class EvacuationConfig(BaseModel):
    site_file: str = "configs/site_demo.yaml"
    walking_speed_mps: float = 1.2
    assisted_speed_mps: float = 0.6
    block_level: str = "GROWING"
    caution_penalty: float = 4.0


class Settings(BaseModel):
    vision: VisionConfig = Field(default_factory=VisionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    hazard: HazardConfig = Field(default_factory=HazardConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    evacuation: EvacuationConfig = Field(default_factory=EvacuationConfig)

    def resolve(self, relative: str) -> Path:
        """Resolve a repository relative path against the project root."""
        path = Path(relative)
        return path if path.is_absolute() else PROJECT_ROOT / path


def _deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_settings(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Settings:
    """Load settings from YAML, then apply environment and explicit overrides."""
    candidate = path or os.environ.get("PYRAGUARD_CONFIG") or PROJECT_ROOT / "configs" / "default.yaml"
    raw: dict[str, Any] = {}
    if candidate and Path(candidate).exists():
        raw = yaml.safe_load(Path(candidate).read_text(encoding="utf-8")) or {}
    env_provider = os.environ.get("PYRAGUARD_LLM_PROVIDER")
    env_model = os.environ.get("PYRAGUARD_LLM_MODEL")
    if env_provider:
        raw.setdefault("llm", {})["provider"] = env_provider
    if env_model:
        raw.setdefault("llm", {})["model"] = env_model
    if overrides:
        _deep_update(raw, overrides)
    return Settings.model_validate(raw)
