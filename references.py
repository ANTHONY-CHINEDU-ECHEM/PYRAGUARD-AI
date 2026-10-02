"""Build the reference image library used by the image leg of retrieval.

Each reference picture carries a caption and hazard tags written for it. A
production deployment should replace these rendered scenes with real,
reviewed photographs from its own cameras (clear views, drills, past
incidents), keeping the same manifest format.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from pyraguard.data.synthetic import SCENE_KINDS, generate_image
from pyraguard.rag.scene import ZONE_TAGS

_VARIANTS = [
    ("small_fire", "fire", 0.08, "A small flame at an early stage in {place}, no smoke layer yet."),
    ("large_fire", "fire_and_smoke", 0.30, "A large flame with a rising smoke plume in {place}."),
    ("smoke_only", "smoke", 0.12, "Grey smoke with no visible flame in {place}, consistent with a smouldering source."),
    ("clear", "none", None, "A normal view of {place} with no flame or smoke."),
]
_PLACES = {
    "office": "an office", "warehouse": "a warehouse with racked goods", "kitchen": "a kitchen",
    "server_room": "a server room with equipment cabinets", "corridor": "a corridor escape route",
}


def build_reference_library(out_dir: str | Path, seed: int = 4242, width: int = 320, height: int = 240) -> list[dict]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for s_idx, scene in enumerate(SCENE_KINDS):
        for v_idx, (variant, category, scale, caption) in enumerate(_VARIANTS):
            sample = generate_image(seed + 97 * s_idx + v_idx, category=category, scene=scene, fire_scale=scale)
            name = f"ref_{scene}_{variant}.jpg"
            cv2.imwrite(str(out / name), cv2.resize(sample.image, (width, height), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
            tags = [] if category == "none" else list(ZONE_TAGS.get(scene, []))
            if category in ("smoke", "fire_and_smoke"):
                tags.append("smoke")
            manifest.append({"file": name, "scene": scene, "variant": variant, "caption": caption.format(place=_PLACES[scene]), "tags": sorted(set(tags))})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
