"""Turn what the cameras measured into retrieval queries.

The scene context is the bridge between perception and knowledge. It holds
only facts the system actually observed or was configured with (labels,
level, zone type, stored materials), so the queries built from it cannot
drift away from the incident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyraguard.schemas import FIRE, HOTSPOT, SMOKE, HazardLevel

ZONE_TAGS: dict[str, list[str]] = {
    "kitchen": ["cooking_oil", "kitchen", "class_f"],
    "server_room": ["electrical", "server_room", "gaseous_suppression"],
    "plant_room": ["electrical", "overheating"],
    "warehouse": ["warehouse", "racking", "sprinklers", "packaging"],
    "loading_bay": ["warehouse", "packaging"],
    "charging_bay": ["lithium", "battery", "thermal_runaway"],
    "chemical_store": ["flammable_liquids", "gas_cylinders", "chemical"],
    "office": ["general"],
    "corridor": ["evacuation", "fire_doors"],
}

MATERIAL_TAGS: dict[str, list[str]] = {
    "lithium_batteries": ["lithium", "battery", "thermal_runaway"],
    "cooking_oil": ["cooking_oil", "class_f"],
    "electrical_equipment": ["electrical"],
    "flammable_liquids": ["flammable_liquids", "chemical"],
    "gas_cylinders": ["gas_cylinders", "gas"],
    "cardboard": ["packaging", "warehouse"],
    "pallets": ["packaging", "warehouse"],
}

ZONE_QUERIES: dict[str, str] = {
    "kitchen": "cooking oil and fat fire in a kitchen: immediate actions, fire blanket, wet chemical extinguisher, never use water",
    "server_room": "fire in live electrical equipment in a server room: isolate power, carbon dioxide extinguisher, gas suppression discharge",
    "plant_room": "fire in electrical plant: isolate the supply, carbon dioxide extinguisher, overheating equipment",
    "warehouse": "fire in warehouse racking and stored goods: evacuation, sprinklers, do not shut the sprinkler valve",
    "loading_bay": "fire among stored goods and packaging in a loading bay: evacuation and forklift trucks",
    "charging_bay": "lithium ion battery thermal runaway fire in a charging area: immediate actions and what not to do",
    "chemical_store": "fire involving flammable liquids and gas cylinders: shut off supply, cylinders exposed to heat",
    "office": "fire in an office: first actions on discovering a fire and choosing an extinguisher",
    "corridor": "fire or smoke in a corridor escape route: alternative exits and fire doors",
}


@dataclass
class SceneContext:
    camera_id: str = "camera"
    zone_id: str | None = None
    zone_name: str = "the monitored area"
    zone_type: str = "general"
    materials: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    level: HazardLevel = HazardLevel.WATCH
    score: float = 0.0
    rationale: list[str] = field(default_factory=list)
    occupancy: int | None = None
    assisted_occupants: bool = False
    caption: str | None = None
    reference_tags: list[str] = field(default_factory=list)
    reference_captions: list[str] = field(default_factory=list)

    @property
    def hotspot_only(self) -> bool:
        return HOTSPOT in self.labels and FIRE not in self.labels and SMOKE not in self.labels

    def describe(self) -> str:
        seen = []
        if FIRE in self.labels:
            seen.append("flame")
        if SMOKE in self.labels:
            seen.append("smoke")
        if HOTSPOT in self.labels:
            seen.append("an abnormal thermal hot spot")
        what = " and ".join(seen) if seen else "a possible hazard"
        if self.hotspot_only:
            what += " with no flame or smoke"
        parts = [f"{self.level.title} level hazard: {what} detected in {self.zone_name} ({self.zone_type.replace('_', ' ')})."]
        if self.materials:
            parts.append("Materials present: " + ", ".join(m.replace("_", " ") for m in self.materials) + ".")
        if self.occupancy:
            parts.append("One person is normally in the zone." if self.occupancy == 1 else f"About {self.occupancy} people are normally in the zone.")
        if self.assisted_occupants:
            parts.append("Some occupants on site need assistance to evacuate.")
        if self.caption:
            parts.append(f"Camera view: {self.caption}")
        return " ".join(parts)

    def tags(self) -> list[str]:
        if self.hotspot_only:
            # nothing is burning yet: fetch pre ignition guidance, not the fire procedure for the room
            return sorted({"overheating", "thermal", "hotspot", "pre_ignition", *self.reference_tags})
        tags: list[str] = list(ZONE_TAGS.get(self.zone_type, []))
        for material in self.materials:
            tags.extend(MATERIAL_TAGS.get(material, []))
        if SMOKE in self.labels:
            tags.append("smoke")
        if FIRE in self.labels:
            tags.append("fire")
        if HOTSPOT in self.labels:
            tags.extend(["overheating", "thermal", "hotspot"])
        tags.extend(self.reference_tags)
        return sorted(set(tags))

    def queries(self) -> list[str]:
        """Decompose the incident into focused sub queries (situation, material, level, people)."""
        queries = [self.describe() + " What are the immediate actions and what must not be done?"]
        if self.hotspot_only:
            queries.append("overheating equipment thermal hot spot with no flame or smoke: actions before ignition")
        else:
            queries.append(ZONE_QUERIES.get(self.zone_type, "first actions on discovering a fire: raise the alarm, call 999, evacuate"))
        for material in self.materials:
            if material == "lithium_batteries" and self.zone_type != "charging_bay":
                queries.append(ZONE_QUERIES["charging_bay"])
            if material in ("flammable_liquids", "gas_cylinders") and self.zone_type != "chemical_store":
                queries.append(ZONE_QUERIES["chemical_store"])
        queries.append(f"Level {self.level.name} actions in the escalation matrix")
        if self.level >= HazardLevel.INCIPIENT:
            people = "evacuation, fire warden sweep, assembly point and roll call"
            if self.assisted_occupants:
                people += ", people who need assistance, refuge and evacuation chair"
            queries.append(people)
        if SMOKE in self.labels:
            queries.append("smoke hazards: moving through smoke, closing doors, what to do if trapped")
        if self.reference_captions:
            queries.append(" ".join(self.reference_captions[:2]))
        return queries

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id, "zone_id": self.zone_id, "zone_name": self.zone_name, "zone_type": self.zone_type,
            "materials": self.materials, "labels": self.labels, "level": self.level.name, "score": round(self.score, 1),
            "rationale": self.rationale, "occupancy": self.occupancy, "assisted_occupants": self.assisted_occupants,
            "caption": self.caption, "reference_tags": self.reference_tags,
        }
