"""Site model: zones, the walkable connections between them, exits, refuges and cameras."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Zone:
    zone_id: str
    name: str
    type: str = "general"
    floor: int = 0
    position: tuple[float, float] = (0.0, 0.0)
    risk_multiplier: float = 1.0
    materials: list[str] = field(default_factory=list)
    occupancy: int = 0
    assisted: bool = False
    refuge: bool = False
    exit_name: str | None = None
    assembly_point: str | None = None

    @property
    def is_exit(self) -> bool:
        return self.exit_name is not None


@dataclass
class Edge:
    a: str
    b: str
    distance_m: float
    kind: str = "door"  # door | open | stair | lift


@dataclass
class Site:
    name: str
    zones: dict[str, Zone]
    edges: list[Edge]
    assembly_points: dict[str, str]
    cameras: dict[str, dict]

    @classmethod
    def from_yaml(cls, path: str | Path) -> Site:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        zones: dict[str, Zone] = {}
        for z in raw.get("zones", []):
            exit_info = z.get("exit") or {}
            zones[z["id"]] = Zone(
                zone_id=z["id"], name=z.get("name", z["id"]), type=z.get("type", "general"), floor=int(z.get("floor", 0)),
                position=tuple(z.get("position", (0, 0))), risk_multiplier=float(z.get("risk_multiplier", 1.0)),
                materials=list(z.get("materials", [])), occupancy=int(z.get("occupancy", 0)), assisted=bool(z.get("assisted", False)),
                refuge=bool(z.get("refuge", False)), exit_name=exit_info.get("name"), assembly_point=exit_info.get("assembly_point"),
            )
        edges = [Edge(e[0], e[1], float(e[2]), e[3] if len(e) > 3 else "door") for e in raw.get("edges", [])]
        for edge in edges:
            if edge.a not in zones or edge.b not in zones:
                raise ValueError(f"Edge references an unknown zone: {edge.a} to {edge.b}")
        site = raw.get("site", {})
        points = {k: v.get("name", k) for k, v in (site.get("assembly_points") or {}).items()}
        return cls(site.get("name", "Site"), zones, edges, points, raw.get("cameras", {}) or {})

    def zone_for_camera(self, camera_id: str) -> Zone | None:
        info = self.cameras.get(camera_id)
        return self.zones.get(info["zone"]) if info else None

    def neighbours(self, zone_id: str) -> list[tuple[str, Edge]]:
        out = []
        for edge in self.edges:
            if edge.a == zone_id:
                out.append((edge.b, edge))
            elif edge.b == zone_id:
                out.append((edge.a, edge))
        return out
