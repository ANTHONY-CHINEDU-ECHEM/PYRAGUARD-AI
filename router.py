"""Hazard aware evacuation routing.

Routes are shortest paths over the site graph with three rules taken from
the guidance in the knowledge base:

* lifts are never used,
* a zone at or above the blocking level cannot be walked through (people
  inside it still leave, but nobody is routed into it),
* zones under suspicion, and zones next to a blocked zone, cost extra, so a
  slightly longer clean route beats a short route past the fire.

When no exit can be reached the origin is reported as trapped and directed
to a reachable refuge if one exists.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from pyraguard.config import EvacuationConfig
from pyraguard.evacuation.site import Site
from pyraguard.schemas import EvacuationRoute, HazardLevel


@dataclass
class EvacuationPlan:
    routes: list[EvacuationRoute] = field(default_factory=list)
    blocked_zones: list[str] = field(default_factory=list)
    trapped_zones: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"routes": [r.to_dict() for r in self.routes], "blocked_zones": self.blocked_zones, "trapped_zones": self.trapped_zones, "notes": self.notes}


class EvacuationRouter:
    def __init__(self, site: Site, config: EvacuationConfig | None = None) -> None:
        self.site = site
        self.cfg = config or EvacuationConfig()
        self.block_level = HazardLevel[self.cfg.block_level]

    def _classify(self, hazards: dict[str, HazardLevel]) -> tuple[set[str], set[str]]:
        blocked = {z for z, lv in hazards.items() if lv >= self.block_level and z in self.site.zones}
        caution = {z for z, lv in hazards.items() if HazardLevel.WATCH <= lv < self.block_level and z in self.site.zones}
        for zone_id in blocked:
            caution.update(n for n, edge in self.site.neighbours(zone_id) if edge.kind != "lift")
        return blocked, caution - blocked

    def _shortest(self, origin: str, targets: set[str], blocked: set[str], caution: set[str]) -> tuple[list[str], float] | None:
        """Dijkstra from origin to the cheapest target. Returns the path and its true length in metres."""
        best = {origin: 0.0}
        length = {origin: 0.0}
        parent: dict[str, str] = {}
        heap = [(0.0, origin)]
        while heap:
            cost, node = heapq.heappop(heap)
            if cost > best.get(node, float("inf")):
                continue
            if node in targets:
                path = [node]
                while path[-1] in parent:
                    path.append(parent[path[-1]])
                return path[::-1], length[node]
            for neighbour, edge in self.site.neighbours(node):
                if edge.kind == "lift" or neighbour in blocked:
                    continue
                step = edge.distance_m * (self.cfg.caution_penalty if neighbour in caution else 1.0)
                if cost + step < best.get(neighbour, float("inf")):
                    best[neighbour] = cost + step
                    length[neighbour] = length[node] + edge.distance_m
                    parent[neighbour] = node
                    heapq.heappush(heap, (cost + step, neighbour))
        return None

    def route_from(self, origin: str, hazards: dict[str, HazardLevel]) -> EvacuationRoute | None:
        zones = self.site.zones
        if origin not in zones:
            raise KeyError(f"Unknown zone: {origin}")
        blocked, caution = self._classify(hazards)
        exits = {z for z, zone in zones.items() if zone.is_exit and (z not in blocked or z == origin)}
        found = self._shortest(origin, exits, blocked - {origin}, caution)
        if found is None:
            return None
        path, distance = found
        exit_zone = zones[path[-1]]
        speed = self.cfg.assisted_speed_mps if zones[origin].assisted else self.cfg.walking_speed_mps
        assembly = self.site.assembly_points.get(exit_zone.assembly_point or "", exit_zone.assembly_point or "the assembly point")
        via = [zones[z].name for z in path[1:]]
        if via:
            instructions = f"From {zones[origin].name} go via {', '.join(via)} and leave by the {exit_zone.exit_name}. Assemble at {assembly}."
        else:
            instructions = f"From {zones[origin].name} leave directly by the {exit_zone.exit_name}. Assemble at {assembly}."
        if zones[origin].assisted and any(zones[z].type == "stair" for z in path):
            refuge = next((zones[z].name for z in path if zones[z].refuge), None)
            if refuge:
                instructions += f" Anyone who cannot use the stairs waits in the refuge at {refuge} and uses the communication point."
        return EvacuationRoute(origin, path, exit_zone.zone_id, assembly, distance, distance / max(speed, 0.1), instructions)

    def plan(self, hazards: dict[str, HazardLevel], origins: list[str] | None = None) -> EvacuationPlan:
        """Route every occupied zone (or the given origins) to its best safe exit."""
        zones = self.site.zones
        blocked, _ = self._classify(hazards)
        origins = origins or [z for z, zone in zones.items() if zone.occupancy > 0]
        plan = EvacuationPlan(blocked_zones=sorted(blocked))
        for origin in origins:
            route = self.route_from(origin, hazards)
            if route is not None:
                plan.routes.append(route)
                continue
            plan.trapped_zones.append(origin)
            refuges = {z for z, zone in zones.items() if zone.refuge and z not in blocked}
            reachable = self._shortest(origin, refuges, blocked - {origin}, set()) if refuges else None
            if reachable:
                plan.notes.append(f"No safe exit from {zones[origin].name}: move to the refuge at {zones[reachable[0][-1]].name}, close doors behind you and call 999 with your exact location.")
            else:
                plan.notes.append(f"No safe exit from {zones[origin].name}: stay in a room with a window, close the door, block gaps under it and call 999 with your exact location.")
        plan.routes.sort(key=lambda r: r.distance_m)
        return plan
