"""Draw the site plan with hazard states and evacuation routes (used by the dashboard and the README figures)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pyraguard.evacuation.site import Site  # noqa: E402
from pyraguard.schemas import EvacuationRoute, HazardLevel  # noqa: E402

LEVEL_FACE = {
    HazardLevel.CLEAR: "#eef2f5", HazardLevel.WATCH: "#fff3bf", HazardLevel.INCIPIENT: "#ffd8a8",
    HazardLevel.GROWING: "#ff8787", HazardLevel.CRITICAL: "#c92a2a",
}
ROUTE_COLOURS = ["#1971c2", "#2f9e44", "#7048e8", "#e8590c", "#0c8599", "#a61e4d", "#5c940d", "#364fc7"]


def draw_site(site: Site, hazards: dict[str, HazardLevel] | None = None, routes: list[EvacuationRoute] | None = None, title: str | None = None):
    hazards = hazards or {}
    routes = routes or []
    floors = sorted({z.floor for z in site.zones.values()})
    fig, axes = plt.subplots(1, len(floors), figsize=(7.2 * len(floors), 5.2), squeeze=False)
    for ax, floor in zip(axes[0], floors):
        ax.set_title("Ground floor" if floor == 0 else ("First floor" if floor == 1 else f"Floor {floor}"), fontsize=12, loc="left", color="#343a40")
        for edge in site.edges:
            a, b = site.zones[edge.a], site.zones[edge.b]
            if a.floor != floor or b.floor != floor:
                continue
            ax.plot([a.position[0], b.position[0]], [a.position[1], b.position[1]], color="#adb5bd", lw=1.4, zorder=1,
                    ls=":" if edge.kind == "lift" else "-")
        for r_idx, route in enumerate(routes):
            colour = ROUTE_COLOURS[r_idx % len(ROUTE_COLOURS)]
            for a_id, b_id in zip(route.path, route.path[1:]):
                a, b = site.zones[a_id], site.zones[b_id]
                if a.floor == floor and b.floor == floor:
                    ax.annotate("", xy=b.position, xytext=a.position, zorder=3,
                                arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 2.6, "shrinkA": 21, "shrinkB": 21,
                                            "connectionstyle": f"arc3,rad={0.06 + 0.035 * (r_idx % 4)}"})
        for zone in site.zones.values():
            if zone.floor != floor:
                continue
            level = hazards.get(zone.zone_id, HazardLevel.CLEAR)
            face = LEVEL_FACE[level]
            edge_colour = "#2b8a3e" if zone.is_exit else ("#1864ab" if zone.refuge else "#495057")
            ax.scatter(*zone.position, s=1500, marker="s" if zone.type != "stair" else "D", c=face, edgecolors=edge_colour,
                       linewidths=2.6 if (zone.is_exit or zone.refuge) else 1.2, zorder=4)
            label = zone.name.replace(" ", "\n", 1) if len(zone.name) > 12 else zone.name
            text_colour = "white" if level == HazardLevel.CRITICAL else "#212529"
            ax.text(*zone.position, label, ha="center", va="center", fontsize=7.4, zorder=5, color=text_colour)
            notes = []
            if zone.is_exit:
                notes.append(f"EXIT: {zone.exit_name}")
            if zone.refuge:
                notes.append("REFUGE")
            if level > HazardLevel.CLEAR:
                notes.append(level.name)
            if notes:
                ax.text(zone.position[0], zone.position[1] - 6.2, " | ".join(notes), ha="center", va="top", fontsize=6.8,
                        color="#c92a2a" if level > HazardLevel.CLEAR else edge_colour, zorder=5, weight="bold")
        ax.set_xlim(0, 100)
        ax.set_ylim(62, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#ced4da")
    fig.suptitle(title or site.name, fontsize=13, weight="bold", color="#212529")
    fig.tight_layout()
    return fig
