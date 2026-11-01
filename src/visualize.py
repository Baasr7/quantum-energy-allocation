"""Launch the local interactive dashboard for the energy system.

Run ``python -m src.visualize`` and open http://127.0.0.1:8765 in a browser.
The plot_allocations helper remains available to callers that explicitly request
a static comparison export; the default experience is the live local dashboard.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


BG = "#071522"
PANEL = "#102437"
PANEL_ALT = "#142c40"
INK = "#eff7f5"
MUTED = "#91a9b7"
TEAL = "#48e0c1"
GOLD = "#ffc857"
BLUE = "#53b9ff"
PINK = "#ff6d91"
SOURCE_COLORS = {
    "solar": GOLD, "wind": TEAL, "hydro": BLUE,
    "coal": "#d48a66", "gas": "#c18aff", "battery": "#f27fb0",
}


def _get_energy_data():
    try:
        from . import energy_data as ed
    except ImportError:
        import energy_data as ed
    return ed


def _curve(ax, start, end, color, label, linestyle="-", bend=0.13, width=2.4):
    """Draw a schematic transport corridor between real site coordinates."""
    sx, sy = start[0], start[1] * 0.78
    ex, ey = end[0], end[1] * 0.78
    patch = FancyArrowPatch(
        (sx, sy), (ex, ey), arrowstyle="-|>", mutation_scale=14,
        connectionstyle=f"arc3,rad={bend}", linewidth=width,
        linestyle=linestyle, color=color, alpha=0.88, zorder=3,
    )
    ax.add_patch(patch)
    if label:
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        ax.text(mx, my + (0.38 if bend >= 0 else -0.45), label,
                color=color, fontsize=8.5, weight="bold", ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.27", "fc": BG, "ec": color,
                      "lw": 0.8, "alpha": 0.94}, zorder=8)


def _sun(ax, x, y, scale=1.0):
    ax.scatter([x], [y], s=185 * scale, marker="o", color=GOLD,
               edgecolors="#fff2bf", linewidths=1.3, zorder=7)
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        ax.plot([x + np.cos(angle) * .30 * scale, x + np.cos(angle) * .44 * scale],
                [y + np.sin(angle) * .30 * scale, y + np.sin(angle) * .44 * scale],
                color=GOLD, linewidth=1.8, solid_capstyle="round", zorder=7)


def _cloud(ax, x, y):
    for dx, dy, r in [(-.18, 0, .15), (0, .08, .2), (.2, 0, .15)]:
        ax.add_patch(Circle((x + dx, y + dy), r, color="#b5d6e6", zorder=7))
    ax.add_patch(Rectangle((x - .23, y - .11), .47, .18, color="#b5d6e6", zorder=7))
    for dx in [-.15, 0, .15]:
        ax.plot([x + dx, x + dx - .05], [y - .18, y - .28], color=BLUE,
                linewidth=1.8, zorder=7)


def _turbine(ax, x, y):
    ax.plot([x, x], [y - .48, y + .02], color=TEAL, linewidth=2.1, zorder=7)
    ax.scatter([x], [y + .02], s=22, color=TEAL, zorder=8)
    for angle in (90, 210, 330):
        a = np.deg2rad(angle)
        ax.plot([x, x + .31 * np.cos(a)], [y + .02, y + .02 + .31 * np.sin(a)],
                color=TEAL, linewidth=2, solid_capstyle="round", zorder=7)


def _site_label(ax, x, y, title, detail, color, dx, dy, align="left"):
    ax.scatter([x], [y], s=118, color=color, edgecolors=INK, linewidths=1.2, zorder=8)
    ax.scatter([x], [y], s=215, color=color, alpha=.17, zorder=7)
    ax.annotate(
        f"{title}\n{detail}", (x, y), xytext=(x + dx, y + dy), textcoords="data",
        color=INK, fontsize=9.2, weight="bold", ha=align, va="center",
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=.42", "fc": PANEL, "ec": color,
              "lw": .9, "alpha": .97},
        arrowprops={"arrowstyle": "-", "color": color, "lw": 1.1}, zorder=9,
    )


def _draw_map(ax, ed, sites, source_rows):
    ax.set_facecolor("#0a2637")
    ax.set_xlim(68.7, 88.7)
    ax.set_ylim(4.0, 31.5 * .78)
    ax.grid(color="#6b92a3", alpha=.10, linewidth=.7)
    ax.tick_params(colors=MUTED, labelsize=8, length=0, pad=7)
    ax.set_xticks([70, 75, 80, 85], ["70°E", "75°E", "80°E", "85°E"])
    ax.set_yticks(np.array([5, 10, 15, 20, 25, 30]) * .78,
                  ["5°N", "10°N", "15°N", "20°N", "25°N", "30°N"])
    for spine in ax.spines.values():
        spine.set_color("#29465b")

    # Stylized southern and central India silhouette. Site positions remain
    # projected from the actual latitude/longitude values in energy_data.py.
    land = np.array([
        [68.2, 35.5], [72.0, 35.4], [73.8, 33.5], [76.0, 31.8], [78.3, 29.5],
        [81.0, 28.6], [84.6, 27.0], [87.2, 25.3], [87.5, 22.3], [85.7, 20.0],
        [84.1, 18.4], [82.5, 16.4], [81.1, 14.4], [80.6, 12.2], [80.0, 10.3],
        [79.2, 8.2], [77.8, 6.3], [76.2, 8.0], [75.0, 10.1], [74.2, 12.0],
        [73.0, 15.0], [72.2, 18.2], [70.7, 21.2], [69.2, 24.5], [68.5, 28.5],
    ])
    land[:, 1] *= .78
    ax.add_patch(Polygon(land, closed=True, facecolor="#153b3b", edgecolor="#37675e",
                         linewidth=1.3, alpha=.94, zorder=1))
    ax.text(77.6, 17.8 * .78, "INDIA", color="#c1ddc8", alpha=.25,
            fontsize=22, weight="bold", rotation=-8, ha="center", zorder=2)
    ax.text(69.25, 12.8 * .78, "ARABIAN SEA", color="#6da9bc", alpha=.67,
            fontsize=8.5, style="italic", rotation=75, ha="center", zorder=1)
    ax.text(86.9, 12.0 * .78, "BAY OF BENGAL", color="#6da9bc", alpha=.67,
            fontsize=8.2, style="italic", rotation=79, ha="center", zorder=1)

    # Rain-fed reservoir and the west-flowing water corridor around Idukki.
    idukki = sites.loc["idukki"]
    ix, iy = idukki.lon, idukki.lat * .78
    ax.add_patch(Circle((ix, iy), .68, facecolor=BLUE, edgecolor="#a1e7ff",
                        linewidth=1.1, alpha=.24, zorder=4))
    ax.add_patch(Circle((ix, iy), .38, facecolor=BLUE, edgecolor="none", alpha=.72, zorder=5))
    water_path = MplPath(
        [(ix + .1, iy - .12), (ix - .2, iy - .46),
         (76.5, 6.9 * .78), (75.9, 7.8 * .78)],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(PathPatch(water_path, edgecolor=BLUE, facecolor="none",
                           linewidth=2.2, alpha=.8, zorder=3))
    ax.text(ix + .52, iy - .42, "reservoir", color=BLUE, fontsize=7.5,
            style="italic", zorder=7)

    hub = sites.loc[ed.DEMAND_SITE]
    routes = [
        ("solar", "power grid", "-", GOLD, .19),
        ("wind", "power grid", "-", TEAL, -.16),
        ("hydro", "hydro line", "-", BLUE, -.23),
        ("coal", "rail freight", "--", "#e99869", .13),
        ("gas", "gas pipeline", ":", "#c18aff", .15),
    ]
    for source, label, style, color, bend in routes:
        plant = next((p for p in ed.PLANTS if p["source"] == source), None)
        if plant is None:
            continue
        origin = sites.loc[plant["origin"]]
        _curve(ax, (origin.lon, origin.lat), (hub.lon, hub.lat), color, None,
               linestyle=style, bend=bend,
               width=2.9 if source in {"solar", "wind", "hydro"} else 2.1)

    route_legend = [
        Line2D([0], [0], color=GOLD, lw=2.7, label="Solar power grid"),
        Line2D([0], [0], color=TEAL, lw=2.7, label="Wind power grid"),
        Line2D([0], [0], color=BLUE, lw=2.7, label="Hydro line"),
        Line2D([0], [0], color="#e99869", lw=2.2, ls="--", label="Coal rail"),
        Line2D([0], [0], color="#c18aff", lw=2.2, ls=":", label="Gas pipeline"),
    ]
    legend = ax.legend(handles=route_legend, loc="upper right", bbox_to_anchor=(.985, .91),
                       fontsize=7.2, labelcolor=INK, frameon=True, borderpad=.65,
                       handlelength=2.3, facecolor=PANEL, edgecolor="#36536a", framealpha=.96)
    legend.set_zorder(10)

    # Weather cues that are based on the modeled site conditions.
    bhadla, muppandal = sites.loc["bhadla"], sites.loc["muppandal"]
    _sun(ax, bhadla.lon + .55, bhadla.lat * .78 + .55)
    _turbine(ax, muppandal.lon - .62, muppandal.lat * .78 + .48)
    _cloud(ax, idukki.lon + .65, iy + .66)

    # Site labels use actual weather metrics and identify the resource placed there.
    source_meta = {p["site"]: p["source"] for p in ed.PLANTS}
    specs = {
        "bhadla": (1.1, 1.05, "left"), "muppandal": (-1.4, -1.2, "right"),
        "idukki": (-1.35, -1.3, "right"), "talcher": (-1.1, 1.25, "right"),
        "puthuvype": (1.15, -1.0, "left"), "kochi": (1.35, 1.15, "left"),
    }
    for site_name, (dx, dy, align) in specs.items():
        row = sites.loc[site_name]
        source = source_meta.get(site_name, "")
        if site_name == "talcher":
            title = "Talcher · coal rail origin"
            detail = f"{row.rain_mm:.1f} mm rain · {row.wind_ms:.1f} m/s wind"
        elif site_name == "puthuvype":
            title = "Puthuvype · gas terminal"
            detail = f"{row.rain_mm:.1f} mm rain · {row.cloud_pct:.0f}% cloud"
        elif site_name == "kochi":
            title = "Kochi · demand hub"
            detail = f"{ed.DEMAND_MW} MW demand · {row.population_m:.1f}M people"
        elif source == "solar":
            title = "Bhadla · solar"
            detail = f"{row.sunlight_h:.1f} h sun · {row.cloud_pct:.0f}% cloud"
        elif source == "wind":
            title = "Muppandal · wind"
            detail = f"{row.wind_ms:.1f} m/s wind · {row.sunlight_h:.1f} h sun"
        else:
            title = "Idukki · hydro"
            detail = f"{row.rain_mm:.1f} mm rain · {row.altitude_m:.0f} m elevation"
        _site_label(ax, row.lon, row.lat * .78, title, detail,
                    SOURCE_COLORS.get(source, GOLD), dx, dy, align)

    ax.text(.02, .02, "SITE COORDINATES: ENERGY_DATA  ·  ROUTES: SCHEMATIC",
            transform=ax.transAxes, color=MUTED, fontsize=7.5, weight="bold",
            va="bottom", alpha=.82)
    ax.text(.018, .978, "RESOURCE & TRANSPORT NETWORK", transform=ax.transAxes,
            color=INK, fontsize=9.5, weight="bold", va="top",
            bbox={"boxstyle": "round,pad=.42", "fc": PANEL, "ec": "#29465b", "lw": .8})


def _dashboard(ax, sites, records, allocations, models, total_cost, total_co2):
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def card(y, h, title, eyebrow=None):
        ax.add_patch(Rectangle((.025, y), .95, h, facecolor=PANEL,
                               edgecolor="#234156", linewidth=1, zorder=0))
        if eyebrow:
            ax.text(.07, y + h - .035, eyebrow.upper(), color=MUTED,
                    fontsize=7.0, weight="bold", va="top")
        ax.text(.07, y + h - (.065 if eyebrow else .035), title, color=INK,
                fontsize=10.2, weight="bold", va="top")

    ax.text(.06, .976, "FIELD CONDITIONS", color=INK, fontsize=14,
            weight="bold", va="top")
    ax.text(.06, .943, "MODEL INPUTS  /  ENERGY_DATA", color=MUTED,
            fontsize=7.5, weight="bold", va="top")

    # Four compact condition indicators, each with a vivid fill bar.
    card(.695, .205, "Weather snapshot", "Across the resource sites")
    weather = [
        ("SUN", sites.loc["bhadla", "sunlight_h"], "h/day", 10, GOLD),
        ("WIND", sites.loc["muppandal", "wind_ms"], "m/s", 12, TEAL),
        ("RAIN", sites.loc["idukki", "rain_mm"], "mm/day", 10, BLUE),
        ("CLOUD", sites.loc["bhadla", "cloud_pct"], "%", 100, "#b3c9db"),
    ]
    for i, (name, value, unit, maximum, color) in enumerate(weather):
        yy = .796 - i * .028
        ax.text(.08, yy, name, color=MUTED, fontsize=7.6, weight="bold", va="center")
        ax.add_patch(Rectangle((.34, yy - .006), .39, .012, color="#29455a", zorder=1))
        ax.add_patch(Rectangle((.34, yy - .006), .39 * min(1, value / maximum), .012,
                               color=color, zorder=2))
        ax.text(.92, yy, f"{value:g} {unit}", color=INK, fontsize=8.2,
                weight="bold", ha="right", va="center")

    # Priority chips summarize the three renewable resource assessments.
    card(.49, .195, "Renewable potential", "Weather-weighted priority")
    priority_rows = [
        ("SOLAR", "bhadla", "solar_priority", GOLD),
        ("WIND", "muppandal", "wind_priority", TEAL),
        ("HYDRO", "idukki", "hydro_priority", BLUE),
    ]
    for i, (label, site, col, color) in enumerate(priority_rows):
        yy = .57 - i * .033
        val = float(sites.loc[site, col])
        ax.text(.08, yy, label, color=color, fontsize=8, weight="bold", va="center")
        ax.add_patch(Rectangle((.34, yy - .006), .39, .012, color="#29455a", zorder=1))
        ax.add_patch(Rectangle((.34, yy - .006), .39 * val, .012, color=color, zorder=2))
        ax.text(.92, yy, f"{val:.0%}", color=INK, fontsize=8.4,
                weight="bold", ha="right", va="center")

    # Use the first listed model for the compact mix table; all selected models
    # remain summarized in the totals below.
    focus = models[0]
    card(.255, .225, "Dispatch mix", focus)
    row_by_name = {str(r.get("source", r.get("name"))): r for r in records}
    y0 = .380
    for i, name in enumerate(row_by_name):
        yy = y0 - i * .022
        cap = float(row_by_name[name].get("capacity_mw", row_by_name[name].get("capacity", 0)))
        used = float(allocations[focus].get(name, 0))
        color = SOURCE_COLORS.get(name, TEAL)
        ax.scatter([.095], [yy], s=23, color=color, zorder=3)
        ax.text(.14, yy, name.upper(), color=INK, fontsize=7.5,
                weight="bold", va="center")
        ax.text(.91, yy, f"{used:g}/{cap:g} MW", color=MUTED, fontsize=7.3,
                ha="right", va="center")

    card(.045, .19, "System pulse · demand & modeled impact")
    # Demand is inferred from the plotted allocations; the model's demand is
    # displayed from energy_data when using the built-in six-source dataset.
    demand = float(sum(allocations[focus].values()))
    ax.text(.08, .151, "DEMAND", color=MUTED, fontsize=7.2, weight="bold", va="center")
    ax.text(.92, .151, f"{demand:g} MW", color=INK, fontsize=9.5,
            weight="bold", ha="right", va="center")
    ax.text(.08, .123, "COST / h", color=MUTED, fontsize=7.2, weight="bold", va="center")
    ax.text(.92, .123, f"₹{total_cost[0]:,.1f}k", color=GOLD, fontsize=9.2,
            weight="bold", ha="right", va="center")
    ax.text(.08, .094, "CO₂ / h", color=MUTED, fontsize=7.2, weight="bold", va="center")
    ax.text(.92, .094, f"{total_co2[0]:.2f} t", color=TEAL, fontsize=9.2,
            weight="bold", ha="right", va="center")
    ax.text(.08, .018, "Illustrative planning model · not live telemetry",
            color=MUTED, fontsize=7.2, style="italic", va="bottom")


def plot_allocations(
    sources: Any,
    allocations: Mapping[str, Mapping[str, float]],
    output: str | Path | None = None,
    show: bool = False,
):
    """Render site conditions, network routes, and a dispatch dashboard.

    ``sources`` may be a sequence of dictionaries or a pandas DataFrame.
    Allocation values are MW; source records accept rich-model column names or
    simple aliases. Site geometry and weather come from ``energy_data.py``.
    """
    global matplotlib, plt, Circle, FancyArrowPatch, PathPatch, Polygon, Rectangle, Line2D, MplPath
    import matplotlib

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Polygon, Rectangle
    from matplotlib.lines import Line2D
    from matplotlib.path import Path as MplPath

    records = sources.to_dict("records") if hasattr(sources, "to_dict") else list(sources)
    if not records or not allocations:
        raise ValueError("At least one source and one allocation are required")
    names, capacities, costs, emissions, block_sizes = [], [], [], [], []
    for row in records:
        name = row.get("source", row.get("name"))
        if name is None:
            raise ValueError("Every source needs a 'source' or 'name'")
        names.append(str(name))
        capacities.append(float(row.get("capacity_mw", row.get("capacity", 0))))
        costs.append(float(row.get("cost_per_block", row.get("cost", 0))))
        emissions.append(float(row.get("co2_t_per_block", row.get("co2", 0))))
        block_sizes.append(float(row.get("mw_per_block", 10)))
    if len(set(names)) != len(names) or any(x < 0 for x in capacities):
        raise ValueError("Source names must be unique and capacities nonnegative")
    for label, allocation in allocations.items():
        if any(float(value) < 0 for value in allocation.values()):
            raise ValueError(f"Allocation '{label}' contains a negative value")
        unknown = set(allocation) - set(names)
        if unknown:
            raise ValueError(f"Allocation '{label}' has unknown sources: {sorted(unknown)}")
    models = list(allocations)
    values = np.array([[float(allocations[m].get(n, 0)) for n in names] for m in models])
    if np.any(values > np.array(capacities)[None, :] + 1e-9):
        raise ValueError("Allocation exceeds source capacity")
    block_values = values / np.array(block_sizes)[None, :]
    total_cost = block_values @ np.array(costs)
    total_co2 = block_values @ np.array(emissions)

    ed = _get_energy_data()
    sites = ed.build_sites()
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    grid = fig.add_gridspec(1, 2, width_ratios=(2.05, .88),
                            left=.045, right=.97, top=.89, bottom=.055, wspace=.045)
    map_ax = fig.add_subplot(grid[0, 0])
    dash_ax = fig.add_subplot(grid[0, 1])
    _draw_map(map_ax, ed, sites, records)
    _dashboard(dash_ax, sites, records, allocations, models, total_cost, total_co2)
    fig.text(.047, .965, "ENERGY ATLAS", color=TEAL, fontsize=9,
             weight="bold", va="top")
    fig.text(.047, .925, "Weather, resources & the routes that connect them",
             color=INK, fontsize=19, weight="bold", va="top")
    fig.text(.97, .959, "INDIA  ·  24H MODEL WINDOW", color=MUTED,
             fontsize=8, weight="bold", ha="right", va="top")
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=180, facecolor=BG, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _default_data():
    ed = _get_energy_data()
    sources = ed.build_sources()
    records = sources.to_dict("records")
    remaining = ed.DEMAND_MW
    allocation = {}
    for row in sorted(records, key=lambda r: r["cost_per_block"]):
        amount = min(float(row["capacity_mw"]), remaining)
        allocation[row["source"]] = amount
        remaining -= amount
    return records, {"Least cost": allocation}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Start the interactive local Energy Atlas dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="local interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="local web port (default: 8765)")
    parser.add_argument("--results-dir", help="directory containing per-scenario JSON solver results")
    parser.add_argument("--no-browser", action="store_true", help="serve the dashboard without opening a browser")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        from .local_dashboard import serve_dashboard
    except ImportError:
        from local_dashboard import serve_dashboard
    serve_dashboard(args.host, args.port, open_browser=not args.no_browser,
                    results_dir=args.results_dir)


if __name__ == "__main__":
    main()
