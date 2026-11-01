"""Plot source allocations/capacities and model cost/emissions comparisons.

Run ``python -m src.visualize [scenario.json] [--output plot.png] [--no-show]``.
JSON input uses ``sources`` (records with source/name, capacity_mw/capacity,
optional cost_per_block/cost and co2_t_per_block/co2) and ``allocations``
(mapping model names to source->MW mappings). Without input, built-in rich
model data and a least-cost allocation are plotted.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_allocations(
    sources: Any,
    allocations: Mapping[str, Mapping[str, float]],
    output: str | Path | None = None,
    show: bool = False,
):
    """Build and optionally save/show a figure from source records and allocations.

    Sources may be a sequence of dictionaries or a pandas DataFrame. Allocation
    values are MW; source records accept rich-model column names or simple aliases.
    """
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

    fig, (ax, cost_ax, co2_ax) = plt.subplots(1, 3, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    x = np.arange(len(names))
    width = 0.8 / len(models)
    for i, (model, row) in enumerate(zip(models, values)):
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, row, width, label=model)
    ax.scatter(x, capacities, marker="_", s=350, color="#222222", label="Capacity", zorder=3)
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylabel("Power (MW)")
    ax.set_title("Allocation by source (markers = capacity)")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    pos = np.arange(len(models))
    cost_ax.barh(pos, total_cost, color="#2878a5")
    cost_ax.set_yticks(pos, models)
    cost_ax.set_xlabel("Modeled cost")
    cost_ax.set_title("Cost (Rs thousand/hour)")
    cost_ax.grid(axis="x", alpha=.2)
    cost_ax.set_axisbelow(True)
    co2_ax.barh(pos, total_co2, color="#e28e2c")
    co2_ax.set_yticks(pos, models)
    co2_ax.set_xlabel("Modeled emissions")
    co2_ax.set_title("CO2 (tonnes/hour)")
    co2_ax.grid(axis="x", alpha=.2)
    co2_ax.set_axisbelow(True)
    fig.tight_layout()
    if output is not None:
        fig.savefig(output, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _default_data():
    try:
        from .energy_data import DEMAND_MW, build_sources
    except ImportError:
        from energy_data import DEMAND_MW, build_sources
    sources = build_sources()
    records = sources.to_dict("records")
    remaining = DEMAND_MW
    allocation = {}
    for row in sorted(records, key=lambda r: r["cost_per_block"]):
        amount = min(float(row["capacity_mw"]), remaining)
        allocation[row["source"]] = amount
        remaining -= amount
    return records, {"Least cost": allocation}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", help="JSON scenario; defaults to rich energy data")
    parser.add_argument("--output", default="allocation_comparison.png")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args(argv)
    if args.input:
        with open(args.input, encoding="utf-8") as stream:
            payload = json.load(stream)
        sources = payload["sources"]
        allocations = payload.get("allocations")
        if allocations is None:
            demand = float(payload["demand_mw"])
            allocation, remaining = {}, demand
            for row in sorted(sources, key=lambda item: item.get("cost_per_block", item.get("cost", 0))):
                name = row.get("source", row.get("name"))
                amount = min(float(row.get("capacity_mw", row.get("capacity", 0))), remaining)
                allocation[name] = amount
                remaining -= amount
            if remaining > 0:
                raise ValueError("Scenario demand exceeds total source capacity")
            allocations = {"Least-cost": allocation}
    else:
        sources, allocations = _default_data()
    plot_allocations(sources, allocations, args.output, show=not args.no_show)
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
