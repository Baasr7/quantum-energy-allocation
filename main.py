"""Classical energy-allocation demo using the project's weather-aware source model."""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import energy_data as ed  # noqa: E402
import qubo_builder as qb  # noqa: E402


def allocate(sources, demand_blocks: int, exact_limit: int = 24):
    """Return merit-order and exact fixed-block allocations."""
    switches = ed.build_switches(sources)
    if demand_blocks > len(switches):
        raise ValueError(f"Infeasible demand: {demand_blocks} blocks, capacity is {len(switches)} blocks")
    greedy = [0] * len(switches)
    for index in sorted(range(len(switches)), key=lambda i: switches.iloc[i].cost):
        if sum(greedy) == demand_blocks:
            break
        greedy[index] = 1

    if len(switches) > exact_limit:
        raise ValueError(f"Exact comparison limited to {exact_limit} switches; this instance has {len(switches)}")
    q = qb.build_qubo(sources, demand_mw=demand_blocks * ed.MW_PER_BLOCK)
    best, best_cost = None, float("inf")
    for selected in itertools.combinations(range(q.n), demand_blocks):
        candidate = [0] * q.n
        for index in selected:
            candidate[index] = 1
        cost = float(qb.real_cost(q, candidate)[0])
        if cost < best_cost:
            best, best_cost = candidate, cost
    return q, greedy, best


def report(label, q, allocation, sources, demand_mw):
    counts = qb.describe(q, allocation)
    cost = float(qb.real_cost(q, allocation)[0])
    co2 = float(qb.real_co2(q, allocation)[0])
    total = sum(counts.values()) * ed.MW_PER_BLOCK
    valid = total == demand_mw and all(counts.get(row.source, 0) <= row.blocks_available
                                       for row in sources.itertuples())
    print(f"\n{label}: {'FEASIBLE' if valid else 'INFEASIBLE'}")
    print(f"{'Source':<10}{'Capacity MW':>13}{'Allocated MW':>14}")
    for row in sources.itertuples():
        print(f"{row.source:<10}{row.capacity_mw:>13.0f}{counts.get(row.source, 0) * ed.MW_PER_BLOCK:>14.0f}")
    print(f"Total cost: Rs {cost:,.2f} thousand/block-hour | CO2: {co2:.3f} t/block-hour")
    print(f"Demand: {demand_mw} MW | allocated: {total} MW | total capacity: {sources.capacity_mw.sum():.0f} MW")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demand", type=int, default=ed.DEMAND_MW, help="Demand MW (positive multiple of 10)")
    args = parser.parse_args()
    sources = ed.build_sources()
    demand_blocks = ed.validate(sources, args.demand)
    q, greedy, optimum = allocate(sources, demand_blocks)
    print(f"Weather-aware energy allocation | demand={args.demand} MW | switches={q.n}")
    report("Merit-order greedy", q, greedy, sources, args.demand)
    report("Exact fixed-block optimum", q, optimum, sources, args.demand)
    print("\nExact search enumerates every fixed-cardinality switch selection; QAOA integration can follow when its solver interface is ready.")


if __name__ == "__main__":
    main()
