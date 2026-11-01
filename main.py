"""Compare greedy, brute-force, and QAOA allocations from one shared CSV dataset."""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
import energy_data as ed  # noqa: E402
import qubo_builder as qb  # noqa: E402
from qaoa_solver import solve_qaoa  # noqa: E402

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "energy_scenarios.csv"


def load_scenario(path: str | Path, scenario_name: str):
    """Load one scenario's source rows; every solver receives this same table."""
    data = pd.read_csv(path)
    required = {"scenario", "demand_mw", "mw_per_block", "source", "capacity_mw",
                "cost_per_block", "co2_t_per_block"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")
    selected = data.loc[data["scenario"] == scenario_name].copy()
    if selected.empty:
        names = ", ".join(sorted(data["scenario"].dropna().unique()))
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available scenarios: {names}")
    if selected["demand_mw"].nunique() != 1 or selected["mw_per_block"].nunique() != 1:
        raise ValueError(f"Scenario '{scenario_name}' must use one demand and one block size")
    if selected[list(required - {"scenario"})].isna().any().any():
        raise ValueError(f"Scenario '{scenario_name}' has missing required values")
    if selected["source"].duplicated().any():
        raise ValueError(f"Scenario '{scenario_name}' contains duplicate source names")
    if (selected["mw_per_block"] <= 0).any():
        raise ValueError("mw_per_block must be positive")
    selected["blocks_available"] = selected["capacity_mw"] / selected["mw_per_block"]
    if (selected["blocks_available"] % 1 != 0).any():
        raise ValueError("Every source capacity must be divisible by mw_per_block")
    selected["blocks_available"] = selected["blocks_available"].astype(int)
    return selected, int(selected["demand_mw"].iloc[0]), int(selected["mw_per_block"].iloc[0])


def allocate(sources, demand_blocks: int, block_mw: int, exact_limit: int = 24):
    """Return greedy, exact fixed-block, and shared QUBO inputs."""
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
    q = qb.build_qubo(sources, demand_mw=demand_blocks * block_mw)
    best, best_cost = None, float("inf")
    for selected in itertools.combinations(range(q.n), demand_blocks):
        candidate = [0] * q.n
        for index in selected:
            candidate[index] = 1
        cost = float(qb.real_cost(q, candidate)[0])
        if cost < best_cost:
            best, best_cost = candidate, cost
    return q, greedy, best


def report(label, q, allocation, sources, demand_mw, block_mw):
    counts = qb.describe(q, allocation)
    cost = float(qb.real_cost(q, allocation)[0])
    co2 = float(qb.real_co2(q, allocation)[0])
    total = sum(counts.values()) * block_mw
    valid = total == demand_mw and all(counts.get(row.source, 0) <= row.blocks_available
                                       for row in sources.itertuples())
    print(f"\n{label}: {'FEASIBLE' if valid else 'INFEASIBLE'}")
    print(f"{'Source':<10}{'Capacity MW':>13}{'Allocated MW':>14}")
    for row in sources.itertuples():
        print(f"{row.source:<10}{row.capacity_mw:>13.0f}{counts.get(row.source, 0) * block_mw:>14.0f}")
    print(f"Total cost: {cost:,.2f} modeled units/hour | CO2: {co2:.3f} tonnes/hour")
    print(f"Demand: {demand_mw} MW | allocated: {total} MW | total capacity: {sources.capacity_mw.sum():.0f} MW")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="shared CSV input dataset")
    parser.add_argument("--scenario", default="base", help="scenario name from the CSV")
    parser.add_argument("--demand", type=int, help="override scenario demand (MW)")
    parser.add_argument("--shots", type=int, default=1024, help="QAOA simulation samples")
    parser.add_argument("--maxiter", type=int, default=100, help="QAOA parameter-search iterations")
    parser.add_argument("--plot", metavar="PATH", help="save allocation/cost/emissions comparison plot")
    args = parser.parse_args()

    sources, scenario_demand, block_mw = load_scenario(args.data, args.scenario)
    demand_mw = args.demand if args.demand is not None else scenario_demand
    demand_blocks = ed.validate(sources, demand_mw)
    q, greedy, optimum = allocate(sources, demand_blocks, block_mw)
    print(f"Energy allocation | scenario={args.scenario} | data={args.data} | demand={demand_mw} MW | qubits={q.n}")
    report("Greedy", q, greedy, sources, demand_mw, block_mw)
    report("Brute force (exact)", q, optimum, sources, demand_mw, block_mw)

    qaoa = solve_qaoa(q, p=1, shots=args.shots, maxiter=args.maxiter)
    qaoa_bits = [int(bit) for bit in qaoa["bitstring"]]
    report("QAOA statevector sample (classical simulation)", q, qaoa_bits, sources, demand_mw, block_mw)
    exact_cost = float(qb.real_cost(q, optimum)[0])
    qaoa_cost = float(qb.real_cost(q, qaoa_bits)[0])
    exact_co2 = float(qb.real_co2(q, optimum)[0])
    qaoa_co2 = float(qb.real_co2(q, qaoa_bits)[0])
    cost_gap = 100 * (qaoa_cost - exact_cost) / exact_cost if exact_cost else 0.0
    print(f"QAOA vs brute-force cost gap: {cost_gap:+.1f}% | emissions difference: {qaoa_co2 - exact_co2:+.3f} tonnes/hour")
    print(f"QAOA diagnostics: {qaoa['diagnostics']}")
    if args.plot:
        from visualize import plot_allocations

        def to_mw(bits):
            return {name: blocks * block_mw for name, blocks in qb.describe(q, bits).items()}

        plot_allocations(sources, {
            "Greedy": to_mw(greedy),
            "Brute force": to_mw(optimum),
            "QAOA sample": to_mw(qaoa_bits),
        }, output=args.plot)
        print(f"Saved comparison plot: {args.plot}")
    print("\nAll three methods use the same CSV costs, capacities, emissions, demand, and block size.")
    print("QAOA is a classical statevector simulation, not quantum hardware or evidence of quantum advantage.")


if __name__ == "__main__":
    main()
