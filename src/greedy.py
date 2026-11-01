from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Sequence
from typing import Any

Source = dict[str, Any]
SOURCES: list[Source] = [
    {"name": "Wind", "capacity": 350, "cost": 30},
    {"name": "Hydro", "capacity": 250, "cost": 20},
    {"name": "Nuclear", "capacity": 400, "cost": 60},
    {"name": "Solar", "capacity": 300, "cost": 35},
]
DEMAND_MW = 1000
BRUTE_STEP_MW = 10
GREEDY_REPEATS = 10_000


def validate_problem(sources: Sequence[Source], demand: float) -> None:
    if demand < 0:
        raise ValueError("Demand must be nonnegative.")
    names = [source["name"] for source in sources]
    if len(names) != len(set(names)):
        raise ValueError("Source names must be unique.")
    for source in sources:
        if source["capacity"] < 0:
            raise ValueError(f"Capacity for {source['name']} must be nonnegative.")
        if source["cost"] < 0:
            raise ValueError(f"Cost for {source['name']} must be nonnegative.")


def greedy_allocate(sources: Sequence[Source], demand: float) -> tuple[dict[str, float], float]:
    validate_problem(sources, demand)
    allocation = {source["name"]: 0.0 for source in sources}
    remaining = demand
    for source in sorted(sources, key=lambda item: item["cost"]):
        if remaining <= 0:
            break
        amount = min(source["capacity"], remaining)
        allocation[source["name"]] = float(amount)
        remaining -= amount
    return allocation, max(remaining, 0.0)


def proportional_allocate(sources: Sequence[Source], demand: float) -> tuple[dict[str, float], float]:
    validate_problem(sources, demand)
    total_capacity = sum(source["capacity"] for source in sources)
    if total_capacity == 0:
        return {source["name"]: 0.0 for source in sources}, demand
    scale = min(1.0, demand / total_capacity)
    allocation = {
        source["name"]: source["capacity"] * scale for source in sources
    }
    return allocation, max(demand - total_capacity, 0.0)


def total_cost(allocation: dict[str, float], sources: Sequence[Source]) -> float:
    cost_by_name = {source["name"]: source["cost"] for source in sources}
    return sum(amount * cost_by_name[name] for name, amount in allocation.items())


def brute_force_allocate(
    sources: Sequence[Source], demand: int, step: int = BRUTE_STEP_MW
) -> tuple[dict[str, float], float, int]:
    validate_problem(sources, demand)
    if step <= 0:
        raise ValueError("Search step must be a positive integer.")
    if int(demand) != demand or any(int(source["capacity"]) != source["capacity"] for source in sources):
        raise ValueError("Brute-force demand and capacities must be whole numbers.")
    if any(source["capacity"] % step for source in sources) or demand % step:
        raise ValueError("Search step must divide every capacity and the demand.")

    ranges = [range(0, source["capacity"] + 1, step) for source in sources]
    costs = [source["cost"] for source in sources]
    best_cost = float("inf")
    best_combo = None
    checked = 0
    for combination in itertools.product(*ranges):
        checked += 1
        if sum(combination) != demand:
            continue
        cost = sum(amount * unit_cost for amount, unit_cost in zip(combination, costs))
        if cost < best_cost:
            best_cost, best_combo = cost, combination
    if best_combo is None:
        return {source["name"]: 0.0 for source in sources}, float(demand), checked
    allocation = {
        source["name"]: float(amount)
        for source, amount in zip(sources, best_combo)
    }
    return allocation, 0.0, checked


def time_it(
    function: Callable[..., Any], *args: Any, repeats: int = 1
) -> tuple[Any, float]:
    if repeats <= 0:
        raise ValueError("Repeat count must be positive.")
    start = time.perf_counter()
    result = None
    for _ in range(repeats):
        result = function(*args)
    return result, (time.perf_counter() - start) / repeats


def print_table(
    title: str, allocation: dict[str, float], sources: Sequence[Source], unmet: float
) -> None:
    capacity_by_name = {source["name"]: source["capacity"] for source in sources}
    cost_by_name = {source["name"]: source["cost"] for source in sources}
    print(f"\n{title}")
    print("-" * 74)
    print(f"{'Source':<10}{'Cap (MW)':>10}{'$/MWh':>8}{'Alloc (MW)':>12}{'Util %':>9}{'Cost ($)':>12}")
    print("-" * 74)
    for source in sources:
        name = source["name"]
        capacity = capacity_by_name[name]
        amount = allocation[name]
        utilization = 100 * amount / capacity if capacity else 0.0
        print(
            f"{name:<10}{capacity:>10}{cost_by_name[name]:>8}"
            f"{amount:>12.1f}{utilization:>8.1f}%{amount * cost_by_name[name]:>12,.0f}"
        )
    print("-" * 74)
    print(
        f"{'TOTAL':<10}{sum(capacity_by_name.values()):>10}{'':>8}"
        f"{sum(allocation.values()):>12.1f}{'':>9}{total_cost(allocation, sources):>12,.0f}"
    )
    print(f"Unmet demand: {unmet:.1f} MW")


def check_feasibility(
    allocation: dict[str, float], sources: Sequence[Source], demand: float, unmet: float
) -> None:
    capacity_ok = all(
        0 <= allocation[source["name"]] <= source["capacity"] + 1e-9
        for source in sources
    )
    demand_ok = abs(sum(allocation.values()) + unmet - demand) < 1e-9
    print(f"  Capacity constraints satisfied : {'YES' if capacity_ok else 'NO'}")
    print(f"  Demand accounted for           : {'YES' if demand_ok else 'NO'}")


def main() -> None:
    validate_problem(SOURCES, DEMAND_MW)
    total_capacity = sum(source["capacity"] for source in SOURCES)
    print("=" * 74)
    print("ENERGY ALLOCATION: GREEDY AND BRUTE FORCE")
    print("=" * 74)
    print(f"Demand: {DEMAND_MW} MW | Total available capacity: {total_capacity} MW")
    if DEMAND_MW > total_capacity:
        print("Warning: demand exceeds total capacity; the problem is infeasible.")

    (greedy_allocation, greedy_unmet), greedy_time = time_it(
        greedy_allocate, SOURCES, DEMAND_MW, repeats=GREEDY_REPEATS
    )
    print_table("GREEDY ALLOCATION", greedy_allocation, SOURCES, greedy_unmet)
    check_feasibility(greedy_allocation, SOURCES, DEMAND_MW, greedy_unmet)

    (brute_allocation, brute_unmet, checked), brute_time = time_it(
        brute_force_allocate, SOURCES, DEMAND_MW, BRUTE_STEP_MW
    )
    print_table(
        f"BRUTE-FORCE ALLOCATION (step = {BRUTE_STEP_MW} MW)",
        brute_allocation,
        SOURCES,
        brute_unmet,
    )
    check_feasibility(brute_allocation, SOURCES, DEMAND_MW, brute_unmet)
    print(f"  Combinations examined          : {checked:,}")

    proportional_allocation, proportional_unmet = proportional_allocate(SOURCES, DEMAND_MW)
    print_table(
        "PROPORTIONAL-TO-CAPACITY BASELINE",
        proportional_allocation,
        SOURCES,
        proportional_unmet,
    )

    greedy_cost = total_cost(greedy_allocation, SOURCES)
    brute_cost = total_cost(brute_allocation, SOURCES)
    proportional_cost = total_cost(proportional_allocation, SOURCES)
    print("\nCOST COMPARISON")
    print("-" * 74)
    print(f"Greedy total cost       : ${greedy_cost:,.0f}")
    print(f"Brute-force total cost  : ${brute_cost:,.0f}")
    print(f"Proportional total cost : ${proportional_cost:,.0f}")
    print(f"Greedy matches brute-force optimum: {'YES' if abs(greedy_cost - brute_cost) < 1e-6 else 'NO'}")
    if proportional_cost:
        savings = proportional_cost - greedy_cost
        print(f"Savings vs proportional : ${savings:,.0f} ({100 * savings / proportional_cost:.1f}%)")

    print("\nTIME COMPARISON")
    print("-" * 74)
    print(f"{'Approach':<14}{'Time per run':>18}{'Combinations/steps':>22}")
    print(f"{'Greedy':<14}{greedy_time * 1e6:>15.2f} us{len(SOURCES):>22,}")
    print(f"{'Brute force':<14}{brute_time * 1e3:>15.2f} ms{checked:>22,}")
    if greedy_time > 0:
        print(f"Brute force is approximately {brute_time / greedy_time:,.0f}x slower than greedy.")


if __name__ == "__main__":
    main()
