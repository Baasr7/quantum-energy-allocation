"""
Simplified Energy Resource Allocation - Classical Greedy Approach
-----------------------------------------------------------------
Goal: meet a single demand value at minimum total generation cost,
subject to per-source capacity limits.

Also includes a brute-force (exhaustive search) solver and a timing
comparison between the two approaches.

Greedy rule: sort sources by cost per MWh (cheapest first) and fill each
to capacity until demand is met. For this problem (linear cost, only
capacity bounds) greedy is provably optimal.
"""

import itertools
import time

# ----------------------------- DATASET ------------------------------------
# capacity in MW, cost in USD per MWh (illustrative values)
SOURCES = [
    {"name": "Wind",    "capacity": 350, "cost": 30},
    {"name": "Hydro",   "capacity": 250, "cost": 20},
    {"name": "Nuclear", "capacity": 400, "cost": 60},
    {"name": "Solar",   "capacity": 300, "cost": 35},
]
DEMAND_MW = 1000  # snapshot demand (MW); treated as 1 hour -> MWh
BRUTE_STEP_MW = 10  # brute-force granularity; smaller = slower but finer
GREEDY_REPEATS = 10_000  # greedy is so fast it needs averaging to be measurable


# ----------------------------- ALGORITHMS ---------------------------------
def greedy_allocate(sources, demand):
    """Fill cheapest sources first. Returns (allocation dict, unmet demand)."""
    allocation = {s["name"]: 0.0 for s in sources}
    remaining = demand
    for s in sorted(sources, key=lambda x: x["cost"]):
        if remaining <= 0:
            break
        used = min(s["capacity"], remaining)
        allocation[s["name"]] = used
        remaining -= used
    return allocation, max(remaining, 0.0)


def proportional_allocate(sources, demand):
    """Naive baseline: share demand in proportion to capacity (ignores cost)."""
    total_cap = sum(s["capacity"] for s in sources)
    if total_cap == 0:
        return {s["name"]: 0.0 for s in sources}, demand
    scale = min(1.0, demand / total_cap)
    allocation = {s["name"]: s["capacity"] * scale for s in sources}
    return allocation, max(demand - total_cap, 0.0)


def total_cost(allocation, sources):
    cost_map = {s["name"]: s["cost"] for s in sources}
    return sum(allocation[n] * cost_map[n] for n in allocation)


def brute_force_allocate(sources, demand, step=BRUTE_STEP_MW):
    """
    Exhaustive search: try EVERY combination of allocations (in multiples of
    `step` MW, from 0 up to each source's capacity) and keep the cheapest
    combination whose total equals demand.
    Search space = product of (capacity/step + 1) over all sources, so it
    grows exponentially with the number of sources and finer step sizes.
    """
    ranges = [range(0, s["capacity"] + 1, step) for s in sources]
    costs = [s["cost"] for s in sources]
    best_cost, best_combo, checked = float("inf"), None, 0
    for combo in itertools.product(*ranges):
        checked += 1
        if sum(combo) != demand:
            continue
        c = sum(a * k for a, k in zip(combo, costs))
        if c < best_cost:
            best_cost, best_combo = c, combo
    if best_combo is None:
        return {s["name"]: 0.0 for s in sources}, demand, checked
    return {s["name"]: float(a) for s, a in zip(sources, best_combo)}, 0.0, checked


def time_it(func, *args, repeats=1):
    """Return (last result, average seconds per call over `repeats` runs)."""
    start = time.perf_counter()
    result = None
    for _ in range(repeats):
        result = func(*args)
    return result, (time.perf_counter() - start) / repeats


# ----------------------------- REPORTING ----------------------------------
def print_table(title, allocation, sources, unmet):
    cap_map = {s["name"]: s["capacity"] for s in sources}
    cost_map = {s["name"]: s["cost"] for s in sources}
    print(f"\n{title}")
    print("-" * 74)
    print(f"{'Source':<10}{'Cap (MW)':>10}{'$/MWh':>8}{'Alloc (MW)':>12}"
          f"{'Util %':>9}{'Cost ($)':>12}")
    print("-" * 74)
    for s in sources:
        n = s["name"]
        a = allocation[n]
        print(f"{n:<10}{cap_map[n]:>10}{cost_map[n]:>8}{a:>12.1f}"
              f"{100 * a / cap_map[n]:>8.1f}%{a * cost_map[n]:>12,.0f}")
    print("-" * 74)
    tot = total_cost(allocation, sources)
    print(f"{'TOTAL':<10}{sum(cap_map.values()):>10}{'':>8}"
          f"{sum(allocation.values()):>12.1f}{'':>9}{tot:>12,.0f}")
    print(f"Unmet demand: {unmet:.1f} MW")


def check_feasibility(allocation, sources, demand, unmet):
    cap_ok = all(allocation[s["name"]] <= s["capacity"] + 1e-9 for s in sources)
    supplied = sum(allocation.values())
    demand_ok = abs(supplied - demand) < 1e-9 and unmet == 0
    print(f"  Capacity constraints satisfied : {'YES' if cap_ok else 'NO'}")
    print(f"  Demand met exactly             : {'YES' if demand_ok else 'NO'}")


# ----------------------------- MAIN ---------------------------------------
def main():
    total_cap = sum(s["capacity"] for s in SOURCES)
    print("=" * 74)
    print("ENERGY ALLOCATION - GREEDY vs BRUTE FORCE")
    print("=" * 74)
    print(f"Demand: {DEMAND_MW} MW | Total available capacity: {total_cap} MW")

    if DEMAND_MW > total_cap:
        print("WARNING: demand exceeds total capacity; problem is infeasible.")

    # Greedy (timed, averaged over many runs)
    (g_alloc, g_unmet), g_time = time_it(
        greedy_allocate, SOURCES, DEMAND_MW, repeats=GREEDY_REPEATS)
    print_table("GREEDY (cheapest-first) ALLOCATION", g_alloc, SOURCES, g_unmet)
    check_feasibility(g_alloc, SOURCES, DEMAND_MW, g_unmet)

    # Brute force (timed, single run)
    (b_alloc, b_unmet, checked), b_time = time_it(
        brute_force_allocate, SOURCES, DEMAND_MW, BRUTE_STEP_MW)
    print_table(f"BRUTE-FORCE ALLOCATION (step = {BRUTE_STEP_MW} MW)",
                b_alloc, SOURCES, b_unmet)
    check_feasibility(b_alloc, SOURCES, DEMAND_MW, b_unmet)
    print(f"  Combinations examined          : {checked:,}")

    # Baseline comparison
    p_alloc, p_unmet = proportional_allocate(SOURCES, DEMAND_MW)
    print_table("BASELINE: PROPORTIONAL-TO-CAPACITY ALLOCATION",
                p_alloc, SOURCES, p_unmet)

    # Cost comparison
    g_cost = total_cost(g_alloc, SOURCES)
    b_cost = total_cost(b_alloc, SOURCES)
    p_cost = total_cost(p_alloc, SOURCES)
    print("\nCOST COMPARISON")
    print("-" * 74)
    print(f"Greedy total cost       : ${g_cost:,.0f}")
    print(f"Brute-force total cost  : ${b_cost:,.0f}")
    print(f"Proportional total cost : ${p_cost:,.0f}")
    print(f"Greedy matches brute-force optimum: "
          f"{'YES' if abs(g_cost - b_cost) < 1e-6 else 'NO'}")
    if p_cost:
        saving = p_cost - g_cost
        print(f"Savings vs proportional : ${saving:,.0f} "
              f"({100 * saving / p_cost:.1f}%)")

    # Time comparison
    print("\nTIME COMPARISON")
    print("-" * 74)
    print(f"{'Approach':<14}{'Time per run':>18}{'Combinations/steps':>22}")
    print(f"{'Greedy':<14}{g_time * 1e6:>15.2f} us{len(SOURCES):>22,}")
    print(f"{'Brute force':<14}{b_time * 1e3:>15.2f} ms{checked:>22,}")
    if g_time > 0:
        print(f"Brute force is ~{b_time / g_time:,.0f}x slower than greedy "
              f"(greedy timed as average of {GREEDY_REPEATS:,} runs).")


if __name__ == "__main__":
    main()