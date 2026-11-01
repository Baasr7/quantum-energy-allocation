# Energy allocation CLI demo

A small hybrid-style demonstration built on `src/energy_data.py` and `src/qubo_builder.py`. The source model derives weather-adjusted capacities and per-block costs from illustrative site, transport, generation, carbon, and resource assumptions. A pedagogical QAOA statevector simulation is included alongside classical baselines; it runs on a classical computer and does not demonstrate quantum advantage.

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
python main.py --demand 80
python main.py --plot allocation.png
python -m src.visualize data/small.json --no-show --output small.png
```

Demand is in MW and must be a positive multiple of 10 MW. The default is 100 MW. The demo prints per-source capacities and allocations, total modeled cost, CO2, and demand/capacity feasibility for both a cheapest-first merit-order baseline and an exact optimum.

## Method and assumptions

One switch represents a 10 MW dispatch block; resource limits are estimated over a 24-hour window. Both classical methods meet demand using exactly the requested number of switches. Merit-order greedily selects the cheapest available blocks; exact search enumerates all fixed-cardinality selections and minimizes per-block cost. Exact enumeration is intentionally capped at 24 switches to keep this a small demo. The report totals model cost in Rs thousand/hour and CO2 in tonnes/hour by summing per-block-hour values. The capacity report shows each source's weather/resource-limited available capacity. The QAOA simulator uses one qubit per available 10 MW block, a penalty QUBO for unmet demand, a statevector simulator, and a small classical parameter search. Its sampled candidate is reported with a feasibility check and compared with the exact result. For a compact figure of allocations and cost/emissions, run the visualization command above.

This is a classical simulation demonstration, not a market dispatch forecast or claim of quantum speedup. Model weights and source data are illustrative. The JSON examples can be visualized directly (for example, `python -m src.visualize data/small.json --no-show`). They are not consumed by the allocation CLI. Exact enumeration is capped at 24 switches; statevector simulation scales exponentially and is intended only for small instances.
