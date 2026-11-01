# Energy allocation CLI demo

A classical comparison built on `src/energy_data.py` and `src/qubo_builder.py`. The source model derives weather-adjusted capacities and per-block costs from its India-inspired site, transport, generation, carbon, and resource assumptions.

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
python main.py --demand 80
```

Demand is in MW and must be a positive multiple of 10 MW. The default is 100 MW. The demo prints per-source capacities and allocations, total modeled cost, CO2, and demand/capacity feasibility for both a cheapest-first merit-order baseline and an exact optimum.

## Method and assumptions

One switch represents one 10 MW block for the model's 24-hour scheduling window. Both methods meet demand using exactly the requested number of switches. Merit-order greedily selects the cheapest available blocks; exact search enumerates all fixed-cardinality selections and minimizes the rich model's per-block cost. Exact enumeration is intentionally capped at 24 switches to keep this a small demo. Cost and CO2 totals are per-block-hour quantities multiplied by selected blocks; reported CO2 is in tonnes per block-hour as represented by the model. The capacity report shows each source's weather/resource-limited available capacity.

This is a classical integration demonstration, not a market dispatch forecast. Model weights and source data are illustrative. `data/small.json` and `data/medium.json` are not consumed by this rich-model CLI. No QAOA solver is imported because its integration interface is not established; the next step is to connect a solver result to the existing QUBO and compare it against this exact baseline.
