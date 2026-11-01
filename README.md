# Quantum Energy Allocation

A small-scale demonstration of how a quantum optimisation algorithm (QAOA) can be applied to an **energy resource allocation problem**, compared against classical approaches (brute force and greedy).

Built for the *Quantum simulation for energy systems* hackathon challenge.

---

## The Problem

Energy systems must balance generation costs, resource availability, capacity constraints and electricity demand. Given several energy sources, each with a cost per MWh and a maximum capacity, the goal is to:

> **Meet the electricity demand at the minimum total generation cost, without exceeding any source's capacity.**

## Approach

The same scenarios (loaded from shared data files) are solved with four methods so the results can be compared fairly.

| Method | Type | Idea |
|---|---|---|
| **Brute force** | Classical | Tries every allocation combination and keeps the cheapest valid one. Guaranteed optimal, but the search space grows exponentially. |
| **Greedy (merit order)** | Classical | Sorts sources by cost and fills the cheapest first until demand is met. Very fast; optimal when costs are linear and only capacity limits apply. |
| **QUBO formulation** | Problem encoding | Rewrites the allocation as a Quadratic Unconstrained Binary Optimisation problem: each source's output is encoded in binary bits, total cost becomes the objective, and the demand requirement becomes a penalty term. |
| **QAOA** | Hybrid quantum-classical | Solves the QUBO with the Quantum Approximate Optimisation Algorithm, simulated with Qiskit Aer. A classical optimiser tunes the circuit angles. |

### How the formulation works

1. **Encode** each source's output level as a set of binary variables (this also enforces capacity).
2. **Cost term:** total cost = sum of (cost per MWh × output).
3. **Demand constraint:** add a penalty `λ × (total output − demand)²`, so any allocation that misses demand is expensive.
4. The lowest-energy bit string of the resulting QUBO is the cheapest feasible allocation.
5. Convert the QUBO to an Ising Hamiltonian, build the QAOA circuit, run it on the Aer simulator, and decode the most likely feasible bit string back into MW per source.

## Project Structure

```
quantum-energy-allocation/
├── README.md
├── main.py                 # entry point: runs the solvers and the comparison
├── requirements.txt        # Python dependencies
├── pip_freeze.log          # exact package versions used (reproducibility)
├── mise.toml               # tool/runtime version configuration (mise)
├── .gitignore
│
├── data/
│   ├── energy_scenarios.csv  # shared scenarios used by all solvers
│   ├── small.json            # small demo instance
│   └── medium.json           # larger instance
│
├── src/
│   ├── __init__.py
│   ├── energy_data.py        # loads and validates scenario data
│   ├── brute_force.py        # classical exhaustive-search solver
│   ├── greedy.py             # classical merit-order solver
│   ├── qubo.py               # QUBO formulation
│   ├── qubo_builder.py       # builds the QUBO matrix used by QAOA
│   ├── qaoa_solver.py        # QAOA on the Qiskit Aer simulator
│   ├── show_steps.py         # prints the solution steps for walkthroughs
│   └── visualize.py          # matplotlib comparison plots
│
└── results/
    └── figures/              # generated plots
```

## Getting Started

### Prerequisites

- Python 3.10 – 3.12 (recommended for Qiskit)
- `pip`

### Installation

```bash
git clone https://github.com/<your-username>/quantum-energy-allocation.git
cd quantum-energy-allocation

python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# macOS / Linux:         source .venv/bin/activate

pip install -r requirements.txt
```

The exact versions used during development are recorded in `pip_freeze.log`.

### Run

```bash
python main.py
```

This loads the scenarios from `data/`, runs the brute-force, greedy and QAOA solvers on the same inputs, prints a cost and feasibility comparison, and saves plots to `results/figures/`.

## Results

> Replace this section with your own numbers after running `main.py`.

| Method | Allocation | Total cost | Feasible | Runtime |
|---|---|---|---|---|
| Brute force | | | | |
| Greedy | | | | |
| QAOA | | | | |

Figures are saved in `results/figures/`:

```markdown
![Comparison](results/figures/<your-figure-name>.png)
```

## Key Takeaways

- **Feasibility:** each method is checked against the same demand and capacity rules.
- **Optimality:** brute force gives the true optimum, so the other methods are measured against it.
- **Scaling:** brute-force runtime grows exponentially with the number of sources and the output resolution, while greedy stays near-instant.
- **Why QUBO:** the QUBO form extends naturally to harder constraints (minimum output levels, start-up costs, ramp limits) where simple greedy rules stop being optimal.

## Limitations

- QAOA is run on a **classical simulator** (Qiskit Aer), so it demonstrates the method, not a quantum speed-up.
- Instances are kept small (limited number of qubits) so they can be simulated quickly.
- For this simplified problem (linear costs, capacity limits only), greedy is already provably optimal; the quantum approach is shown as a proof of concept for richer problems.
- QAOA is approximate and may return suboptimal or infeasible bit strings depending on depth, penalty weight and optimiser settings.

## Tech Stack

- Python
- Qiskit and Qiskit Aer (quantum circuit simulation)
- NumPy
- Matplotlib
