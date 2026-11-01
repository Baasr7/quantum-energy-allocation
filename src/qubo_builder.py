"""QUBO builder for the energy allocation problem.

    Score(x) = sum_i c_i x_i + lam * (sum_i x_i - D)^2

One qubit per block (see energy_data.build_switches). Qubit 0 is the FIRST column
of every pattern array here; Qiskit bitstrings put qubit 0 on the RIGHT.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from . import energy_data as ed
except ImportError:  # support running modules directly from the src directory
    import energy_data as ed


@dataclass
class Qubo:
    n: int                  # number of qubits (= blocks)
    costs: np.ndarray       # c_i per switch
    co2: np.ndarray         # CO2 (t per block-hour) per switch
    owners: list            # source name per switch
    demand_blocks: int      # D
    lam: float              # penalty weight
    Q: np.ndarray           # upper-triangular matrix: score = x^T Q x + offset
    offset: float


def build_qubo(sources: pd.DataFrame, demand_mw: int = ed.DEMAND_MW, lam: float | None = None) -> Qubo:
    D = ed.validate(sources, demand_mw)
    sw = ed.build_switches(sources)
    co2_by_src = sources.set_index("source")["co2_t_per_block"]
    costs = sw["cost"].to_numpy(float)
    owners = sw["source"].tolist()
    co2 = np.array([co2_by_src[s] for s in owners], float)
    n = len(costs)
    if lam is None:
        lam = 1.25 * costs.max()                      # must exceed the priciest block
    Q = np.zeros((n, n))
    np.fill_diagonal(Q, costs + lam * (1 - 2 * D))    # cost_i + lam*(1-2D)
    Q[np.triu_indices(n, 1)] = 2 * lam                # every pair
    return Qubo(n, costs, co2, owners, D, float(lam), Q, float(lam * D * D))


def all_patterns(n: int) -> np.ndarray:
    """All 2^n patterns, shape (2^n, n); column i = qubit i."""
    idx = np.arange(2 ** n)
    return ((idx[:, None] >> np.arange(n)) & 1).astype(np.int8)


def score(q: Qubo, X) -> np.ndarray:
    X = np.atleast_2d(X).astype(float)
    return np.einsum("ki,ij,kj->k", X, q.Q, X) + q.offset


def real_cost(q: Qubo, X) -> np.ndarray:
    return np.atleast_2d(X) @ q.costs


def real_co2(q: Qubo, X) -> np.ndarray:
    return np.atleast_2d(X) @ q.co2


def describe(q: Qubo, x) -> dict:
    """{source: blocks used} for one pattern (sources with 0 blocks omitted)."""
    out: dict = {}
    for on, s in zip(np.asarray(x).ravel(), q.owners):
        if on:
            out[s] = out.get(s, 0) + 1
    return out


def min_valid_lambda(q: Qubo, X) -> float:
    """Smallest lambda for which no invalid pattern beats the best valid one."""
    rc, on = real_cost(q, X), np.asarray(X).sum(1)
    bad = on != q.demand_blocks
    best_valid = rc[~bad].min()
    need = (best_valid - rc[bad]) / (on[bad] - q.demand_blocks) ** 2
    return float(max(0.0, need.max()))
