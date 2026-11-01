"""Pedagogical QAOA statevector simulator (classical simulation, no quantum advantage)."""
from __future__ import annotations

from typing import Any

import numpy as np

try:  # Support both ``import src.qaoa_solver`` and running from the src directory.
    from . import qubo_builder as qb
except ImportError:  # pragma: no cover - depends on invocation style
    import qubo_builder as qb


def _energies(qubo: qb.Qubo) -> np.ndarray:
    if not isinstance(qubo.n, (int, np.integer)) or qubo.n < 1:
        raise ValueError("QUBO must have at least one qubit")
    if np.shape(qubo.Q) != (qubo.n, qubo.n):
        raise ValueError(f"Q shape must be ({qubo.n}, {qubo.n})")
    if np.shape(qubo.costs) != (qubo.n,) or len(qubo.owners) != qubo.n:
        raise ValueError("QUBO costs and owners must match its qubit count")
    patterns = qb.all_patterns(qubo.n)
    return qb.score(qubo, patterns)


def _state(energies: np.ndarray, n: int, angles: np.ndarray) -> np.ndarray:
    """Apply alternating diagonal cost and X-mixer unitaries; bit i is qubit i."""
    gammas, betas = angles[:len(angles) // 2], angles[len(angles) // 2:]
    state = np.full(1 << n, 1 / np.sqrt(1 << n), dtype=complex)
    for gamma, beta in zip(gammas, betas):
        state *= np.exp(-1j * gamma * energies)
        c, s = np.cos(beta), -1j * np.sin(beta)
        for bit in range(n):
            stride = 1 << bit
            view = state.reshape(-1, stride * 2)
            zero, one = view[:, :stride].copy(), view[:, stride:].copy()
            view[:, :stride] = c * zero + s * one
            view[:, stride:] = s * zero + c * one
    return state


def solve_qaoa(
    qubo: qb.Qubo,
    p: int = 1,
    *,
    shots: int = 1024,
    seed: int = 0,
    maxiter: int = 100,
) -> dict[str, Any]:
    """Optimize QAOA angles and sample bitstrings. Returns best observed sample and diagnostics.

    Returned bitstrings use QUBO column order (bit i is qubit i), not Qiskit's
    displayed reversed order. If SciPy is absent, a seeded deterministic search is used.
    """
    if not isinstance(p, (int, np.integer)) or p <= 0:
        raise ValueError("p must be a positive integer")
    if not isinstance(shots, (int, np.integer)) or shots <= 0:
        raise ValueError("shots must be a positive integer")
    energies = _energies(qubo)
    rng = np.random.default_rng(seed)
    dimension = 2 * p

    def objective(theta: np.ndarray) -> float:
        state = _state(energies, qubo.n, theta)
        return float(np.dot(np.abs(state) ** 2, energies))

    method = "deterministic parameter search"
    try:
        from scipy.optimize import minimize
        initial = rng.uniform(-np.pi, np.pi, dimension)
        result = minimize(objective, initial, method="COBYLA", options={"maxiter": maxiter})
        angles, optimized_expectation = np.asarray(result.x), float(result.fun)
        method = "SciPy COBYLA"
        optimizer_success = bool(result.success)
        iterations = int(getattr(result, "nfev", 0))
    except ImportError:
        # Fixed, reproducible low-discrepancy-like uniform candidates, including zero angles.
        candidates = np.vstack([np.zeros(dimension), rng.uniform(-np.pi, np.pi, (max(1, maxiter - 1), dimension))])
        values = np.array([objective(candidate) for candidate in candidates])
        best = int(np.argmin(values))
        angles, optimized_expectation = candidates[best], float(values[best])
        optimizer_success, iterations = True, len(candidates)

    probabilities = np.abs(_state(energies, qubo.n, angles)) ** 2
    probabilities /= probabilities.sum()
    counts = rng.multinomial(shots, probabilities)
    sampled = np.flatnonzero(counts)
    best_index = int(sampled[np.argmin(energies[sampled])])
    bitstring = "".join(str(int(bit)) for bit in qb.all_patterns(qubo.n)[best_index])
    bits = qb.all_patterns(qubo.n)[best_index]
    allocation = qb.describe(qubo, bits)
    return {
        "bitstring": bitstring,
        "allocation": allocation,
        "energy": float(energies[best_index]),
        "angles": angles.tolist(),
        "diagnostics": {
            "simulation": "classical statevector; no quantum advantage",
            "p": int(p), "shots": int(shots), "seed": seed,
            "optimizer": method, "optimizer_success": optimizer_success,
            "iterations": iterations, "expected_energy": optimized_expectation,
            "sampled_unique": int(len(sampled)),
        },
    }
