"""hamiltonian.py - turn the physics score table into a QAOA cost Hamiltonian.

Encoding (agreed by the team): 6 qubits = 6 bits
    bits [0:2] material, bits [2:4] geometry/nanostructure, bits [4:6] passivation
The bitstring key "b0 b1 b2 b3 b4 b5" is the SAME string Qiskit prints in counts
(leftmost character = highest qubit, q5), and index i = int(key, 2) has qubit k
equal to bit k of i. So everything below is indexed by i = int(bitstring, 2).
NO bit reversal is needed anywhere: look Qiskit's counts keys up in scores.json as-is.

Objective (what QAOA minimises):
    C(x) = -s(x) + penalty * [x infeasible]
  s(x)        = efficiency from physics.py, min-max normalised to [0, 1] over FEASIBLE designs
  infeasible  = process-temperature clash (physics.py says so)  OR  cost > COST_BUDGET (if set)
The penalty makes forbidden bit patterns expensive so they do not win.

Because C is diagonal in the computational basis it is a sum of Z-strings:
    C = sum_S c_S * prod_{k in S} Z_k
and c_S comes from a Walsh-Hadamard transform of the 64 cost values.

scores.json may be either
  * physics.py output: {"meta": {...}, "designs": {bitstring: {"efficiency", "cost", "feasible", ...}}}
  * the old placeholder: {bitstring: {"score", "feasible"}}
"""
import json
from pathlib import Path

import numpy as np
from qiskit.quantum_info import SparsePauliOp

ROOT = Path(__file__).resolve().parent.parent


def _read(path=None):
    path = Path(path) if path else ROOT / "data" / "scores.json"
    with open(path) as f:
        return json.load(f)


def load_scores(path=None, budget=None):
    """Read data/scores.json -> (score, feasible) arrays indexed by int(bitstring, 2).

    budget: optional cost cap in USD/m2. Designs above it are marked infeasible
            (only possible with physics.py output, which has a cost column).
    """
    raw = _read(path)
    rows = raw["designs"] if "designs" in raw else raw
    n_qubits = len(next(iter(rows)))
    size = 2 ** n_qubits
    if len(rows) != size:
        raise ValueError("scores.json must contain all %d bitstrings, found %d" % (size, len(rows)))
    score = np.zeros(size)
    feasible = np.zeros(size, dtype=bool)
    for key, v in rows.items():
        i = int(key, 2)
        if "efficiency" in v:                       # physics.py format
            score[i] = float(v["efficiency"])
            feasible[i] = bool(v["feasible"]) and (budget is None or float(v["cost"]) <= budget + 1e-9)
        else:                                       # old placeholder format
            score[i] = float(v["score"])
            feasible[i] = bool(v["feasible"])
    return score, feasible


def design_labels(path=None):
    """Human-readable name for each index, e.g. 'c-Si / Micro-pyramids / Al2O3 (ALD)'."""
    raw = _read(path)
    rows = raw["designs"] if "designs" in raw else raw
    n_qubits = len(next(iter(rows)))
    labels = [format(i, "0%db" % n_qubits) for i in range(2 ** n_qubits)]
    for key, v in rows.items():
        if "material" in v:
            labels[int(key, 2)] = "%s / %s / %s" % (v["material"], v["geometry"], v["passivation"])
    return labels


def normalise_scores(score, feasible):
    """Min-max normalise to [0, 1] using feasible designs only (so the best feasible = 1)."""
    lo, hi = score[feasible].min(), score[feasible].max()
    if hi == lo:
        raise ValueError("All feasible scores are identical - nothing to optimise.")
    return (score - lo) / (hi - lo)


def cost_vector(score, feasible, penalty):
    """Diagonal of the cost Hamiltonian: lower is better."""
    s = normalise_scores(score, feasible)
    return -s + penalty * (~feasible).astype(float)


def diag_to_pauli(cost):
    """Exact Z-string decomposition of a diagonal operator given as a length-2^n vector."""
    size = len(cost)
    n = int(round(np.log2(size)))
    idx = np.arange(size)
    terms = []
    for subset in range(size):
        parity = np.array([bin(int(x) & subset).count("1") % 2 for x in idx])
        coeff = float(np.dot(cost, 1 - 2 * parity) / size)
        if abs(coeff) > 1e-12:
            label = "".join("Z" if (subset >> k) & 1 else "I" for k in reversed(range(n)))
            terms.append((label, coeff))
    return SparsePauliOp.from_list(terms)


def cost_hamiltonian(score, feasible, penalty):
    """Return (SparsePauliOp H, cost vector). H's diagonal equals the cost vector exactly."""
    cost = cost_vector(score, feasible, penalty)
    return diag_to_pauli(cost), cost


def truncate_hamiltonian(H, max_weight):
    """Keep only Z-strings with at most `max_weight` Z's (2 = pairwise, QUBO-like).

    A k-body term costs many two-qubit gates on hardware, so dropping the high-order terms
    makes the circuit far shallower - at the price of an APPROXIMATE cost function.
    Always judge the result against the true scores (summarise() does this).
    """
    keep = [(lab, c) for lab, c in zip(H.paulis.to_labels(), H.coeffs) if lab.count("Z") <= max_weight]
    return SparsePauliOp.from_list(keep)


def hamiltonian_diag(H):
    """The 2^n costs a Hamiltonian actually implements (for the numpy simulator)."""
    return np.real(np.diag(H.to_matrix()))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    try:
        import config
        budget = getattr(config, "COST_BUDGET", None)
    except ImportError:
        budget = None
    score, feasible = load_scores(budget=budget)
    labels = design_labels()
    H, cost = cost_hamiltonian(score, feasible, penalty=5.0)
    diag = np.real(np.diag(H.to_matrix()))
    print("Cost budget:", budget, "| feasible designs:", int(feasible.sum()), "of", len(feasible))
    print("Pauli terms:", len(H), "| max |H diag - cost| =", float(np.abs(diag - cost).max()))
    best = int(np.argmin(cost))
    print("Exact best feasible design:", format(best, "06b"), "->", labels[best])