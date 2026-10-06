"""hamiltonian.py - turn the score table into a QAOA cost Hamiltonian.

Encoding (agreed by the team): 6 qubits = 6 bits
    bits [0:2] material, bits [2:4] nanostructure, bits [4:6] passivation
The bitstring key "b0 b1 b2 b3 b4 b5" is the SAME string Qiskit prints in counts
(leftmost character = highest qubit, q5), and index i = int(key, 2) has qubit k
equal to bit k of i. So everything below is indexed by i = int(bitstring, 2).

We MINIMISE   C(x) = -s(x) + penalty * [x infeasible]
where s(x) is the physics score min-max normalised to [0, 1] over FEASIBLE designs.
The penalty makes the 16 invalid bit patterns (passivation "11") expensive.
Because C is diagonal in the computational basis it is a sum of Z-strings:
    C = sum_S c_S * prod_{k in S} Z_k
and c_S comes from a Walsh-Hadamard transform of the 64 cost values.
"""
import json
from pathlib import Path

import numpy as np
from qiskit.quantum_info import SparsePauliOp

ROOT = Path(__file__).resolve().parent.parent


def load_scores(path=None):
    """Read data/scores.json -> (score, feasible) arrays indexed by int(bitstring, 2)."""
    path = Path(path) if path else ROOT / "data" / "scores.json"
    with open(path) as f:
        raw = json.load(f)
    n_qubits = len(next(iter(raw)))
    size = 2 ** n_qubits
    if len(raw) != size:
        raise ValueError("scores.json must contain all %d bitstrings, found %d" % (size, len(raw)))
    score = np.zeros(size)
    feasible = np.zeros(size, dtype=bool)
    for key, v in raw.items():
        i = int(key, 2)
        score[i] = float(v["score"])
        feasible[i] = bool(v["feasible"])
    return score, feasible


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


if __name__ == "__main__":
    score, feasible = load_scores()
    H, cost = cost_hamiltonian(score, feasible, penalty=5.0)
    diag = np.real(np.diag(H.to_matrix()))
    print("Pauli terms:", len(H), "| max |H diag - cost| =", float(np.abs(diag - cost).max()))
    print("Best feasible design:", format(int(np.argmin(cost)), "06b"))