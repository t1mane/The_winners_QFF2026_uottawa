"""Six-qubit encoding for the frozen QNanoVolt catalogue.

Bit layout, matching data/catalogue.json, src/physics.py and src/hamiltonian.py:

    bitstring = material(2) | geometry(2) | passivation(2), most significant bit first
    state index s = int(bitstring, 2) = 16*material + 4*geometry + layer
    qubit q holds bit q of s

A Qiskit count string is already in that order: the leftmost character is qubit 5
and the rightmost character is qubit 0. Look those strings up with int(key, 2).

The frozen catalogue has four passivation options, including "None", so every
index 0..63 is a real design. Feasibility is separate from that: a design is
feasible when the passivation process is compatible with the absorber and the
module cost is within the budget.

Cost scales
    feasible_minmax   r = (eta - min feasible eta) / (max feasible eta - min)
                      This is the scale src/hamiltonian.py trains on. The best
                      feasible design has r = 1. An over-budget design can sit
                      slightly above 1.
    catalogue_max     r = eta / max(eta over all 64 designs), so r is in [0, 1].
                      Used to check the penalty theorem. It is not the training scale.
"""

import json
import numbers
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCORES_PATH = ROOT / "data" / "scores.json"

N_QUBITS = 6
N_STATES = 1 << N_QUBITS
N_OPTIONS = 4
BUDGET_TOL = 1e-9
PENALTY_GAP_TOL = 1e-10

PUBLIC_FIELDS = (
    "index",
    "bitstring",
    "material_id",
    "geometry_id",
    "layer_id",
    "material",
    "geometry",
    "passivation",
    "efficiency",
    "cost",
    "compatible",
    "within_budget",
    "feasible",
    "category_valid",
    "budget",
)


def _as_int(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def decode(state_index):
    """Return (material_id, geometry_id, layer_id, category_valid).

    layer_id is the passivation index. category_valid is True for every
    index in 0..63 and False outside that range.
    """
    index = _as_int(state_index, "state_index")
    if index < 0 or index >= N_STATES:
        return None, None, None, False
    material_id = (index >> 4) & 0b11
    geometry_id = (index >> 2) & 0b11
    layer_id = index & 0b11
    return material_id, geometry_id, layer_id, True


def encode(material_id, geometry_id, layer_id):
    """Pack three category indices into a state index in 0..63."""
    material_id = _as_int(material_id, "material_id")
    geometry_id = _as_int(geometry_id, "geometry_id")
    layer_id = _as_int(layer_id, "layer_id")
    for name, value in (
        ("material_id", material_id),
        ("geometry_id", geometry_id),
        ("layer_id", layer_id),
    ):
        if value not in (0, 1, 2, 3):
            raise ValueError(f"{name} must be 0, 1, 2 or 3")
    return (material_id << 4) | (geometry_id << 2) | layer_id


def index_to_bitstring(state_index):
    """Six-character string, qubit 5 on the left and qubit 0 on the right."""
    index = _as_int(state_index, "state_index")
    if index < 0 or index >= N_STATES:
        raise ValueError(f"state_index must be in 0..{N_STATES - 1}")
    return format(index, "06b")


def bitstring_to_index(bitstring):
    """Inverse of index_to_bitstring. Spaces are ignored."""
    if not isinstance(bitstring, str):
        raise TypeError("bitstring must be a string")
    text = bitstring.replace(" ", "")
    if len(text) != N_QUBITS or set(text) - {"0", "1"}:
        raise ValueError(f"bitstring must be {N_QUBITS} characters of 0/1, got {bitstring!r}")
    return int(text, 2)


def qubit_bits(state_index):
    """Return six bits, index 0 being qubit 0 (the rightmost Qiskit character)."""
    index = _as_int(state_index, "state_index")
    if index < 0 or index >= N_STATES:
        raise ValueError(f"state_index must be in 0..{N_STATES - 1}")
    return [(index >> qubit) & 1 for qubit in range(N_QUBITS)]


def require_ordered(designs):
    """Return designs sorted by state index, or raise if an index is missing."""
    if not designs:
        raise ValueError("design list is empty")
    ordered = sorted(designs, key=lambda design: design["index"])
    expected = list(range(len(ordered)))
    if [design["index"] for design in ordered] != expected:
        raise ValueError(
            f"expected one design for every index 0..{len(ordered) - 1}"
        )
    if len(ordered) == N_STATES:
        for design in ordered:
            material_id, geometry_id, layer_id, valid = decode(design["index"])
            got = (
                design["material_id"],
                design["geometry_id"],
                design["layer_id"],
                design["category_valid"],
            )
            if got != (material_id, geometry_id, layer_id, valid):
                raise ValueError(f"category ids disagree with the encoding at {design['index']}")
            if design["bitstring"] != index_to_bitstring(design["index"]):
                raise ValueError(f"bitstring disagrees with the state index at {design['index']}")
    return ordered


def public_design(design):
    """JSON-ready copy of the fields other modules should rely on."""
    return {key: design[key] for key in PUBLIC_FIELDS if key in design}


def load_designs(path=None, budget=None):
    """Load the 64-row score table and apply the cost budget.

    `budget` is in USD/m2. None leaves every process-compatible design feasible.
    The score file's own "feasible" field is process compatibility only; this
    loader stores that as `compatible` and stores the budget decision separately.
    """
    path = Path(path) if path is not None else SCORES_PATH
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    rows = raw["designs"] if isinstance(raw, dict) and "designs" in raw else raw
    budget_value = None if budget is None else float(budget)
    designs = []
    for key, row in rows.items():
        index = bitstring_to_index(key)
        material_id, geometry_id, layer_id, valid = decode(index)
        if "efficiency" in row:
            efficiency = float(row["efficiency"])
            compatible = bool(row["feasible"])
            cost = float(row["cost"])
        else:
            efficiency = float(row["score"])
            compatible = bool(row["feasible"])
            cost = float(row["cost"]) if "cost" in row else 0.0
        within_budget = True if budget_value is None else cost <= budget_value + BUDGET_TOL
        feasible = bool(valid and compatible and within_budget)
        designs.append({
            "index": index,
            "bitstring": index_to_bitstring(index),
            "material_id": material_id,
            "geometry_id": geometry_id,
            "layer_id": layer_id,
            "material": row.get("material", str(material_id)),
            "geometry": row.get("geometry", str(geometry_id)),
            "passivation": row.get("passivation", str(layer_id)),
            "efficiency": efficiency,
            "cost": cost,
            "compatible": compatible,
            "within_budget": within_budget,
            "feasible": feasible,
            "category_valid": valid,
            "budget": budget_value,
        })
    ordered = require_ordered(designs)
    if len(ordered) != N_STATES:
        raise ValueError(f"expected {N_STATES} designs, found {len(ordered)}")
    return ordered


def normalized_scores(designs, scale="feasible_minmax"):
    """Return (scores aligned with index order, low bound, high bound)."""
    ordered = require_ordered(designs)
    efficiencies = [design["efficiency"] for design in ordered]
    if scale == "feasible_minmax":
        feasible_eff = [design["efficiency"] for design in ordered if design["feasible"]]
        if not feasible_eff:
            raise ValueError("no feasible design, so the feasible min-max scale is undefined")
        low, high = min(feasible_eff), max(feasible_eff)
        if high == low:
            raise ValueError("all feasible efficiencies are identical, so min-max is undefined")
        scores = [(efficiency - low) / (high - low) for efficiency in efficiencies]
        return scores, low, high
    if scale == "catalogue_max":
        high = max(efficiencies)
        if high <= 0:
            raise ValueError("catalogue efficiencies are not positive, so the unit scale is undefined")
        if min(efficiencies) < -1e-15:
            raise ValueError("a negative efficiency is outside the unit-interval model")
        scores = [efficiency / high for efficiency in efficiencies]
        return scores, 0.0, high
    raise ValueError(f"unknown score scale {scale!r}")


def build_cost_table(designs, penalty, scale="feasible_minmax"):
    """Diagonal energies E = -r + penalty * (1 if infeasible else 0).

    The returned energy still contains the identity contribution. Callers that
    compare objective values should keep it.
    """
    if isinstance(penalty, bool) or not isinstance(penalty, numbers.Real):
        raise TypeError("penalty must be a real number")
    ordered = require_ordered(designs)
    scores, low, high = normalized_scores(ordered, scale=scale)
    energy = [
        -score + (0.0 if design["feasible"] else float(penalty))
        for score, design in zip(scores, ordered)
    ]
    return {
        "energy": np.asarray(energy, dtype=float),
        "normalized_score": np.asarray(scores, dtype=float),
        "lo": float(low),
        "hi": float(high),
        "scale": scale,
        "penalty": float(penalty),
    }


def build_unit_cost_table(designs, penalty):
    """Cost table on the catalogue-max scale, where every r is in [0, 1]."""
    return build_cost_table(designs, penalty, scale="catalogue_max")


def penalty_threshold(designs, scale="feasible_minmax"):
    """Smallest M such that every infeasible energy is strictly above the best feasible energy.

    E(infeasible) = -r + M and E(best feasible) = -r*. The bound is
    max(r_infeasible - r*). A positive value means M = 0 would rank an
    infeasible design ahead of the physical optimum.
    """
    table = build_cost_table(designs, 0.0, scale=scale)
    scores = table["normalized_score"]
    ordered = require_ordered(designs)
    best_feasible = max(scores[design["index"]] for design in ordered if design["feasible"])
    infeasible_scores = [scores[design["index"]] for design in ordered if not design["feasible"]]
    if not infeasible_scores:
        return 0.0
    return float(max(score - best_feasible for score in infeasible_scores))


def infeasible_gap(energy, designs):
    """Compare the global energy minimizer with the best feasible energy.

    Ties break toward lower cost, then lower state index, matching exact enumeration.
    `all_infeasible_worse` is True when the closest infeasible energy exceeds the
    best feasible energy by more than 1e-10.
    """
    ordered = require_ordered(designs)
    values = [float(item) for item in energy]
    if len(values) != len(ordered):
        raise ValueError("energy length does not match the design table")

    def key(index):
        return (values[index], ordered[index]["cost"], index)

    global_index = min(range(len(ordered)), key=key)
    feasible_indices = [index for index, design in enumerate(ordered) if design["feasible"]]
    if not feasible_indices:
        raise ValueError("no feasible design")
    best_feasible = min(feasible_indices, key=key)
    infeasible_indices = [index for index, design in enumerate(ordered) if not design["feasible"]]
    if not infeasible_indices:
        closest = None
        gap = None
        worse = True
    else:
        closest = min(infeasible_indices, key=key)
        gap = values[closest] - values[best_feasible]
        worse = gap > PENALTY_GAP_TOL
    return {
        "global_minimizer_index": global_index,
        "global_minimizer_bitstring": ordered[global_index]["bitstring"],
        "global_minimizer_feasible": bool(ordered[global_index]["feasible"]),
        "best_feasible_energy_index": best_feasible,
        "best_feasible_energy": values[best_feasible],
        "closest_infeasible_index": closest,
        "closest_infeasible_bitstring": None if closest is None else ordered[closest]["bitstring"],
        "closest_infeasible_energy": None if closest is None else values[closest],
        "infeasible_energy_gap": gap,
        "all_infeasible_worse": worse,
    }


def audit_penalty(designs, penalty, scale="feasible_minmax"):
    """Return whether this penalty's energy minimizer is the physical optimum."""
    ordered = require_ordered(designs)
    table = build_cost_table(ordered, penalty, scale=scale)
    report = infeasible_gap(table["energy"], ordered)
    physical = max(
        (design for design in ordered if design["feasible"]),
        key=lambda design: (design["efficiency"], -design["cost"], -design["index"]),
    )
    report["matches_physical_optimum"] = report["global_minimizer_index"] == physical["index"]
    report["physical_optimum_bitstring"] = physical["bitstring"]
    report["penalty"] = float(penalty)
    report["scale"] = scale
    report["lo"] = table["lo"]
    report["hi"] = table["hi"]
    report["threshold"] = penalty_threshold(ordered, scale=scale)
    return report


if __name__ == "__main__":
    material_id, geometry_id, layer_id, valid = decode(bitstring_to_index("000111"))
    print("000111 ->", material_id, geometry_id, layer_id, "valid" if valid else "invalid")
    print("repacked", index_to_bitstring(encode(material_id, geometry_id, layer_id)))
    print("qubit 0 is the rightmost bit:", qubit_bits(bitstring_to_index("000111")))
