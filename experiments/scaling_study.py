"""How long a linear scan of a diagonal objective takes as the catalogue grows.

Stretch check, separate from the core experiment. The synthetic rows are a
planted diagonal, not a physical model and not a QAOA run. The frozen
six-qubit catalogue is timed with the real exact enumeration.

Run from the repository root:

    python experiments/scaling_study.py
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from baselines import enumerate_best, exact_enumeration  # noqa: E402
from encoding import load_designs  # noqa: E402

OUTPUT_PATH = ROOT / "results" / "raw" / "scaling_study.json"
SYNTHETIC_QUBITS = (6, 8, 10, 12, 14, 16, 18)


def synthetic_instance(n_qubits, seed=2026):
    """Deterministic scores with one planted feasible optimum.

    Returns (scores, feasible, planted_index). About a quarter of the states
    are feasible, and index size//3 is given score 2 so it is the unique best.
    """
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int):
        raise TypeError("n_qubits must be an integer")
    if n_qubits < 1 or n_qubits > 20:
        raise ValueError("synthetic scaling is capped at 20 qubits")
    size = 1 << n_qubits
    state = seed & 0x7FFFFFFF
    scores = [0.0] * size
    feasible = [False] * size
    for index in range(size):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        scores[index] = state / float(0x7FFFFFFF)
        feasible[index] = (state & 255) < 64
    planted = size // 3
    scores[planted] = 2.0
    feasible[planted] = True
    return scores, feasible, planted


def median_scan_seconds(scores, feasible, repeats=5):
    """Median time of `repeats` scans, after one warmup scan."""
    enumerate_best(scores, feasible)
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        enumerate_best(scores, feasible)
        times.append(time.perf_counter() - started)
    times.sort()
    return times[len(times) // 2], times[0]


def main():
    rows = []
    for n_qubits in SYNTHETIC_QUBITS:
        scores, feasible, planted = synthetic_instance(n_qubits)
        found, _value = enumerate_best(scores, feasible)
        if found != planted:
            raise RuntimeError(f"planted optimum {planted} was not found at {n_qubits} qubits")
        median, fastest = median_scan_seconds(scores, feasible)
        rows.append({
            "n_qubits": n_qubits,
            "n_states": 1 << n_qubits,
            "median_seconds": median,
            "min_seconds": fastest,
            "planted_index": planted,
            "found_index": found,
        })
        print(
            f"synthetic {n_qubits:2d} qubits  {1 << n_qubits:8d} states"
            f"  median {median:.6f} s"
        )

    budget = getattr(config, "COST_BUDGET", None)
    designs = load_designs(budget=budget)
    exact_enumeration(designs)
    real_times = []
    real = None
    for _ in range(21):
        started = time.perf_counter()
        real = exact_enumeration(designs)
        real_times.append(time.perf_counter() - started)
    real_times.sort()
    real_row = {
        "n_qubits": 6,
        "n_states": len(designs),
        "n_feasible": real["n_feasible"],
        "best_bitstring": None if real["design"] is None else real["design"]["bitstring"],
        "median_seconds": real_times[len(real_times) // 2],
        "min_seconds": real_times[0],
        "n_queries": real["n_queries"],
        "budget_usd_m2": budget,
    }
    print(
        f"frozen catalogue  {real_row['n_states']} states"
        f"  best {real_row['best_bitstring']}"
        f"  median {real_row['median_seconds']:.6f} s"
    )
    payload = {
        "description": (
            "Timing of a linear scan over a diagonal objective. The synthetic "
            "rows plant one feasible optimum and are not a physical model. "
            "The frozen six-qubit catalogue is the exact reference used by the study."
        ),
        "synthetic": rows,
        "frozen_catalogue": real_row,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"saved {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
