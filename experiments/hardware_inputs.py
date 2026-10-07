"""
hardware_inputs.py

Everything a hardware or noisy-simulator run needs,
rebuilt from the saved QAOA sweep.
"""

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


import config

from hamiltonian import (
    load_scores,
    design_labels,
    cost_hamiltonian,
    truncate_hamiltonian,
)

from qaoa import (
    build_circuit,
    summarise,
)


def load_run(
    penalty=1.0,
    p=1,
    sweep_file="sim_sweep.json",
    seed=None,
):
    """
    Return the circuit, tuned angles and problem data
    for one (penalty, depth[, seed]) setting.

    seed:
        Pick the entry tuned with this optimiser seed.

    None:
        Use the first matching entry.
    """

    path = ROOT / "results" / "raw" / sweep_file

    entries = json.load(open(path))

    match = [
        entry
        for entry in entries
        if abs(entry["penalty"] - penalty) < 1e-9
        and entry["p"] == p
        and (
            seed is None
            or entry.get("seed") == seed
        )
    ]

    if not match:

        available = sorted(
            {
                (
                    entry["penalty"],
                    entry["p"],
                    entry.get("seed"),
                )
                for entry in entries
            }
        )

        raise ValueError(
            "No entry for "
            f"penalty={penalty}, "
            f"p={p}, "
            f"seed={seed} "
            f"in {sweep_file}. "
            f"Available: {available}"
        )

    entry = match[0]

    budget = entry["budget"]

    score, feasible = load_scores(
        budget=budget
    )

    H, _ = cost_hamiltonian(
        score,
        feasible,
        penalty,
    )

    if entry.get("max_pauli_weight") is not None:

        H = truncate_hamiltonian(
            H,
            entry["max_pauli_weight"],
        )

    gammas = entry["params"]["gammas"]
    betas = entry["params"]["betas"]

    circuit = build_circuit(
        H,
        gammas,
        betas,
        measure=True,
    )

    return {
        "penalty": penalty,
        "p": p,
        "seed": entry.get("seed"),
        "budget": budget,
        "max_pauli_weight":
            entry.get("max_pauli_weight"),
        "gammas": gammas,
        "betas": betas,
        "H": H,
        "circuit": circuit,
        "score": score,
        "feasible": feasible,
        "labels": design_labels(),
        "shots": entry.get(
            "shots",
            config.SHOTS,
        ),
        "sim_metrics": entry["metrics"],
    }


def counts_to_probs(
    counts,
    n_qubits=6,
):
    """
    Convert hardware measurement counts into probabilities.

    IMPORTANT:
    Bitstrings are used as-is.
    Do not reverse them.
    """

    probs = np.zeros(
        2 ** n_qubits
    )

    total = sum(
        counts.values()
    )

    for key, value in counts.items():

        clean_key = key.replace(
            " ",
            "",
        )

        index = int(
            clean_key,
            2,
        )

        probs[index] += (
            value / total
        )

    return probs


def evaluate(
    counts,
    run,
):
    """
    Evaluate measured hardware or noisy-simulator counts
    against the true design problem.
    """

    metrics = summarise(
        counts_to_probs(counts),
        run["score"],
        run["feasible"],
    )

    metrics["best_label"] = (
        run["labels"][
            int(
                metrics["best_design"],
                2,
            )
        ]
    )

    return metrics


def top_designs(
    counts,
    run,
    k=5,
):
    """
    Return the most frequently measured designs.
    """

    output = []

    ordered = sorted(
        counts.items(),
        key=lambda item: -item[1],
    )

    for key, value in ordered[:k]:

        index = int(
            key.replace(" ", ""),
            2,
        )

        output.append(
            (
                key,
                run["labels"][index],
                value,
                bool(
                    run["feasible"][index]
                ),
            )
        )

    return output