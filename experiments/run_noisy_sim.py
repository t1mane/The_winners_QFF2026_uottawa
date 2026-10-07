"""
Run frozen QAOA circuits using a noise model derived
from the IBM Quantum backend.

TEST:
    python experiments/run_noisy_sim.py --test

FINAL:
    python experiments/run_noisy_sim.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from qiskit_aer import AerSimulator


ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))


from hardware_inputs import (
    load_run,
    evaluate,
    top_designs,
)

from ibm_backend import (
    connect_to_ibm,
    get_backend,
    compile_circuit,
    get_resource_metrics,
)


BACKEND_NAME = "ibm_quebec"

OUTPUT_FILE = (
    ROOT
    / "results"
    / "raw"
    / "noisy_sim.json"
)

SHOTS = 2048

TRANSPILER_SEED = 42
SIMULATOR_SEED = 2026


def json_converter(obj):

    if isinstance(
        obj,
        np.integer,
    ):
        return int(obj)

    if isinstance(
        obj,
        np.floating,
    ):
        return float(obj)

    if isinstance(
        obj,
        np.ndarray,
    ):
        return obj.tolist()

    raise TypeError(
        f"Cannot serialize {type(obj)}"
    )


def main(test_mode=False):
    if not test_mode:

        hardware_file = (
            ROOT
            / "results"
            / "raw"
            / "hardware_params.json"
        )

        if not hardware_file.exists():
            raise FileNotFoundError(
                "Missing results/raw/hardware_params.json. "
                "Wait for the final p=1, M=2, seeds 11/22/33 "
                "parameters before running the final noisy simulation."
            )


    print(
        "Connecting to PINQ2 / IBM..."
    )

    service = connect_to_ibm()

    backend = get_backend(
        service,
        BACKEND_NAME,
    )

    print(
        "Backend:",
        backend.name,
    )

    print(
        "Creating backend-derived "
        "noise simulator..."
    )

    simulator = AerSimulator.from_backend(
        backend,
        method="matrix_product_state",
    )

    # ---------------------------------------------
    # TEST MODE
    # ---------------------------------------------

    if test_mode:

        configurations = [
            {
                "seed": 42,
                "penalty": 1.0,
                "p": 1,
                "file": "sim_sweep.json",
                "shots": 256,
            }
        ]

        output_file = (
            ROOT
            / "results"
            / "raw"
            / "noisy_test.json"
        )

    # ---------------------------------------------
    # FINAL HARDWARE-MATCHED RUNS
    # ---------------------------------------------

    else:

        configurations = [
            {
                "seed": 11,
                "penalty": 2.0,
                "p": 1,
                "file": "hardware_params.json",
                "shots": SHOTS,
            },
            {
                "seed": 22,
                "penalty": 2.0,
                "p": 1,
                "file": "hardware_params.json",
                "shots": SHOTS,
            },
            {
                "seed": 33,
                "penalty": 2.0,
                "p": 1,
                "file": "hardware_params.json",
                "shots": SHOTS,
            },
        ]

        output_file = OUTPUT_FILE

    saved_results = []

    for config in configurations:

        seed = config["seed"]

        print()
        print("=" * 60)

        print(
            f"Running seed {seed}"
        )

        print("=" * 60)

        run = load_run(
            penalty=config["penalty"],
            p=config["p"],
            sweep_file=config["file"],
            seed=seed,
        )

        circuit = run["circuit"]

        print(
            "Gamma:",
            run["gammas"],
        )

        print(
            "Beta:",
            run["betas"],
        )

        print(
            "Max Pauli weight:",
            run["max_pauli_weight"],
        )

        compiled = compile_circuit(
            circuit,
            backend,
            optimization_level=1,
            transpiler_seed=TRANSPILER_SEED,
        )

        resources = get_resource_metrics(
            compiled
        )

        print(
            "Compiled depth:",
            resources["compiled_depth"],
        )

        print(
            "Two-qubit gates:",
            resources[
                "native_two_qubit_gates"
            ],
        )

        job = simulator.run(
            compiled,
            shots=config["shots"],
            seed_simulator=(
                SIMULATOR_SEED
                + seed
            ),
        )

        result = job.result()

        counts = result.get_counts()

        metrics = evaluate(
            counts,
            run,
        )

        top = top_designs(
            counts,
            run,
        )

        print(
            "Returned shots:",
            sum(counts.values()),
        )

        print(
            "Metrics:",
            metrics,
        )

        print(
            "Top designs:"
        )

        for design in top:
            print(design)

        saved_results.append(
            {
                "environment":
                    "backend_derived_noisy_simulator",

                "backend":
                    backend.name,

                "seed":
                    seed,

                "p":
                    run["p"],

                "penalty":
                    run["penalty"],

                "budget":
                    run["budget"],

                "gammas":
                    run["gammas"],

                "betas":
                    run["betas"],

                "max_pauli_weight":
                    run[
                        "max_pauli_weight"
                    ],

                "requested_shots":
                    config["shots"],

                "returned_shots":
                    sum(
                        counts.values()
                    ),

                "resources":
                    resources,

                "counts":
                    counts,

                "metrics":
                    metrics,

                "top_designs":
                    top,
            }
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_file,
        "w",
    ) as file:

        json.dump(
            saved_results,
            file,
            indent=2,
            default=json_converter,
        )

    print()
    print("=" * 60)

    print(
        "Saved results to:"
    )

    print(output_file)

    print("=" * 60)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Run one small noisy simulation "
            "using existing seed-42 parameters."
        ),
    )

    args = parser.parse_args()

    main(
        test_mode=args.test
    )