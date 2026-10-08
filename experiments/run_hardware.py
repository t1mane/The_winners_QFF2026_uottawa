"""
Submit and retrieve the three frozen QAOA hardware circuits.

SUBMIT:
    python experiments/run_hardware.py --submit

RETRIEVE:
    python experiments/run_hardware.py --retrieve JOB_ID
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from qiskit_ibm_runtime.executor_sampler import Sampler


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
    get_counts,
    get_reproducibility_metadata,
)


BACKEND_NAME = "ibm_quebec"
SHOTS = 2048
TRANSPILER_SEED = 42

RESULT_DIR = ROOT / "results" / "raw"


def json_converter(obj):
    """
    Convert NumPy and other non-standard objects
    into JSON-serializable values.
    """

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return str(obj)


def load_hardware_runs():
    """
    Load the three final frozen QAOA parameter sets.

    Final hardware configuration:
    - penalty M = 2.0
    - QAOA depth p = 1
    - seeds = 11, 22, 33
    """

    runs = []

    for seed in [11, 22, 33]:
        run = load_run(
            penalty=2.0,
            p=1,
            sweep_file="hardware_params.json",
            seed=seed,
        )

        runs.append(run)

    return runs


def submit():
    """
    Compile and submit the three final QAOA circuits
    as one IBM Quantum job.
    """

    hardware_file = (
        ROOT
        / "results"
        / "raw"
        / "hardware_params.json"
    )

    # Safety check:
    # never submit real hardware before the final
    # frozen parameters are available.
    if not hardware_file.exists():
        raise FileNotFoundError(
            "Missing results/raw/hardware_params.json. "
            "Do not submit to IBM hardware until the final "
            "p=1, M=2, seeds 11/22/33 parameters are available."
        )

    print("Loading frozen QAOA circuits...")

    runs = load_hardware_runs()

    print("Connecting to IBM / PINQ2...")

    service = connect_to_ibm()

    backend = get_backend(
        service,
        BACKEND_NAME,
    )

    print(
        "Backend:",
        backend.name,
    )

    compiled_circuits = []
    circuit_metadata = []

    for run in runs:
        print()
        print(
            f"Compiling seed {run['seed']}"
        )

        compiled = compile_circuit(
            run["circuit"],
            backend,
            optimization_level=1,
            transpiler_seed=TRANSPILER_SEED,
        )

        resources = get_resource_metrics(
            compiled
        )

        print(
            "Depth:",
            resources["compiled_depth"],
        )

        print(
            "Two-qubit gates:",
            resources["native_two_qubit_gates"],
        )

        compiled_circuits.append(
            compiled
        )

        circuit_metadata.append(
            {
                "seed":
                    run["seed"],

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
                    run["max_pauli_weight"],

                "resources":
                    resources,
            }
        )

    # Record the exact code/environment BEFORE submission.
    reproducibility = get_reproducibility_metadata()

    print()
    print("Submitting three circuits...")

    sampler = Sampler(
        mode=backend
    )

    job = sampler.run(
        compiled_circuits,
        shots=SHOTS,
    )

    job_id = job.job_id()
    status = str(job.status())

    print()
    print(
        "JOB ID:",
        job_id,
    )

    print(
        "STATUS:",
        status,
    )

    record = {
        "job_id":
            job_id,

        "status":
            status,

        "backend":
            backend.name,

        "execution_mode":
            "job",

        "shots_per_circuit":
            SHOTS,

        "transpiler_seed":
            TRANSPILER_SEED,

        "reproducibility":
            reproducibility,

        "circuits":
            circuit_metadata,
    }

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        RESULT_DIR
        / f"hardware_job_{job_id}.json"
    )

    # Save the job ID and submission metadata immediately.
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            record,
            file,
            indent=2,
            default=json_converter,
        )

    print()
    print("Saved submission metadata:")
    print(path)

    print()
    print("DO NOT resubmit if queued.")

    print(
        "Retrieve later with:"
    )

    print(
        "python experiments/run_hardware.py "
        f"--retrieve {job_id}"
    )


def retrieve(job_id):
    """
    Retrieve a previously submitted IBM Quantum job.

    This function never submits a new job.
    """

    print("Connecting to IBM / PINQ2...")

    service = connect_to_ibm()

    job = service.job(
        job_id
    )

    status = str(
        job.status()
    )

    print(
        "JOB ID:",
        job_id,
    )

    print(
        "STATUS:",
        status,
    )

    path = (
        RESULT_DIR
        / f"hardware_job_{job_id}.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing submission file: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        record = json.load(
            file
        )

    record["status"] = status

    # If the job is still queued/running,
    # update its status without submitting anything new.
    if "DONE" not in status.upper():
        print(
            "Job is not finished."
        )

        print(
            "No new job was submitted."
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                record,
                file,
                indent=2,
                default=json_converter,
            )

        return

    print(
        "Downloading hardware results..."
    )

    result = job.result()

    runs = load_hardware_runs()

    for index, run in enumerate(
        runs
    ):
        counts = get_counts(
            result,
            pub_index=index,
        )

        metrics = evaluate(
            counts,
            run,
        )

        top = top_designs(
            counts,
            run,
        )

        record[
            "circuits"
        ][index]["counts"] = counts

        record[
            "circuits"
        ][index][
            "returned_shots"
        ] = sum(
            counts.values()
        )

        record[
            "circuits"
        ][index][
            "metrics"
        ] = metrics

        record[
            "circuits"
        ][index][
            "top_designs"
        ] = top

    record[
        "job_metrics"
    ] = job.metrics()

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            record,
            file,
            indent=2,
            default=json_converter,
        )

    print()
    print(
        "Results saved:"
    )

    print(path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Submit or retrieve the final "
            "QNanoVolt IBM Quantum hardware experiment."
        )
    )

    group = (
        parser
        .add_mutually_exclusive_group(
            required=True
        )
    )

    group.add_argument(
        "--submit",
        action="store_true",
        help=(
            "Submit the three final frozen "
            "QAOA circuits."
        ),
    )

    group.add_argument(
        "--retrieve",
        metavar="JOB_ID",
        help=(
            "Retrieve an existing IBM job. "
            "This does not submit a new job."
        ),
    )

    args = parser.parse_args()

    if args.submit:
        submit()

    else:
        retrieve(
            args.retrieve
        )


if __name__ == "__main__":
    main()