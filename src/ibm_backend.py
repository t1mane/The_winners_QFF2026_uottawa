"""
Helper functions for IBM Quantum execution.

This file handles:
- PINQ2 / IBM Quantum connection
- backend selection
- circuit transpilation
- circuit resource metrics
- extracting measurement counts
"""

import getpass
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService


DEFAULT_BACKEND = "ibm_quebec"
DEFAULT_TRANSPILER_SEED = 42


def connect_to_ibm(token=None, instance=None):
    """
    Connect to the Qiskit Fall Fest PINQ2 IBM Quantum allocation.

    The API token is:
    1. passed directly,
    2. read from PINQ2_TOKEN,
    3. or requested securely from the user.
    """

    if token is None:
        token = os.getenv("PINQ2_TOKEN")

    if not token:
        token = getpass.getpass("PINQ2 API token: ").strip()

    if not token:
        raise ValueError("No PINQ2 API token was provided.")

    if instance is None:
        instance = os.getenv("PINQ2_INSTANCE")

    if not instance:
        instance = "auto"

    service = QiskitRuntimeService(
        channel="ibm_cloud",
        token=token,
        instance=instance,
    )

    return service


def get_backend(service, backend_name=DEFAULT_BACKEND):
    """
    Return the IBM backend used for the experiment.
    """

    return service.backend(backend_name)


def compile_circuit(
    circuit,
    backend,
    optimization_level=1,
    transpiler_seed=DEFAULT_TRANSPILER_SEED,
):
    """
    Compile a circuit for the selected IBM quantum computer.
    """

    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=optimization_level,
        seed_transpiler=transpiler_seed,
    )

    compiled_circuit = pass_manager.run(circuit)

    return compiled_circuit


def get_resource_metrics(circuit):
    """
    Collect circuit depth, gate counts and native two-qubit gate usage.
    """

    operation_counts = {
        str(name): int(count)
        for name, count in circuit.count_ops().items()
    }

    two_qubit_gate_counts = {}

    ignored_operations = {
        "barrier",
        "measure",
        "reset",
        "delay",
    }

    for instruction in circuit.data:
        operation = instruction.operation
        qubits = instruction.qubits
        gate_name = operation.name

        if (
            len(qubits) == 2
            and gate_name not in ignored_operations
        ):
            two_qubit_gate_counts[gate_name] = (
                two_qubit_gate_counts.get(gate_name, 0) + 1
            )

    total_two_qubit_gates = sum(
        two_qubit_gate_counts.values()
    )

    return {
        "compiled_depth": int(circuit.depth()),
        "num_qubits": int(circuit.num_qubits),
        "operation_counts": operation_counts,
        "two_qubit_gate_counts": two_qubit_gate_counts,
        "native_two_qubit_gates": int(total_two_qubit_gates),
    }

def get_reproducibility_metadata():
    """
    Record the exact code and software environment used for a run.
    """

    repo_root = Path(__file__).resolve().parent.parent

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
        ).strip()
    except Exception:
        git_commit = None

    try:
        git_dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                text=True,
            ).strip()
        )
    except Exception:
        git_dirty = None

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python_version": platform.python_version(),
        "qiskit_version": version("qiskit"),
        "qiskit_aer_version": version("qiskit-aer"),
        "qiskit_ibm_runtime_version": version("qiskit-ibm-runtime"),
    }

def get_counts(result, pub_index=0):
    """
    Extract measurement counts from an IBM Sampler result.

    QNanoVolt circuits use measure_all(), which creates
    the measurement register named 'meas'.
    """

    return result[pub_index].data.meas.get_counts()