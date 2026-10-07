# IBM Quantum Hardware Notes

## Environment

The IBM Quantum workflow was tested using the Qiskit Fall Fest environment and the PINQ2 shared IBM Quantum allocation.

- Backend: `ibm_quebec`
- Python: 3.12
- Transpiler optimization level: 1
- Transpiler seed: 42

API tokens are entered securely at runtime and are not stored in the repository.

## IBM hardware smoke test

A two-qubit Bell-state circuit was used to verify access to the real IBM Quantum backend.

Configuration:

- Backend: `ibm_quebec`
- Shots: 256
- Compiled circuit depth: 8
- Native two-qubit gates: 1

IBM job ID:

`db2nrj42ljfc73d4r3i0`

Measured counts:

- `00`: 122
- `11`: 131
- `01`: 3

The expected Bell-state outcomes (`00` and `11`) represented 253 of 256 shots, approximately 98.83%.

This confirmed that authentication, backend selection, transpilation, job submission, and result retrieval were working.

## Backend-derived noisy simulator test

The noisy-simulation workflow was validated using an existing frozen QAOA parameter set.

Test configuration:

- Backend noise model: `ibm_quebec`
- Penalty: M = 1.0
- QAOA depth: p = 1
- Optimization seed: 42
- Gamma: 0.9115090896179474
- Beta: 2.773012667772475
- Shots: 256
- Compiled circuit depth: 830
- Native two-qubit gates: 287

The simulator returned all 256 requested shots.

Measured metrics:

- P(best): 0.03125
- P(feasible): 0.44921875
- P(top 3): 0.10546875
- Mean score conditioned on feasibility: approximately 0.5837

This run was used only to validate the backend-derived noisy simulation and compilation pipeline. It is not one of the final hardware comparison runs.

## Final hardware experiment

The final hardware experiment will use three frozen QAOA parameter sets:

- QAOA depth: p = 1
- Penalty: M = 2.0
- Optimization seeds: 11, 22, 33
- Shots: 2048 per circuit

The same frozen parameter sets will be evaluated using:

1. Ideal simulation
2. Backend-derived noisy simulation
3. Real IBM Quantum hardware

The final runs will only be executed after `results/raw/hardware_params.json` contains all three required parameter sets.

## Reproducibility

The IBM workflow records:

- Optimization seed
- Gamma and beta parameters
- Backend name
- Transpiler seed
- Circuit depth
- Native two-qubit gate count
- Number of shots
- Measurement counts
- Quality metrics
- IBM job ID
- IBM job metrics