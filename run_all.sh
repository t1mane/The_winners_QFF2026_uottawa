#!/usr/bin/env bash
echo "=== Running QAOA Nuclear Array Optimization Pipeline ==="

run_script() {
    if [ -f "$1" ]; then
        echo "[EXEC] Running $1..."
        python "$1" || echo "[WARNING] $1 encountered an error, continuing pipeline..."
    else
        echo "[SKIP] $1 not found, skipping for now."
    fi
}

run_script "experiments/run_exact.py"
run_script "experiments/run_sim_sweep.py"
run_script "experiments/run_noisy_sim.py"
run_script "experiments/run_hardware.py"
run_script "SRC/plotting.py"

echo "=== Pipeline execution completed! ==="
