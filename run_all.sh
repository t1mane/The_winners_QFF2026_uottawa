#!/usr/bin/env bash
echo "=== QNanoVolt: QAOA solar-cell design pipeline ==="
run_script() {
    if [ -f "$1" ]; then
        echo "[EXEC] Running $1..."
        python "$1" || echo "[WARNING] $1 encountered an error, continuing pipeline..."
    else
        echo "[SKIP] $1 not found, skipping."
    fi
}
run_script experiments/run_exact.py
run_script experiments/run_sim_sweep.py

if [ -n "$RUN_IBM" ]; then run_script experiments/run_noisy_sim.py; fi

run_script src/plotting.py
echo "=== Pipeline execution completed! ==="
