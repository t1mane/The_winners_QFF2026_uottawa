#!/usr/bin/env bash
set -e
python experiments/run_exact.py
python experiments/run_sim_sweep.py
python src/plotting.py
