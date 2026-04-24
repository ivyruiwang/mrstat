#!/bin/bash
# ============================================================================
# Run problem size scaling analysis on a GPU compute node
#
# Usage: bash scripts/problem_size_scaling/run_analysis.sh
# ============================================================================

cd /home/iwang3/mrstat_main

RESULTS_DIR="/home/iwang3/mrstat_main/problem_size_results"

srun --gres=gpu:A4000:1 --time=00:10:00 bash -c "
module load julia/1.10.3 cuda12.3/toolkit/12.3
export JULIA_DEPOT_PATH=/var/scratch/iwang3/julia_depot
export JULIA_CONDAPKG_ENV=/var/scratch/iwang3/condapkg_env
export JULIA_CONDAPKG_OFFLINE=true
export MPLBACKEND=agg
cd /home/iwang3/mrstat_main

echo '=== Problem Size Scaling Analysis ==='
julia --project=. scripts/problem_size_scaling/plot_problem_size.jl ${RESULTS_DIR}
"
