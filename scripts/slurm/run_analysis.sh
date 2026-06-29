#!/bin/bash

# bash scripts/slurm/run_analysis.sh <results_dir>

RESULTS_DIR="$1"

cd /home/iwang3/mrstat_main

srun --gres=gpu:A4000:1 --time=00:30:00 bash -c "
module load julia/1.10.3 cuda12.3/toolkit/12.3
export JULIA_DEPOT_PATH=/var/scratch/iwang3/julia_depot
export JULIA_CONDAPKG_ENV=/var/scratch/iwang3/condapkg_env
export JULIA_CONDAPKG_OFFLINE=true
export MPLBACKEND=agg
cd /home/iwang3/mrstat_main

# Check if results have noise
HAS_NOISE=\$(julia --project=. -e '
    using JLD2
    d = load(\"${RESULTS_DIR}/results_single_gpu.jld2\")
    nf = haskey(d, \"noise_floor\") ? d[\"noise_floor\"] : 0.0
    print(nf > 0 ? \"yes\" : \"no\")
')
echo \"Noise detected: \${HAS_NOISE}\"

echo 'Step 1: Correctness Validation '
julia --project=. scripts/compare_results.jl \\
    ${RESULTS_DIR}/results_single_gpu.jld2 \\
    ${RESULTS_DIR}/results_mpi_1n2g.jld2 \\
    ${RESULTS_DIR}/results_mpi_1n4g.jld2 \\
    ${RESULTS_DIR}/results_mpi_2n6g.jld2 \\
    ${RESULTS_DIR}/results_mpi_2n8g.jld2

echo ''
echo 'Step 2: Scaling Analysis '
julia --project=. scripts/plot_scaling.jl \\
    ${RESULTS_DIR}/results_mpi_1n1g.jld2 \\
    ${RESULTS_DIR}/results_mpi_1n2g.jld2 \\
    ${RESULTS_DIR}/results_mpi_1n4g.jld2 \\
    ${RESULTS_DIR}/results_mpi_2n6g.jld2 \\
    ${RESULTS_DIR}/results_mpi_2n8g.jld2

if [ \"\${HAS_NOISE}\" = \"yes\" ]; then
    echo ''
    echo 'Step 3: Noisy Parameter Map Figure '
    julia --project=. scripts/plot_parammap_figure.jl \\
        ${RESULTS_DIR}/results_single_gpu.jld2 \\
        ${RESULTS_DIR}/results_mpi_1n4g.jld2 \\
        ${RESULTS_DIR}/results_mpi_2n8g.jld2
else
    echo ''
    echo 'Step 3: Skipped (no noise) '
fi

if [ \"\${HAS_NOISE}\" = \"no\" ]; then
    echo ''
    echo 'Step 4: Noiseless Parameter Map Figure '
    julia --project=. scripts/plot_parammap_noiseless.jl \\
        ${RESULTS_DIR}/results_single_gpu.jld2 \\
        ${RESULTS_DIR}/results_mpi_1n4g.jld2 \\
        ${RESULTS_DIR}/results_mpi_2n8g.jld2
else
    echo ''
    echo 'Step 4: Skipped (has noise) '
fi

ls -la /home/iwang3/mrstat_main/analysis_results/ 2>/dev/null
"
