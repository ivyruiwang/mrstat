#!/bin/bash
# ============================================================================
# One-time setup: install all Julia dependencies
# Step 1: Login node — Pkg.instantiate + PythonPlot (needs internet)
# Step 2: GPU node — recompile CUDA runtime + precompile MRSTAT (needs GPU)
#
# Safe to re-run: already-compiled packages are skipped automatically.
#
# Usage: bash scripts/slurm/setup_depot.sh
# ============================================================================

cd /home/iwang3/mrstat_main

module load julia/1.10.3

export JULIA_DEPOT_PATH=/var/scratch/iwang3/julia_depot
mkdir -p /var/scratch/iwang3/julia_depot
export JULIA_CONDAPKG_ENV=/var/scratch/iwang3/condapkg_env
mkdir -p /var/scratch/iwang3/condapkg_env
export MPLBACKEND=agg

# Step 1: Login node — install packages and precompile PythonPlot (needs internet)
echo "=== Step 1: Login node — Pkg.instantiate + PythonPlot ==="
julia --project=. -e '
    using Pkg
    Pkg.instantiate()
    println("Packages instantiated")
    using PythonPlot
    println("PythonPlot OK")
'

echo ""
echo "=== Step 2: GPU node — CUDA runtime + MRSTAT precompilation ==="

srun --gres=gpu:A4000:1 --time=00:30:00 bash -c '
module load julia/1.10.3 cuda12.3/toolkit/12.3 openmpi4/4.1.6-cuda
export JULIA_DEPOT_PATH=/var/scratch/iwang3/julia_depot
export JULIA_CONDAPKG_ENV=/var/scratch/iwang3/condapkg_env
export MPLBACKEND=agg
cd /home/iwang3/mrstat_main
julia --project=. -e "
    # Step 1 compiled CUDA_Runtime_jll without GPU — fix it here
    pkg = Base.PkgId(Base.UUID(\"76a88914-d11a-5bdc-97e0-2f5a05c973a2\"), \"CUDA_Runtime_jll\")
    Base.compilecache(pkg)
    println(\"CUDA_Runtime_jll recompiled on GPU node\")

    using MPI; println(\"MPI OK\")
    using CUDA; println(\"CUDA OK: \", CUDA.functional(), \" \", CUDA.name(CUDA.device()))
    using MRSTAT; println(\"MRSTAT OK\")
    using JLD2; println(\"JLD2 OK\")
    println(\"All OK\")
"
'
