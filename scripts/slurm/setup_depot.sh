#!/bin/bash

# One-time install all Julia dependencies
# Step 1: On Login node, Pkg.instantiate + PythonPlot 
# Step 2: On compute node, recompile CUDA runtime + precompile MRSTAT 
# Safe to re-run because already compiled packages are skipped automatically
# bash scripts/slurm/setup_depot.sh


cd /home/iwang3/mrstat_main

module load julia/1.10.3

export JULIA_DEPOT_PATH=/var/scratch/iwang3/julia_depot
mkdir -p /var/scratch/iwang3/julia_depot
export JULIA_CONDAPKG_ENV=/var/scratch/iwang3/condapkg_env
mkdir -p /var/scratch/iwang3/condapkg_env
export MPLBACKEND=agg

echo "Step 1: Login node Pkg.instantiate + PythonPlot"
julia --project=. -e '
    using Pkg
    Pkg.instantiate()
    println("Packages instantiated")
    using PythonPlot
    println("PythonPlot OK")
'

echo "Step 2: GPU node CUDA runtime + MRSTAT precompilation"

srun --gres=gpu:A4000:1 --time=00:30:00 bash -c '
module load julia/1.10.3 cuda12.3/toolkit/12.3 openmpi4/4.1.6-cuda
export JULIA_DEPOT_PATH=/var/scratch/iwang3/julia_depot
export JULIA_CONDAPKG_ENV=/var/scratch/iwang3/condapkg_env
export MPLBACKEND=agg
cd /home/iwang3/mrstat_main
julia --project=. -e "
    pkg = Base.PkgId(Base.UUID(\"76a88914-d11a-5bdc-97e0-2f5a05c973a2\"), \"CUDA_Runtime_jll\")
    Base.compilecache(pkg)
    using MPI; 
    using CUDA; 
    using MRSTAT;
    using JLD2; 
"
'
