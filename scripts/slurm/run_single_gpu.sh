#!/bin/bash
#SBATCH --job-name=mrstat-baseline
#SBATCH --output=mrstat_baseline_%j.out
#SBATCH --error=mrstat_baseline_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A4000:1
#SBATCH --time=01:00:00

module load julia/1.10.3
module load cuda12.3/toolkit/12.3

echo "Setting up Julia depot..."
export JULIA_DEPOT_PATH="/var/scratch/iwang3/julia_depot"
mkdir -p "$JULIA_DEPOT_PATH"

echo "Setting up CondaPkg environment..."
export JULIA_CONDAPKG_ENV="/var/scratch/iwang3/condapkg_env"
mkdir -p "$JULIA_CONDAPKG_ENV"

export MPLBACKEND="agg"

cd /home/iwang3/mrstat_main

julia --project=. scripts/single_gpu_recon.jl

echo "Job finished"
