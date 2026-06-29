#!/bin/bash
# submit MPI jobs with 1, 2, 4, 8 GPUs (all A4000)
# bash scripts/slurm/run_scaling_benchmark.sh

set -e
cd /home/iwang3/mrstat_main

COMMON="--time=01:00:00 --partition=defq"
SCRIPT="scripts/mpi_mrstat_recon.jl"

# JOB1=$(sbatch --parsable \
#     --job-name=mrstat-1g \
#     --output=mrstat_scaling_1g_%j.out \
#     --error=mrstat_scaling_1g_%j.err \
#     --nodes=1 --ntasks=1 --gres=gpu:A4000:1 \
#     $COMMON \
#     --wrap="
# module load julia/1.10.3 cuda12.3/toolkit/12.3 openmpi4/4.1.6-cuda
# export JULIA_DEPOT_PATH=/var/scratch/iwang3/julia_depot
# mkdir -p /var/scratch/iwang3/julia_depot
# export JULIA_CONDAPKG_ENV=/var/scratch/iwang3/condapkg_env
# mkdir -p /var/scratch/iwang3/condapkg_env
# export MPLBACKEND=agg
# export UCX_TLS=posix,sm,self
# export UCX_NET_DEVICES=''
# cd /home/iwang3/mrstat_main
# mpirun julia --project=. $SCRIPT
# ")
# echo "1 GPU:  job $JOB1"

# JOB2=$(sbatch --parsable \
#     --job-name=mrstat-2g \
#     --output=mrstat_scaling_2g_%j.out \
#     --error=mrstat_scaling_2g_%j.err \
#     --nodes=1 --ntasks=2 --gres=gpu:A4000:2 \
#     $COMMON \
#     --wrap="
# module load julia/1.10.3 cuda12.3/toolkit/12.3 openmpi4/4.1.6-cuda
# export JULIA_DEPOT_PATH=/var/scratch/iwang3/julia_depot
# mkdir -p /var/scratch/iwang3/julia_depot
# export JULIA_CONDAPKG_ENV=/var/scratch/iwang3/condapkg_env
# mkdir -p /var/scratch/iwang3/condapkg_env
# export MPLBACKEND=agg
# export UCX_TLS=posix,sm,self
# export UCX_NET_DEVICES=''
# cd /home/iwang3/mrstat_main
# mpirun julia --project=. $SCRIPT
# ")
# echo "2 GPUs: job $JOB2"

# JOB4=$(sbatch --parsable \
#     --job-name=mrstat-4g \
#     --output=mrstat_scaling_4g_%j.out \
#     --error=mrstat_scaling_4g_%j.err \
#     --nodes=1 --ntasks=4 --gres=gpu:A4000:4 \
#     $COMMON \
#     --wrap="
# module load julia/1.10.3 cuda12.3/toolkit/12.3 openmpi4/4.1.6-cuda
# export JULIA_DEPOT_PATH=/var/scratch/iwang3/julia_depot
# mkdir -p /var/scratch/iwang3/julia_depot
# export JULIA_CONDAPKG_ENV=/var/scratch/iwang3/condapkg_env
# mkdir -p /var/scratch/iwang3/condapkg_env
# export MPLBACKEND=agg
# export JULIA_CONDAPKG_OFFLINE=true
# export UCX_TLS=posix,sm,self
# export UCX_NET_DEVICES=''
# cd /home/iwang3/mrstat_main
# mpirun julia --project=. $SCRIPT
# ")
# echo "4 GPUs: job $JOB4"

JOB6=$(sbatch --parsable \
    --job-name=mrstat-6g \
    --output=mrstat_scaling_6g_%j.out \
    --error=mrstat_scaling_6g_%j.err \
    --nodes=2 --ntasks-per-node=3 --gres=gpu:A4000:3 \
    $COMMON \
    --wrap="
module load julia/1.10.3 cuda12.3/toolkit/12.3 openmpi4/4.1.6-cuda
export JULIA_DEPOT_PATH=/var/scratch/iwang3/julia_depot
mkdir -p /var/scratch/iwang3/julia_depot
export JULIA_CONDAPKG_ENV=/var/scratch/iwang3/condapkg_env
mkdir -p /var/scratch/iwang3/condapkg_env
export MPLBACKEND=agg
export JULIA_CONDAPKG_OFFLINE=true
export UCX_TLS=tcp,self,sm
export UCX_NET_DEVICES=all
cd /home/iwang3/mrstat_main
mpirun --mca pml ob1 --mca btl tcp,self julia --project=. $SCRIPT
")
echo "6 GPUs: job $JOB6"

# JOB8=$(sbatch --parsable \
#     --job-name=mrstat-8g \
#     --output=mrstat_scaling_8g_%j.out \
#     --error=mrstat_scaling_8g_%j.err \
#     --nodes=2 --ntasks-per-node=4 --gres=gpu:A4000:4 \
#     $COMMON \
#     --wrap="
# module load julia/1.10.3 cuda12.3/toolkit/12.3 openmpi4/4.1.6-cuda
# export JULIA_DEPOT_PATH=/var/scratch/iwang3/julia_depot
# mkdir -p /var/scratch/iwang3/julia_depot
# export JULIA_CONDAPKG_ENV=/var/scratch/iwang3/condapkg_env
# mkdir -p /var/scratch/iwang3/condapkg_env
# export MPLBACKEND=agg
# export JULIA_CONDAPKG_OFFLINE=true
# export UCX_TLS=tcp,self,sm
# export UCX_NET_DEVICES=all
# cd /home/iwang3/mrstat_main
# mpirun --mca pml ob1 --mca btl tcp,self julia --project=. $SCRIPT
# ")
# echo "8 GPUs: job $JOB8"

echo "After completion, run: julia --project=. scripts/plot_scaling.jl"
