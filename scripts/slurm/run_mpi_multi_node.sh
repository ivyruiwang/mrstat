#!/bin/bash
# ============================================================================
#  sbatch scripts/slurm/run_mpi_multi_node.sh
# ============================================================================
#SBATCH --job-name=mrstat-mpi-mn
#SBATCH --output=mrstat_mpi_%j.out
#SBATCH --error=mrstat_mpi_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A4000:4
#SBATCH --time=01:00:00

module load julia/1.10.3
module load cuda12.3/toolkit/12.3
module load openmpi4/4.1.6-cuda

export JULIA_DEPOT_PATH="/var/scratch/iwang3/julia_depot"
export JULIA_CONDAPKG_ENV="/var/scratch/iwang3/condapkg_env"
export MPLBACKEND="agg"

# IB unreliable on this cluster, use TCP for inter-node
export UCX_TLS=tcp,self,sm
export UCX_NET_DEVICES=all

cd /home/iwang3/mrstat_main

echo "============================================="
echo "  MPI MRSTAT — 2 nodes × 4 GPUs per node"
echo "  Nodes: ${SLURM_NNODES}"
echo "  Tasks: ${SLURM_NTASKS}"
echo "  Job  : ${SLURM_JOB_ID}"
echo "============================================="

mpirun --mca pml ob1 --mca btl tcp,self julia --project=. scripts/mpi_mrstat_recon.jl

echo "Job finished at $(date)"
