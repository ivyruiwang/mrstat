#!/bin/bash

# bash scripts/problem_size_scaling/run_problem_size_scaling.sh

set -e
cd /home/iwang3/mrstat_main

COMMON="--partition=defq"
SCRIPT_1GPU="scripts/problem_size_scaling/single_gpu_recon_N.jl"
SCRIPT_MPI="scripts/problem_size_scaling/mpi_mrstat_recon_N.jl"

OUTDIR="/home/iwang3/mrstat_main/problem_size_results"
mkdir -p "$OUTDIR"

echo "Output directory: $OUTDIR"


for N in 128 224 320 384 448 512; do

    if [ $N -le 224 ]; then
        TIME="02:00:00"
    elif [ $N -le 384 ]; then
        TIME="06:00:00"
    else
        TIME="12:00:00"
    fi

    # 1 GPU
    JOB=$(sbatch --parsable \
        --job-name=ps-1g-N${N} \
        --output=${OUTDIR}/ps_1g_N${N}_%j.out \
        --error=${OUTDIR}/ps_1g_N${N}_%j.err \
        --nodes=1 --ntasks=1 --gres=gpu:A4000:1 \
        --time=${TIME} \
        $COMMON \
        --wrap="
module load julia/1.10.3 cuda12.3/toolkit/12.3
export JULIA_DEPOT_PATH=/var/scratch/iwang3/julia_depot
mkdir -p /var/scratch/iwang3/julia_depot
export JULIA_CONDAPKG_ENV=/var/scratch/iwang3/condapkg_env
mkdir -p /var/scratch/iwang3/condapkg_env
export MPLBACKEND=agg
export JULIA_CONDAPKG_OFFLINE=true
export MRSTAT_N=${N}
cd ${OUTDIR}
julia --project=/home/iwang3/mrstat_main /home/iwang3/mrstat_main/${SCRIPT_1GPU}
")
    echo "  N=$N, 1 GPU:  job $JOB"

    # 2 GPUs (1 node)
    JOB=$(sbatch --parsable \
        --job-name=ps-2g-N${N} \
        --output=${OUTDIR}/ps_2g_N${N}_%j.out \
        --error=${OUTDIR}/ps_2g_N${N}_%j.err \
        --nodes=1 --ntasks=2 --gres=gpu:A4000:2 \
        --time=${TIME} \
        $COMMON \
        --wrap="
module load julia/1.10.3 cuda12.3/toolkit/12.3 openmpi4/4.1.6-cuda
export JULIA_DEPOT_PATH=/var/scratch/iwang3/julia_depot
mkdir -p /var/scratch/iwang3/julia_depot
export JULIA_CONDAPKG_ENV=/var/scratch/iwang3/condapkg_env
mkdir -p /var/scratch/iwang3/condapkg_env
export MPLBACKEND=agg
export JULIA_CONDAPKG_OFFLINE=true
export UCX_TLS=posix,sm,self
export UCX_NET_DEVICES=''
export MRSTAT_N=${N}
cd ${OUTDIR}
mpirun julia --project=/home/iwang3/mrstat_main /home/iwang3/mrstat_main/${SCRIPT_MPI}
")
    echo "  N=$N, 2 GPUs: job $JOB"

    # 4 GPUs (1 node)
    JOB=$(sbatch --parsable \
        --job-name=ps-4g-N${N} \
        --output=${OUTDIR}/ps_4g_N${N}_%j.out \
        --error=${OUTDIR}/ps_4g_N${N}_%j.err \
        --nodes=1 --ntasks=4 --gres=gpu:A4000:4 \
        --time=${TIME} \
        $COMMON \
        --wrap="
module load julia/1.10.3 cuda12.3/toolkit/12.3 openmpi4/4.1.6-cuda
export JULIA_DEPOT_PATH=/var/scratch/iwang3/julia_depot
mkdir -p /var/scratch/iwang3/julia_depot
export JULIA_CONDAPKG_ENV=/var/scratch/iwang3/condapkg_env
mkdir -p /var/scratch/iwang3/condapkg_env
export MPLBACKEND=agg
export JULIA_CONDAPKG_OFFLINE=true
export UCX_TLS=posix,sm,self
export UCX_NET_DEVICES=''
export MRSTAT_N=${N}
cd ${OUTDIR}
mpirun julia --project=/home/iwang3/mrstat_main /home/iwang3/mrstat_main/${SCRIPT_MPI}
")
    echo "  N=$N, 4 GPUs: job $JOB"

    # 6 GPUs (2 nodes)
    JOB=$(sbatch --parsable \
        --job-name=ps-6g-N${N} \
        --output=${OUTDIR}/ps_6g_N${N}_%j.out \
        --error=${OUTDIR}/ps_6g_N${N}_%j.err \
        --nodes=2 --ntasks-per-node=3 --gres=gpu:A4000:3 \
        --time=${TIME} \
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
export MRSTAT_N=${N}
cd ${OUTDIR}
mpirun --mca pml ob1 --mca btl tcp,self julia --project=/home/iwang3/mrstat_main /home/iwang3/mrstat_main/${SCRIPT_MPI}
")
    echo "  N=$N, 6 GPUs: job $JOB"

    # 8 GPUs (2 nodes)
    JOB=$(sbatch --parsable \
        --job-name=ps-8g-N${N} \
        --output=${OUTDIR}/ps_8g_N${N}_%j.out \
        --error=${OUTDIR}/ps_8g_N${N}_%j.err \
        --nodes=2 --ntasks-per-node=4 --gres=gpu:A4000:4 \
        --time=${TIME} \
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
export MRSTAT_N=${N}
cd ${OUTDIR}
mpirun --mca pml ob1 --mca btl tcp,self julia --project=/home/iwang3/mrstat_main /home/iwang3/mrstat_main/${SCRIPT_MPI}
")
    echo "  N=$N, 8 GPUs: job $JOB"

    echo ""
done

