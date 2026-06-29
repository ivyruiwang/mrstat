# Multi-GPU MR-STAT Reconstruction

This repository contains the Julia implementation developed for the thesis *Multi-GPU Implementation of MR-STAT for Quantitative MRI Reconstruction*.

The implementation extends an MR-STAT reconstruction workflow to multiple GPUs using CUDA and MPI.

## New Added Code

- `src/mpi/`: multi-GPU/MPI implementation
- `scripts/`: entry script used for reconstruction
- `scripts/slurm/`: SLURM script used on the DAS-6 cluster
- `scripts/problem_size_scaling/`: script for problem-size scaling experiments
- `scripts/compare_results.jl`: script for reconstruction accuracy and convergence behaviour
- `scripts/plot_scaling.jl`: script for strong-scaling results
