#!/usr/bin/env julia
# ============================================================================
# Distributed Solver — MPI MRSTAT Reconstruction
# Run commands below on SLURM:
# sbatch scripts/slurm/run_mpi_single_node.sh
# sbatch scripts/slurm/run_mpi_multi_node.sh
# ============================================================================

using MPI
MPI.Init()

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))   # activate mrstat_main project

using CUDA
using BlochSimulators
using BlochSimulators: f32, gpu
using MRSTAT
using MRSTAT: NUM_COILS, TrustRegionReflective, DerivativeOperations,
              optim_to_physical_pars, plot_T₁T₂ρ
using MRSTAT.DerivativeOperations: simulate_derivatives, Jv, Jᴴv
using ComputationalResources: AbstractResource, CUDALibs
using LinearAlgebra, LinearMaps, StaticArrays, StructArrays, Statistics
using JLD2

# Load MPI + distributed solver definitions
include(joinpath(@__DIR__, "..", "src", "mpi", "MPIResources.jl"))
include(joinpath(@__DIR__, "..", "src", "mpi", "mpi_objective.jl"))
include(joinpath(@__DIR__, "..", "src", "mpi", "mpi_utils.jl"))
include(joinpath(@__DIR__, "..", "src", "mpi", "mpi_steihaug.jl"))
include(joinpath(@__DIR__, "..", "src", "mpi", "mpi_solver.jl"))

# MPI Setup
res    = MPICUDALibs()
comm   = res.comm
rank   = res.rank
nranks = res.nranks
rank == 0 && println("=== MPI MRSTAT Reconstruction (Distributed Solver) ===")
rank == 0 && println("    Ranks: $nranks")
println("    Rank $rank → GPU $(res.local_gpu_id) ($(CUDA.name(CUDA.device())))")
MPI.Barrier(comm)

# Data Generation
# All ranks generate identical data then partition voxel-wise
rank == 0 && println("\nGenerating simulation data...")
using Random; Random.seed!(42)   # !!! MUST fixed seed so all ranks generate exactly same phantom
raw_data, sequence, coords_full, coils_full, trajectory = MRSTAT.generate_simulation_data()

total_nvox = length(coords_full)
N = isqrt(total_nvox)

rank == 0 && println("    Image: $N × $N ($total_nvox voxels)")

# # Add complex Gaussian noise (SNR_dB = 15.36, matching original paper)
# # All ranks use same seed → same noise → same corrupted data
# SNR_dB = 15.36
# raw_data_cpu = Array(raw_data)
# rms_signal = sqrt(mean(abs.(raw_data_cpu) .^ 2))
# rms_noise = Float32(rms_signal / sqrt(10^(SNR_dB / 10)))
# Random.seed!(123)   # separate seed for noise (same across all ranks)
# noise = rms_noise * randn(ComplexF32, size(raw_data_cpu))
# noise_floor = Float64(0.5 * sum(abs.(noise) .^ 2))
# raw_data = gpu(raw_data_cpu .+ noise)
# rank == 0 && println("    Noise: SNR_dB=$SNR_dB, noise_floor=$noise_floor")

# Voxel Partitioning
vr = voxel_partition(total_nvox, rank, nranks)
local_nvox = length(vr)
println("    Rank $rank → voxels $(first(vr)):$(last(vr)) ($local_nvox voxels)")

local_coords = partition_structarray(coords_full, vr)
local_coils  = gpu(f32(Array(coils_full)[vr, :]))
local_tx     = ones(Float32, local_nvox)   # uniform transmit field

MPI.Barrier(comm)
rank == 0 && println("Data partitioned.\n")

# Optimization — local vectors
x0_per = Float32[log(1.0), log(0.100), 1.0, 0.0]
LB_per = Float32[log(0.1), log(0.001), -Inf, -Inf]
UB_per = Float32[log(7.0), log(3.000),  Inf,  Inf]

x0 = repeat(x0_per', local_nvox) |> vec
LB = repeat(LB_per', local_nvox) |> vec
UB = repeat(UB_per', local_nvox) |> vec

transmit_field_full = ones(Float32, total_nvox)

# plotfun receives global x (already gathered by mpi_solver)
nnodes_cfg = parse(Int, get(ENV, "SLURM_NNODES", "1"))
config_tag = "$(nnodes_cfg)n$(nranks)g"

plotfun(x, figtitle) = if is_root(res)
    physical = MRSTAT.optim_to_physical_pars(x, transmit_field_full)
    MRSTAT.plot_T₁T₂ρ(physical, N, N, "$(config_tag)_$(figtitle)")
end

# Initial plot needs gather
x0_full = mpi_gather_field_major(x0, local_nvox, total_nvox, comm)
plotfun(x0_full, "Initial Guess")

objfun = (x_local, mode) -> mpi_objective(
    x_local, res, mode,
    raw_data, sequence, local_coords, local_coils,
    trajectory, local_tx,
)

trf_opts = TrustRegionReflective.SolverOptions()

# Run Reconstruction

rank == 0 && println("Running distributed TRF solver …\n")
MPI.Barrier(comm)

output = mpi_solver(objfun, x0, LB, UB, trf_opts, plotfun, comm, total_nvox)

MPI.Barrier(comm)

if is_root(res)
    println("\n=== Reconstruction complete ===")
    println("    Final cost: $(output.f[end])")
    println("    Iterations: $(size(output.f, 2))")

    # Save results + ground truth for comparison
    Random.seed!(42)
    ground_truth = MRSTAT.make_phantom(N)
    transmit_field = transmit_field_full

    nnodes = parse(Int, get(ENV, "SLURM_NNODES", "1"))
    meta = Dict(
        "ngpus"    => nranks,
        "nnodes"   => nnodes,
        "solver"   => "mpi",
        "job_id"   => get(ENV, "SLURM_JOB_ID", "local"),
        "hostname" => gethostname(),
    )
    outfile = "results_mpi_$(nnodes)n$(nranks)g.jld2"
    # jldsave(outfile; output, ground_truth, N, transmit_field, meta, noise_floor) # noisy phantom
    jldsave(outfile; output, ground_truth, N, transmit_field, meta) # noiseless phantom
    println("    Saved to: $outfile")
end

MPI.Finalize()
