#!/usr/bin/env julia

# e.g. MRSTAT_N=320 mpirun julia --project=. scripts/problem_size_scaling/mpi_mrstat_recon_N.jl

using MPI
MPI.Init()

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CUDA
using BlochSimulators
using BlochSimulators: f32, gpu
using MRSTAT
using MRSTAT: NUM_COILS, TrustRegionReflective, DerivativeOperations,
    optim_to_physical_pars
using MRSTAT.DerivativeOperations: simulate_derivatives, Jv, Jᴴv
using ComputationalResources: AbstractResource, CUDALibs
using LinearAlgebra, LinearMaps, StaticArrays, StructArrays, Statistics
using JLD2

include(joinpath(@__DIR__, "..", "..", "src", "mpi", "MPIResources.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "mpi", "mpi_objective.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "mpi", "mpi_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "mpi", "mpi_steihaug.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "mpi", "mpi_solver.jl"))

res = MPICUDALibs()
comm = res.comm
rank = res.rank
nranks = res.nranks

N_size = parse(Int, get(ENV, "MRSTAT_N", "224"))

MPI.Barrier(comm)

using Random;
Random.seed!(42)
raw_data, sequence, coords_full, coils_full, trajectory = MRSTAT.generate_simulation_data(; N=N_size)

total_nvox = length(coords_full)
N = isqrt(total_nvox)

# Add complex Gaussian noise
SNR_dB = 15.36
raw_data_cpu = Array(raw_data)
rms_signal = sqrt(mean(abs.(raw_data_cpu) .^ 2))
rms_noise = Float32(rms_signal / sqrt(10^(SNR_dB / 10)))
Random.seed!(123)   # separate seed for noise (same across all ranks)
noise = rms_noise * randn(ComplexF32, size(raw_data_cpu))
noise_floor = Float64(0.5 * sum(abs.(noise) .^ 2))
raw_data = gpu(raw_data_cpu .+ noise)

vr = voxel_partition(total_nvox, rank, nranks)
local_nvox = length(vr)

local_coords = partition_structarray(coords_full, vr)
local_coils = gpu(f32(Array(coils_full)[vr, :]))
local_tx = ones(Float32, local_nvox)

MPI.Barrier(comm)

x0_per = Float32[log(1.0), log(0.100), 1.0, 0.0]
LB_per = Float32[log(0.1), log(0.001), -Inf, -Inf]
UB_per = Float32[log(7.0), log(3.000), Inf, Inf]

x0 = repeat(x0_per', local_nvox) |> vec
LB = repeat(LB_per', local_nvox) |> vec
UB = repeat(UB_per', local_nvox) |> vec

transmit_field_full = ones(Float32, total_nvox)

plotfun(x, figtitle) = nothing

x0_full = mpi_gather_field_major(x0, local_nvox, total_nvox, comm)

objfun = (x_local, mode) -> mpi_objective(
    x_local, res, mode,
    raw_data, sequence, local_coords, local_coils,
    trajectory, local_tx,
)

trf_opts = TrustRegionReflective.SolverOptions()

rank == 0 && println("Running distributed TRF solver …\n")
MPI.Barrier(comm)

output = mpi_solver(objfun, x0, LB, UB, trf_opts, plotfun, comm, total_nvox)

MPI.Barrier(comm)

if is_root(res)

    Random.seed!(42)
    ground_truth = MRSTAT.make_phantom(N)
    transmit_field = transmit_field_full

    nnodes = parse(Int, get(ENV, "SLURM_NNODES", "1"))
    meta = Dict(
        "ngpus" => nranks,
        "nnodes" => nnodes,
        "solver" => "mpi",
        "N" => N,
        "job_id" => get(ENV, "SLURM_JOB_ID", "local"),
        "hostname" => gethostname(),
    )
    outfile = "results_N$(N)_$(nnodes)n$(nranks)g.jld2"
    jldsave(outfile; output, ground_truth, N, transmit_field, meta, noise_floor)

    csv_old = "solver_iter_breakdown_$(nnodes)n$(nranks)g.csv"
    csv_new = "solver_iter_breakdown_N$(N)_$(nnodes)n$(nranks)g.csv"
    isfile(csv_old) && mv(csv_old, csv_new, force=true)
end

MPI.Finalize()
