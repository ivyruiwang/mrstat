#!/usr/bin/env julia

# N is read from environment variable MRSTAT_N (default 224)


using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using MRSTAT
using MRSTAT: TrustRegionReflective, optim_to_physical_pars
using CUDA
using BlochSimulators: f32, gpu
using ComputationalResources: CUDALibs
using LinearAlgebra, StaticArrays, StructArrays, Statistics
using Random
using JLD2

N_size = parse(Int, get(ENV, "MRSTAT_N", "224"))

Random.seed!(42)

println("Single-GPU Problem Size Scaling ")
println("GPU: $(CUDA.name(CUDA.device()))")
println(" N = $N_size")


println("\nGenerating simulation data (N=$N_size)")
raw_data, sequence, coords, coils, trajectory = MRSTAT.generate_simulation_data(; N=N_size)

total_nvox = length(coords)
N = isqrt(total_nvox)
println("Image: $N x $N ($total_nvox voxels)")

Random.seed!(42)
ground_truth = MRSTAT.make_phantom(N)

transmit_field = ones(Float32, total_nvox)

# Add complex Gaussian noise
SNR_dB = 15.36
raw_data_cpu = Array(raw_data)
rms_signal = sqrt(mean(abs.(raw_data_cpu) .^ 2))
rms_noise = Float32(rms_signal / sqrt(10^(SNR_dB / 10)))
Random.seed!(123)  
noise = rms_noise * randn(ComplexF32, size(raw_data_cpu))
noise_floor = Float64(0.5 * sum(abs.(noise) .^ 2))
raw_data = gpu(raw_data_cpu .+ noise)
println(" Noise: SNR_dB=$SNR_dB, noise_floor=$noise_floor")

# Reconstruction
println("\nRunning single-GPU TRF solver ...\n")

x0_per = Float32[log(1.0), log(0.100), 1.0, 0.0]
LB_per = Float32[log(0.1), log(0.001), -Inf, -Inf]
UB_per = Float32[log(7.0), log(3.000),  Inf,  Inf]
x0 = repeat(x0_per', total_nvox) |> vec
LB = repeat(LB_per', total_nvox) |> vec
UB = repeat(UB_per', total_nvox) |> vec

resource = CUDALibs()
objfun = (x, mode) -> MRSTAT.objective(x, resource, mode, raw_data, sequence, coords, coils, trajectory, transmit_field)

plotfun(x, figtitle) = nothing

output = TrustRegionReflective.solver(objfun, x0, LB, UB, TrustRegionReflective.SolverOptions(), plotfun)

outfile = "results_N$(N)_1gpu.jld2"
meta = Dict(
    "ngpus"    => 1,
    "nnodes"   => 1,
    "solver"   => "single_gpu",
    "N"        => N,
    "job_id"   => get(ENV, "SLURM_JOB_ID", "local"),
    "hostname" => gethostname(),
)
jldsave(outfile; output, ground_truth, N, transmit_field, meta, noise_floor)

println("\nDone")
println("Final cost: $(output.f[end])")
println("Iterations: $(size(output.f, 2) - 1)")
println("Wall-time: $(output.t[end]) s")
println("Saved to: $outfile")
