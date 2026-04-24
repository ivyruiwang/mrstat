#!/usr/bin/env julia

using Pkg
Pkg.activate(@__DIR__)   # 激活当前目录的环境

using MRSTAT
using CUDA

using BlochSimulators: f32, gpu 

# simulate data
raw_data, sequence, coordinates, coil_sensitivities, trajectory =
    MRSTAT.generate_simulation_data()

N  = isqrt(length(coordinates))  
Nz = 1

# B1 field
transmit_field = ones(Float32, N, N, Nz)

# reconstruction part
@info "Starting MRSTAT reconstruction" N=N Nz=Nz
output = MRSTAT.mrstat_recon(
    raw_data, sequence, coordinates, coil_sensitivities, trajectory, transmit_field;
    intermediate_plots = true,
)

@info "Reconstruction finished" output=output
