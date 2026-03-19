# MPI version of the objective function

function mpi_objective(
    optimpars::Vector{<:Real},
    mpi_res::MPICUDALibs,
    mode::Int,
    raw_data,
    sequence,
    local_coords,
    local_coils,
    trajectory,
    local_transmit,
    voxel_range::UnitRange{Int},
    total_nvox::Int,
)
    comm      = mpi_res.comm
    local_res = CUDALibs()

    # Extract local parameters from optimpars
    local_optim  = extract_local_optimpars(optimpars, voxel_range)
    local_params = optim_to_physical_pars(local_optim, local_transmit)
    local_params = gpu(f32(local_params))

    # Bloch simulation
    echos = simulate_magnetization(local_res, sequence, local_params)

    if mode == 0
        phase_encoding!(echos, trajectory, local_coords)

        # Signal
        s = magnetization_to_signal(local_res, echos, local_params,
                                    trajectory, local_coords, local_coils)
        allreduce_sum!(s, comm)

        # Residual & cost
        r = s - raw_data
        r = reshape(collect(r), :, NUM_COILS)
        r = map(SVector{NUM_COILS}, eachrow(r)) |> gpu
        f = 0.5 * sum(norm.(r) .^ 2)

        return f, r
    end

    # mode 1 or 2: gradient (+ Hessian)
    ∂echos = simulate_derivatives(echos, local_res, sequence, local_params)

    phase_encoding!(echos,  trajectory, local_coords)
    phase_encoding!(∂echos, trajectory, local_coords)

    s = magnetization_to_signal(local_res, echos, local_params,
                                trajectory, local_coords, local_coils)
    allreduce_sum!(s, comm)

    r = s - raw_data
    r = reshape(collect(r), :, NUM_COILS)
    r = map(SVector{NUM_COILS}, eachrow(r)) |> gpu
    f = 0.5 * sum(norm.(r) .^ 2)

    cs_svec = map(SVector{NUM_COILS}, eachrow(collect(local_coils))) |> gpu

    # Gradient
    g_local = Jᴴv(local_res, echos, ∂echos, local_params,
                   cs_svec, trajectory, local_coords, r)
    g_local = StructArray(g_local)
    g_local = reduce(vcat, fieldarrays(g_local))
    g_local = real.(g_local)
    g_local = collect(g_local)
    local_nvox = length(voxel_range)
    g_local = vec(permutedims(reshape(g_local, local_nvox, 4)))
    g = mpi_allgatherv(g_local, comm)
    g = vec(permutedims(reshape(g, 4, total_nvox)))

    mode == 1 && return f, r, g

    # Hessian
    reJᴴJ(x) = begin
        np = 4
        x_loc = extract_local_optimpars(x, voxel_range)
        x_loc = reshape(x_loc, :, np)
        x_loc = map(SVector{np}, eachcol(x_loc)...) |> gpu

        # Jv: sum over voxels -> Allreduce
        y = Jv(local_res, echos, ∂echos, local_params,
               cs_svec, trajectory, local_coords, x_loc)
        allreduce_sum!(y, comm)

        # JHv: per-voxel -> Allgatherv
        z_loc = Jᴴv(local_res, echos, ∂echos, local_params,
                     cs_svec, trajectory, local_coords, y)
        z_loc_vec = real.(reduce(vcat, fieldarrays(StructArray(z_loc))))
        z_loc_vec = collect(z_loc_vec)
        
        local_nvox = length(voxel_range)
        z_loc_vec = vec(permutedims(reshape(z_loc_vec, local_nvox, 4)))
        z_gathered = mpi_allgatherv(z_loc_vec, comm)
        return vec(permutedims(reshape(z_gathered, 4, total_nvox)))
    end

    H = LinearMap(v -> reJᴴJ(v), v -> v, length(g), length(g))
    return f, r, g, H
end
