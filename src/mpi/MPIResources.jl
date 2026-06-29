# MPI + CUDA

using MPI

"""
Each MPI rank binds to one GPU.
"""
struct MPICUDALibs <: AbstractResource{Nothing}
    comm::MPI.Comm
    rank::Int
    nranks::Int
    local_gpu_id::Int
end

function MPICUDALibs(comm::MPI.Comm=MPI.COMM_WORLD)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    local_rank = parse(Int, get(ENV, "OMPI_COMM_WORLD_LOCAL_RANK",
        get(ENV, "SLURM_LOCALID", string(rank))))
    local_gpu_id = local_rank % length(CUDA.devices())
    CUDA.device!(local_gpu_id)
    return MPICUDALibs(comm, rank, nranks, local_gpu_id)
end

is_root(r::MPICUDALibs) = r.rank == 0

function voxel_partition(total::Int, rank::Int, nranks::Int)
    base, rem = divrem(total, nranks)
    if rank < rem
        count = base + 1
        offset = rank * (base + 1)
    else
        count = base
        offset = rem * (base + 1) + (rank - rem) * base
    end
    return (offset+1):(offset+count)
end

function partition_structarray(sa, vr::UnitRange)
    x_local = Array(sa.x)[vr]
    y_local = Array(sa.y)[vr]
    z_local = Array(sa.z)[vr]
    return gpu(f32(StructArray{Coordinates{Float32}}((x_local, y_local, z_local))))
end

function extract_local_optimpars(optimpars::AbstractVector, voxel_range::UnitRange)
    total_voxels = length(optimpars) ÷ 4
    M = reshape(optimpars, total_voxels, 4)
    return vec(M[voxel_range, :])
end

function allreduce_sum!(buf::CuArray, comm::MPI.Comm)
    sendbuf = Array(buf)
    recvbuf = similar(sendbuf)
    MPI.Allreduce!(sendbuf, recvbuf, +, comm)
    copyto!(buf, recvbuf)
end

function allreduce_sum(val::T, comm::MPI.Comm) where {T<:Real}
    result = Ref(val)
    MPI.Allreduce!(Ref(val), result, +, comm)
    return result[]
end

function mpi_allgatherv(local_vec::Vector{T}, comm::MPI.Comm) where {T}
    local_count = Int32(length(local_vec))
    counts = MPI.Allgather(local_count, comm)
    total = sum(counts)
    recv = Vector{T}(undef, total)
    MPI.Allgatherv!(local_vec, VBuffer(recv, counts), comm)
    return recv
end

# Distributed solver helpers

function mpi_dot(a::AbstractVector, b::AbstractVector, comm::MPI.Comm)
    return allreduce_sum(dot(a, b), comm)
end

function mpi_norm(a::AbstractVector, comm::MPI.Comm)
    return sqrt(mpi_dot(a, a, comm))
end

function mpi_norminf(a::AbstractVector, comm::MPI.Comm)
    local_max = length(a) > 0 ? maximum(abs, a) : zero(eltype(a))
    result = Ref(local_max)
    MPI.Allreduce!(Ref(local_max), result, max, comm)
    return result[]
end

function mpi_minimum(a::AbstractVector, comm::MPI.Comm)
    local_min = length(a) > 0 ? minimum(a) : typemax(eltype(a))
    result = Ref(local_min)
    MPI.Allreduce!(Ref(local_min), result, min, comm)
    return result[]
end

function mpi_all(val::Bool, comm::MPI.Comm)
    result = Ref(Int32(val))
    MPI.Allreduce!(Ref(Int32(val)), result, min, comm)
    return result[] != 0
end

function mpi_any(val::Bool, comm::MPI.Comm)
    result = Ref(Int32(val))
    MPI.Allreduce!(Ref(Int32(val)), result, max, comm)
    return result[] != 0
end

function mpi_colnorms(M::AbstractMatrix, comm::MPI.Comm)
    local_sq = [dot(c, c) for c in eachcol(M)]
    global_sq = similar(local_sq)
    MPI.Allreduce!(local_sq, global_sq, +, comm)
    return sqrt.(global_sq)
end

function mpi_gather_field_major(local_vec::Vector{T}, local_nvox::Int, total_nvox::Int, comm::MPI.Comm) where T
    np = length(local_vec) ÷ local_nvox
    vi = vec(permutedims(reshape(local_vec, local_nvox, np)))
    gathered = mpi_allgatherv(vi, comm)
    return vec(permutedims(reshape(gathered, np, total_nvox)))
end
