
function mpi_steihaug(H, g, Δ, P, maxit, tol, z0, total_length, comm)

    println("    Steihaug CG:")
    ϵ = eps()

    # Use total_length (global element count) 
    η = min(tol, mpi_norm(g, comm) / total_length)

    tol = η * mpi_norm(g, comm)

    z = zeros(length(g))
    r = g

    Y = P(r)
    d = -Y

    steps = eltype(g)[]
    sizehint!(steps, maxit*length(d))

    if mpi_norm(r, comm) < tol
        println("        Nothing to gain, residual is already small enough from the start")
        append!(steps, z)
    end

    iter = 1

    while iter <= maxit

        print(".")

        Hd = H(d)
        dHd = mpi_dot(d, Hd, comm)

        if dHd < ϵ
            println("        Direction of negative curvature encountered: should not occur because of Gauss-Newton method?")
            τ = mpi_distanceToTrustRegion(z, d, Δ, comm)
            append!(steps, z + τ * d)
            break
        end

        α = mpi_dot(r, Y, comm) / dHd
        z_new = z + α * d

        if mpi_norm(z_new, comm) > Δ
            println("        Fell out of trust radius after iteration $(iter)")
            τ = mpi_distanceToTrustRegion(z, d, Δ, comm)
            append!(steps, z + τ * d)
            break
        end

        r_new = r + α * Hd
        norm_r_new = mpi_norm(r_new, comm)

        if norm_r_new < tol
            println("        Steihaug-CG converged with CG-residual = $(norm_r_new) after iteration $(iter)")
            append!(steps, z_new)
            break
        end

        Y_new = P(r_new)
        β = mpi_dot(Y_new, r_new, comm) / mpi_dot(Y, r, comm)
        d_new = -Y_new + β * d

        r = r_new
        d = d_new
        z = z_new
        Y = Y_new

        if iter == maxit
            println("        Steihaug-CG failed to converge, CG-residual = $(norm_r_new)")
            append!(steps, z_new)
            break
        else
            iter = iter + 1
            append!(steps, z_new)
        end
    end

    steps = reshape(steps, length(g), :)

    return steps
end
