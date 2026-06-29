
# All vectors are local (local_nvox × 4)

using MRSTAT: TrustRegionReflective
using TickTock

function mpi_solver(objective, x0, LB, UB, options::TrustRegionReflective.SolverOptions,
    plotfun, comm, total_nvox)

    rank = MPI.Comm_rank(comm)
    local_nvox = length(x0) ÷ 4
    total_length = total_nvox * 4

    x = x0

    println("    Calling f,r,g,H = objective(x,2)")
    t0 = time()
    f, r, g, H = objective(x, 2)
    t_obj = (time() - t0) * 1000

    # Initial trust radius
    println("Setting initial trust radius")
    Δ = 0.1 * mpi_norm(x, comm)
    Δlimit = Δ * 1E-10

    iter = 1
    converged = false
    t = 0.0

    x_full = mpi_gather_field_major(x, local_nvox, total_nvox, comm)
    state = TrustRegionReflective.SolverOutput(x_full, f, r, t)

    options.save_every_iter && write_to_disk(state)

    # CSV logging
    iterlog = nothing
    if rank == 0
        nnodes = parse(Int, get(ENV, "SLURM_NNODES", "1"))
        nranks_total = MPI.Comm_size(comm)
        csvname = "solver_iter_breakdown_$(nnodes)n$(nranks_total)g.csv"
        iterlog = open(csvname, "w")
        println(iterlog, "iter,t_iter_ms,t_obj_ms,t_precond_ms,t_steihaug_ms,t_choose_ms,t_evalnew_ms")
    end

    while ((iter < (options.max_iter_trf + 1)) && (!converged))

        println("ITERATION #$(iter)")
        tick()

        t_iter = time()

        if iter > 1
            t0 = time()
            println("    Calling f,r,g,H = objective(x,2)")
            f, r, g, H = objective(x, 2)
            t_obj = (time() - t0) * 1000
        end
        # iter == 1 时 t_obj 保留循环前（第 20 行）的初始 objective 耗时

        println("    f: $(f)",)
        println("    Δ: $(Δ)")

        # Scaling matrices D, C, Ĥ all local
        t0 = time()
        v, dv = mpi_computeDistanceToBoundaries(x, g, LB, UB, comm)

        D = sqrt.(v)
        ĝ = D .* g
        C = dv .* g
        Ĥ = x -> (D .* (H * (D .* x))) + (C .* x)
        t_precond = (time() - t0) * 1000

        step_accepted = false
        perform_steihaug = true
        sh_iter = -1

        steps = zeros(length(x), options.max_iter_steihaug)

        t_steihaug = 0.0
        t_choose = 0.0
        t_evalnew = 0.0

        while !step_accepted

            P = y -> y
            z0 = zeros(length(ĝ))

            # Steihaug CG
            t0 = time()
            if perform_steihaug
                steps = mpi_steihaug(Ĥ, ĝ, Δ, P, options.max_iter_steihaug, options.tol_steihaug, z0, total_length, comm)
                ŝ = steps[:, end]
            else
                ŝ = steps[:, sh_iter]
            end
            t_steihaug += (time() - t0) * 1000

            s = D .* ŝ
            x_new = x + s

            # Choose step
            t0 = time()
            θ = max(0.995, 1 - mpi_norminf(v .* g, comm))
            @info "Choose step"
            @time step, step_hat, step_value = mpi_chooseStep(x, Ĥ, ĝ, s, ŝ, D, Δ, θ, LB, UB, comm)
            t_choose += (time() - t0) * 1000

            x_new = x + step

            # Evaluate new point
            println("    Calling f,r = objective(x_new,0)")
            t0 = time()
            f_new, r_new = objective(x_new, 0)
            t_evalnew += (time() - t0) * 1000

            # Convergence logic
            actualReduction = -(f_new - f)

            Hs = H * s
            gs = mpi_dot(g, s, comm)
            sHs = mpi_dot(s, Hs, comm)
            predictedReduction = -(gs + 0.5 * sHs)

            sCŝ = mpi_dot(ŝ, C .* ŝ, comm)
            modifiedReduction = -(f_new - f + 0.5 * sCŝ)
            ratio = modifiedReduction / predictedReduction

            println("   reduction: $(actualReduction)")
            println("   ratio: $(ratio)")

            if (actualReduction > 0 && ratio > 0.1) || Δ < Δlimit
                println("    Step accepted")
                step_accepted = true

                Δ = mpi_adjustTrustRadius(ratio, ŝ, Δ, options.min_ratio, comm)
                x = x_new
                f = f_new
                r = r_new

                t += tok()

                x_full = mpi_gather_field_major(x, local_nvox, total_nvox, comm)
                state.x = hcat(state.x, x_full)
                state.f = hcat(state.f, f)
                state.r = hcat(state.r, r)
                state.t = hcat(state.t, t)

                perform_steihaug = true
            else
                println("    Find a smaller step (reduction: $(actualReduction)")

                sh_iter = size(steps, 2)

                while sh_iter >= size(steps, 2)
                    Δ = 0.5 * Δ
                    println("   Trust radius reduced to: $(Δ)")

                    col_norms = mpi_colnorms(steps, comm)
                    sh_iter = findlast(col_norms .<= Δ)

                    if sh_iter === nothing
                        sh_iter = 1
                        break
                    end
                end
                if sh_iter == 1
                    perform_steihaug = true
                else
                    perform_steihaug = false
                end
            end
        end # Step accepted

        # CSV logging
        t_iter_ms = (time() - t_iter) * 1000
        if rank == 0
            println(iterlog, "$(iter),$(t_iter_ms),$(t_obj),$(t_precond),$(t_steihaug),$(t_choose),$(t_evalnew)")
        end

        iter += 1

        plotfun(x_full, "Iteration: $iter")
    end

    if rank == 0
        close(iterlog)
    end

    return state
end
