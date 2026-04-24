# Distributed versions of TrustRegionReflective/utils.jl
# All vector operations use MPI distributed primitives from MPIResources.jl

function mpi_computeDistanceToBoundaries(x, g, LB, UB, comm)
    v = ones(eltype(g), length(g))
    v[ (g .< 0) .& (UB .<  Inf) ] = UB[ (g .< 0) .& (UB .< Inf)  ] -  x[ (g .< 0) .& (UB .< Inf)  ]
    v[ (g .> 0) .& (LB .> -Inf) ] =  x[ (g .> 0) .& (LB .> -Inf) ] - LB[ (g .> 0) .& (LB .> -Inf) ]

    if mpi_any(any(v .< 0), comm)
        println("    Somehow x is not within the bounds")
        return
    end

    dv = zeros(eltype(v), length(v))
    dv[ (g .< 0) .& (UB .<  Inf) ] .= -1
    dv[ (g .> 0) .& (LB .> -Inf) ] .=  1

    return v, dv
end

function mpi_distanceToFeasibleRegion(x, s, LB, UB, comm)
    non_zero = s .!= 0
    steps = Inf * ones(length(x))
    steps[non_zero] = max.( (LB[non_zero] - x[non_zero]) ./ s[non_zero], (UB[non_zero] - x[non_zero]) ./ s[non_zero])
    stepsize = mpi_minimum(steps, comm)
    boundary_hit = (steps .== stepsize)

    return stepsize, boundary_hit
end

function mpi_distanceToTrustRegion(x, p, trustRadius, comm)
    a = mpi_dot(p, p, comm)
    b = 2 * mpi_dot(x, p, comm)
    c = mpi_dot(x, x, comm) - trustRadius^2

    τ = (-b + sqrt( b^2 - 4*a*c) ) / (2*a)
    return τ
end

function mpi_distanceToTrustRegion2(x, s, trust_radius, comm)
    a = mpi_dot(s, s, comm)

    if a == 0
        println("    distanceToTrustRegion2: WTF s is zero")
        return
    end

    b = mpi_dot(x, s, comm)
    c = mpi_dot(x, x, comm) - trust_radius^2

    if c > 0
        println("     distanceToTrustRegion2: WTF")
        return
    end

    d = sqrt(b*b - a*c)
    q = -(b + abs(d) * sign(b))
    t1 = q / a
    t2 = c / q

    if t1 < t2
        t_negative = t1
        t_positive = t2
    else
        t_negative = t2
        t_positive = t1
    end

    return t_negative, t_positive
end

function mpi_adjustTrustRadius(ratio, step, Δ, min_ratio, comm)
    norm_step = mpi_norm(step, comm)
    println("        Norm of step: $(norm_step), Trust Radius: $(Δ)")

    if ratio < min_ratio
        println("    Trust Radius too large")
        Δ = (1/4) * Δ
    elseif (ratio > 1/2) && (norm_step > ( 0.95 * Δ) )
        println("    Trust Radius too small")
        Δ = 2 * Δ
    else
        println("    Trust Radius just fine")
    end

    return Δ
end

function mpi_withinBounds(parameters, LB, UB, comm)
    local_check = all( (parameters .>= LB) .& (parameters .<= UB) )
    return mpi_all(local_check, comm)
end

function mpi_evaluateQuadratic(H, g, s, comm)
    Hs = H(s)
    value = 0.5 * mpi_dot(s, Hs, comm) + mpi_dot(g, s, comm)
    return value
end

function mpi_buildQuadratic1D(H, g, s, s0, comm)
    Hs = H(s)
    a = 0.5 * mpi_dot(s, Hs, comm)

    # Check if s0 is zero to avoid redundant Hv calls
    if iszero(s0)
        b = mpi_dot(g, s, comm)
        c = zero(a)
    else
        Hs0 = H(s0)
        b = mpi_dot(g, s, comm) + mpi_dot(s, Hs0, comm)
        c = mpi_dot(g, s0, comm) + 0.5 * mpi_dot(s0, Hs0, comm)
    end

    return a, b, c
end

# minimizeQuadratic1D is pure scalar — no change needed, use original directly
function minimizeQuadratic1D(a, b, lb, ub, c)
    t = [lb, ub]
    if a != 0
        extremum = -0.5 * b / a
        if ((lb < extremum) && (extremum < ub))
            t = [lb, extremum, ub]
        end
    end

    y = a .* t.^2 .+ b .* t .+ c
    minval = minimum(y)
    index = findfirst(x -> x == minval, y)
    argument = t[index[1]]

    return argument, minval
end

function mpi_chooseStep(x, H_hat, g_hat, gn, gn_hat, D, trust_radius, theta, LB, UB, comm)

    if mpi_withinBounds(x + gn, LB, UB, comm)
        step        = gn
        step_hat    = gn_hat
        step_value  = mpi_evaluateQuadratic(H_hat, g_hat, gn_hat, comm)
        println("          The Inexact Newton step was chosen")
        return step, step_hat, step_value
    end

    # POTENTIAL STEP 1: Reflected Newton
    p_steplength, boundary_hit = mpi_distanceToFeasibleRegion(x, gn, LB, UB, comm)

    rf_hat = gn_hat
    rf_hat[boundary_hit] = -1 * rf_hat[boundary_hit]
    rf = D .* rf_hat

    gn             = p_steplength * gn
    gn_hat         = p_steplength * gn_hat
    x_on_boundary = x + gn

    meh, to_trust = mpi_distanceToTrustRegion2(gn_hat, rf_hat, trust_radius, comm)
    to_feasible, meh = mpi_distanceToFeasibleRegion(x_on_boundary, rf, LB, UB, comm)

    rf_steplength = min(to_trust, to_feasible)

    if rf_steplength > 0
        rf_steplength_l = (1 - theta) * p_steplength / rf_steplength

        if rf_steplength == to_feasible
            rf_steplength_u = theta * to_feasible
        elseif rf_steplength == to_trust
            rf_steplength_u = to_trust
        end
    else
        println("        rf_steplength <= 0? What's going on?")
        rf_steplength_l = 0
        rf_steplength_u = -1
    end

    if rf_steplength_l <= rf_steplength_u
        a, b, c = mpi_buildQuadratic1D(H_hat, g_hat, rf_hat, gn_hat, comm)
        rf_steplength, rf_value = minimizeQuadratic1D(a, b, rf_steplength_l, rf_steplength_u, c)
        rf_hat = rf_hat * rf_steplength
        rf_hat = rf_hat + gn_hat
        rf = D .* rf_hat
    else
        rf_value = Inf
    end

    # POTENTIAL STEP 2: Truncated Newton
    gn       = theta * gn
    gn_hat   = theta * gn_hat
    gn_value  = mpi_evaluateQuadratic(H_hat, g_hat, gn_hat, comm)

    # POTENTIAL STEP 3: Steepest Descent
    sd_hat = -g_hat
    sd = D .* sd_hat

    to_trust         = trust_radius / mpi_norm(sd_hat, comm)
    to_feasible, meh = mpi_distanceToFeasibleRegion(x, sd, LB, UB, comm)

    if to_feasible < to_trust
        sd_steplength_max = theta * to_feasible
    else
        sd_steplength_max = to_trust
    end

    a, b, c = mpi_buildQuadratic1D(H_hat, g_hat, sd_hat, zeros(length(g_hat)), comm)
    sd_steplength, sd_value = minimizeQuadratic1D(a, b, 0, sd_steplength_max, 0)
    sd_hat = sd_steplength * sd_hat
    sd     = sd_steplength * sd

    # Choose the best step
    values = [gn_value rf_value sd_value]
    minVal = minimum(values)
    index = findfirst(y -> y == minVal, values)
    index = index[1]
    println("        gn: $(gn_value), rf: $(rf_value), sd: $(sd_value)")

    if index == 1
        step        = gn
        step_hat    = gn_hat
        step_value  = gn_value
        println("        The Inexact Newton step, restricted to feasible region, was chosen")
    end
    if index == 2
        step        = rf
        step_hat    = rf_hat
        step_value  = rf_value
        println("        The Reflected Inexact Newton step was chosen")
    end
    if index == 3
        step        = sd
        step_hat    = sd_hat
        step_value  = sd_value
        println("        The Steepest Descent step was chosen")
    end

    return step, step_hat, step_value
end
