# Hard-coded to simulate T₁ and T₂ derivatives using finite differences.
function simulate_derivatives(m, resource, sequence, parameters, Δ = 10^-4 )

    Δ = Float32(Δ)
    # parameters = collect(parameters)
    base_T₁ = collect(parameters.T₁)
    base_T₂ = collect(parameters.T₂)
    base_B₁ = collect(parameters.B₁)


    # derivatives w.r.t. T₁
    Δpars = map((T₁,T₂, B₁) -> T₁T₂B₁(T₁+Δ, T₂, B₁), base_T₁, base_T₂, base_B₁) |> gpu
    Δm = simulate_magnetization(resource, sequence, Δpars)
    ∂m∂T₁ = @. (Δm - m)/Δ

    # derivatives w.r.t. T₂
    Δpars = map((T₁,T₂, B₁) -> T₁T₂B₁(T₁, T₂+Δ, B₁), base_T₁, base_T₂, base_B₁) |> gpu
    Δm = simulate_magnetization(resource, sequence, Δpars)
    ∂m∂T₂ = @. (Δm - m)/Δ

    ∂m = map((∂T₁,∂T₂)-> ∂mˣʸ∂T₁T₂(∂T₁,∂T₂), ∂m∂T₁, ∂m∂T₂)

    return ∂m
end