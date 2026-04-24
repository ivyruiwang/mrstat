#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using JLD2
using PythonPlot

if isempty(ARGS)
    println("Usage: julia --project=. scripts/problem_size_scaling/plot_problem_size.jl <results_dir>")
    exit(1)
end

results_dir = ARGS[1]
outdir = joinpath(results_dir, "analysis")
mkpath(outdir)


struct PSResult
    N::Int
    ngpus::Int
    nnodes::Int
    wall_time::Float64
end

results = PSResult[]
for f in readdir(results_dir)
    endswith(f, ".jld2") || continue
    d = load(joinpath(results_dir, f))
    meta = d["meta"]
    t = d["output"].t[end]
    push!(results, PSResult(meta["N"], meta["ngpus"], meta["nnodes"], t))
end

sort!(results, by=r -> (r.ngpus, r.N))

println("=== Problem Size Scaling Results ===")
println("Found $(length(results)) result files")


gpu_counts = sort(unique(r.ngpus for r in results))
println("  GPU configs: ", gpu_counts)

open(joinpath(outdir, "problem_size_summary.csv"), "w") do io
    println(io, "N,N_squared,ngpus,nnodes,wall_time_s,speedup")
    for r in results
        t1 = nothing
        for r2 in results
            if r2.N == r.N && r2.ngpus == 1
                t1 = r2.wall_time
                break
            end
        end
        speedup = t1 !== nothing ? t1 / r.wall_time : NaN
        println(io, "$(r.N),$(r.N^2),$(r.ngpus),$(r.nnodes),$(round(r.wall_time, digits=2)),$(round(speedup, digits=3))")
    end
end
println("  Saved: problem_size_summary.csv")

println("\n  N      | N²      | GPUs | Nodes | Wall-time (s) | Speedup")
println("  ", "-"^65)
for r in results
    t1 = nothing
    for r2 in results
        if r2.N == r.N && r2.ngpus == 1
            t1 = r2.wall_time
            break
        end
    end
    speedup = t1 !== nothing ? t1 / r.wall_time : NaN
    println("  $(rpad(r.N, 7))| $(rpad(r.N^2, 8))| $(rpad(r.ngpus, 5))| $(rpad(r.nnodes, 6))| $(rpad(round(r.wall_time, digits=1), 14))| $(round(speedup, digits=2))")
end

markers = ["o-", "s--", "^-.", "d:", "v-"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

figure(figsize=(6, 5))
for (i, ng) in enumerate(gpu_counts)
    subset = filter(r -> r.ngpus == ng, results)
    sort!(subset, by=r -> r.N)
    Ns = [r.N^2 for r in subset]
    ts = [r.wall_time for r in subset]
    label = ng == 1 ? "1 GPU" : "$ng GPUs"
    plot(Ns, ts, markers[mod1(i, length(markers))], color=colors[mod1(i, length(colors))],
         label=label, linewidth=2, markersize=6)
end
xlabel("Problem size (N²)"); ylabel("Wall-time (s)")
title("Wall-time vs Problem Size")
legend(); grid(true, alpha=0.3)
tight_layout()
savefig(joinpath(outdir, "walltime_vs_N2.png"), dpi=600)
println("\n  Saved: walltime_vs_N2.png")

figure(figsize=(6, 5))
for (i, ng) in enumerate(gpu_counts)
    ng == 1 && continue
    subset = filter(r -> r.ngpus == ng, results)
    sort!(subset, by=r -> r.N)
    speedups = Float64[]
    Ns = Int[]
    for r in subset
        t1 = nothing
        for r2 in results
            if r2.N == r.N && r2.ngpus == 1
                t1 = r2.wall_time
                break
            end
        end
        if t1 !== nothing
            push!(speedups, t1 / r.wall_time)
            push!(Ns, r.N^2)
        end
    end
    label = "$ng GPUs"
    plot(Ns, speedups, markers[mod1(i, length(markers))], color=colors[mod1(i, length(colors))],
         label=label, linewidth=2, markersize=6)
end
xlabel("Problem size (N²)"); ylabel("Speedup (vs 1 GPU)")
title("Speedup vs Problem Size")
legend(); grid(true, alpha=0.3)
tight_layout()
savefig(joinpath(outdir, "speedup_vs_N2.png"), dpi=600)
println("  Saved: speedup_vs_N2.png")

println("\n=== Problem size scaling analysis complete ===")
