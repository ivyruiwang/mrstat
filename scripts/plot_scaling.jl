#!/usr/bin/env julia
# ============================================================================
# Plot scaling law: speedup, wall-time, and per-iteration time breakdown
#
# Usage: julia --project=. scripts/plot_scaling.jl results_mpi_1n1g.jld2 results_mpi_1n2g.jld2 ...
# ============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JLD2, Statistics
using PythonPlot
using DelimitedFiles

if isempty(ARGS)
    println("Usage: julia --project=. scripts/plot_scaling.jl <1gpu.jld2> <2gpu.jld2> ...")
    exit(1)
end

# Infer results directory from first .jld2 path
results_dir = dirname(abspath(ARGS[1]))

# ── Load wall-time from .jld2 files ──────────────────────────

struct ScalingPoint
    ngpus::Int
    nnodes::Int
    wall_time::Float64
    label::String
end

points = ScalingPoint[]
for path in ARGS
    d = load(path)
    meta = d["meta"]
    t = d["output"].t[end]
    ngpus = meta["ngpus"]
    nnodes = meta["nnodes"]
    label = "$(nnodes)N$(ngpus)G"
    push!(points, ScalingPoint(ngpus, nnodes, t, label))
end

sort!(points, by=p -> p.ngpus)

outdir = "/home/iwang3/mrstat_main/analysis_results"
mkpath(outdir)

# ── Print & save scaling summary ─────────────────────────────

t_base = points[1].wall_time

println("=== Scaling Results ===")
println("  GPUs | Nodes | Wall-time (s) | Speedup | Efficiency")
println("  ", "-"^55)

open(joinpath(outdir, "scaling_summary.csv"), "w") do io
    println(io, "ngpus,nnodes,wall_time_s,speedup,efficiency_pct")
    for p in points
        speedup = t_base / p.wall_time
        efficiency = speedup / p.ngpus * 100
        println("  $(rpad(p.ngpus, 5))| $(rpad(p.nnodes, 6))| $(rpad(round(p.wall_time, digits=1), 14))| $(rpad(round(speedup, digits=2), 8))| $(round(efficiency, digits=1))%")
        println(io, "$(p.ngpus),$(p.nnodes),$(round(p.wall_time, digits=2)),$(round(speedup, digits=3)),$(round(efficiency, digits=1))")
    end
end
println("  Saved: scaling_summary.csv")

# ── Plot 1: Speedup vs GPU count ─────────────────────────────

ngpus_list = [p.ngpus for p in points]
speedups = [t_base / p.wall_time for p in points]
max_gpu = maximum(ngpus_list)

figure(figsize=(5, 4))
plot(ngpus_list, speedups, "o-", color="tab:blue", label="Measured", linewidth=2, markersize=6)
plot([1, max_gpu], [1, max_gpu], "--", color="gray", label="Ideal (linear)")

for (p, s) in zip(points, speedups)
    display_label = "$(p.nnodes)N$(p.ngpus)G"
    annotate(display_label, (p.ngpus, s), textcoords="offset points", xytext=(5, 5), fontsize=8)
end

xlabel("Number of GPUs"); ylabel("Speedup")
title("Scaling: Speedup vs GPU Count")
xticks(ngpus_list); legend(title="N=Node, G=GPU(s)"); grid(true, alpha=0.3)
tight_layout()
savefig(joinpath(outdir, "scaling_speedup.png"), dpi=600)
println("  Saved: scaling_speedup.png")

# ── Plot 2: Wall-time bar chart ──────────────────────────────

walltimes = [p.wall_time for p in points]
bar_labels = ["$(p.nnodes) Node\n$(p.ngpus) GPU$(p.ngpus > 1 ? "s" : "")" for p in points] 

figure(figsize=(5, 4))
bar(bar_labels, walltimes, color="tab:orange")
ylabel("Wall-time (s)"); xlabel("Configuration")
title("Reconstruction Wall-time")
grid(true, axis="y", alpha=0.3)
tight_layout()
savefig(joinpath(outdir, "scaling_walltime.png"), dpi=600)
println("  Saved: scaling_walltime.png")

# ── Plot 3: Per-iteration breakdown (one figure per config) ──

step_names = ["Objective", "Steihaug-CG", "Step Selection"]
colors = ["tab:blue", "tab:red", "tab:purple"]
# CSV columns: iter(1), t_iter(2), t_obj(3), t_precond(4), t_steihaug(5), t_choose(6), t_evalnew(7)
# Merge t_obj(3) + t_evalnew(7) into "Objective"; drop t_precond(4) (negligible)

# First pass: find global y-axis max across all configs
global_ymax = 0.0
for p in points
    csvname = joinpath(results_dir, "solver_iter_breakdown_$(p.nnodes)n$(p.ngpus)g.csv")
    isfile(csvname) || continue
    data, _ = readdlm(csvname, ',', Float64, '\n'; header=true)
    global global_ymax = max(global_ymax, maximum(data[:, 2]))
end
global_ymax = ceil(global_ymax * 1.15 / 1000) * 1000   # 15% headroom, round up to nearest 1000

# Second pass: plot each config with unified y-axis
for p in points
    csvname = joinpath(results_dir, "solver_iter_breakdown_$(p.nnodes)n$(p.ngpus)g.csv")
    if !isfile(csvname)
        println("  Warning: $csvname not found, skipping breakdown for $(p.label)")
        continue
    end
    data, _ = readdlm(csvname, ',', Float64, '\n'; header=true)
    iters = Int.(data[:, 1])
    bottom = zeros(length(iters))

    # Combine objective + trial point eval; drop preconditioner (negligible)
    step_data = [
        data[:, 3] .+ data[:, 7],   # Objective = t_obj + t_evalnew
        data[:, 5],                   # Steihaug-CG
        data[:, 6],                   # Step Selection
    ]

    figure(figsize=(5, 4))
    for (j, vals) in enumerate(step_data)
        bar(iters, vals, bottom=bottom, color=colors[j], label=step_names[j], width=0.8)
        bottom .+= vals
    end
    ylim(0, global_ymax)
    xlabel("Iteration"); ylabel("Time (ms)")
    title("Per-iteration Breakdown: $(p.nnodes) Node, $(p.ngpus) GPUs")
    xticks(iters); legend(loc="upper right", fontsize=7); grid(true, axis="y", alpha=0.3)
    tight_layout()
    fname = "breakdown_$(p.label).png"
    savefig(joinpath(outdir, fname), dpi=600)
    println("  Saved: $fname")
end

println("\n=== Scaling analysis complete ===")
