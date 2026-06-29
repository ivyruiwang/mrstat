#!/usr/bin/env julia

# Plot scaling: speedup, wall-time, and per-iteration time breakdown
# julia --project=. scripts/plot_scaling.jl xxx.jld2 yyy.jld2 ...

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JLD2, Statistics
using PythonPlot
using DelimitedFiles

results_dir = dirname(abspath(ARGS[1]))

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

t_base = points[1].wall_time

open(joinpath(outdir, "scaling_summary.csv"), "w") do io
    for p in points
        speedup = t_base / p.wall_time
        efficiency = speedup / p.ngpus * 100
    end
end
println("Saved: scaling_summary.csv")

# Keep 2N6G in scaling_summary.csv but excludes it from figures
filter!(p -> p.label != "2N6G", points)

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

xlabel("Number of GPUs");
ylabel("Speedup")
title("Scaling: Speedup vs GPU Count")
xticks(ngpus_list);
legend(title="N=Node, G=GPU(s)");
grid(true, alpha=0.3)
tight_layout()
savefig(joinpath(outdir, "scaling_speedup.png"), dpi=600)
println("Saved: scaling_speedup.png")

efficiencies = [s / n for (s, n) in zip(speedups, ngpus_list)]

figure(figsize=(5, 4))
plot(ngpus_list, efficiencies, "o-", color="tab:green", label="Measured", linewidth=2, markersize=6)
axhline(1.0, color="gray", linestyle="--", alpha=0.6, label="Ideal (E=1)")

for (p, e) in zip(points, efficiencies)
    display_label = "$(p.nnodes)N$(p.ngpus)G"
    annotate(display_label, (p.ngpus, e), textcoords="offset points", xytext=(5, 5), fontsize=8)
end

xlabel("Number of GPUs");
ylabel("Parallel efficiency (E = S/p)")
title("Scaling: Parallel Efficiency vs GPU Count")
xticks(ngpus_list);
ylim(0, 1.15);
legend(title="N=Node, G=GPU(s)");
grid(true, alpha=0.3)
tight_layout()
savefig(joinpath(outdir, "scaling_efficiency.png"), dpi=600)
println("Saved: scaling_efficiency.png")

walltimes = [p.wall_time for p in points]
bar_labels = ["$(p.nnodes) Node\n$(p.ngpus) GPU$(p.ngpus > 1 ? "s" : "")" for p in points]

figure(figsize=(5, 4))
bar(bar_labels, walltimes, color="tab:orange")
ylabel("Wall-time (s)");
xlabel("Configuration")
title("Reconstruction Wall-time")
grid(true, axis="y", alpha=0.3)
tight_layout()
savefig(joinpath(outdir, "scaling_walltime.png"), dpi=600)
println("Saved: scaling_walltime.png")

step_names = ["Objective", "Steihaug-CG", "Step Selection"]
colors = ["tab:blue", "tab:red", "tab:purple"]

global_ymax = 0.0
for p in points
    csvname = joinpath(results_dir, "solver_iter_breakdown_$(p.nnodes)n$(p.ngpus)g.csv")
    isfile(csvname) || continue
    data, _ = readdlm(csvname, ',', Float64, '\n'; header=true)
    global global_ymax = max(global_ymax, maximum(data[:, 2]))
end
global_ymax = ceil(global_ymax * 1.15 / 1000) * 1000   # 15% headroom, round up to nearest 1000

for p in points
    csvname = joinpath(results_dir, "solver_iter_breakdown_$(p.nnodes)n$(p.ngpus)g.csv")
    if !isfile(csvname)
        continue
    end
    data, _ = readdlm(csvname, ',', Float64, '\n'; header=true)
    iters = Int.(data[:, 1])
    bottom = zeros(length(iters))

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
    xlabel("Iteration");
    ylabel("Time (ms)")
    title("Per-iteration Breakdown: $(p.nnodes) Node, $(p.ngpus) GPUs")
    xticks(iters);
    legend(loc="upper right", fontsize=7);
    grid(true, axis="y", alpha=0.3)
    tight_layout()
    fname = "breakdown_$(p.label).png"
    savefig(joinpath(outdir, fname), dpi=600)
    println("Saved: $fname")
end

mean_obj = Float64[]
mean_cg = Float64[]
mean_choose = Float64[]
config_lbls = String[]

for p in points
    csvname = joinpath(results_dir, "solver_iter_breakdown_$(p.nnodes)n$(p.ngpus)g.csv")
    isfile(csvname) || continue
    data, _ = readdlm(csvname, ',', Float64, '\n'; header=true)
    push!(mean_obj, mean(data[:, 3] .+ data[:, 7]))
    push!(mean_cg, mean(data[:, 5]))
    push!(mean_choose, mean(data[:, 6]))
    push!(config_lbls, p.label)
end

if !isempty(config_lbls)
    figure(figsize=(7, 5))
    x_pos = collect(1:length(config_lbls))
    bar(x_pos, mean_obj, color=colors[1], label=step_names[1])
    bar(x_pos, mean_cg, bottom=mean_obj, color=colors[2], label=step_names[2])
    bar(x_pos, mean_choose, bottom=mean_obj .+ mean_cg, color=colors[3], label=step_names[3])

    totals = mean_obj .+ mean_cg .+ mean_choose
    for (i, total) in enumerate(totals)
        if mean_obj[i] / total > 0.005
            text(x_pos[i], mean_obj[i] / 2,
                "$(round(Int, 100 * mean_obj[i] / total))%",
                ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        end
        if mean_cg[i] / total > 0.005
            text(x_pos[i], mean_obj[i] + mean_cg[i] / 2,
                "$(round(Int, 100 * mean_cg[i] / total))%",
                ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        end
        if mean_choose[i] / total > 0.005
            text(x_pos[i], mean_obj[i] + mean_cg[i] + mean_choose[i] / 2,
                "$(round(Int, 100 * mean_choose[i] / total))%",
                ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        end
    end

    xticks(x_pos, config_lbls)
    xlabel("Configuration");
    ylabel("Mean time per iteration (ms)")
    title("Mean Per-iteration Breakdown across Configurations")
    legend(loc="upper right", fontsize=9);
    grid(true, axis="y", alpha=0.3)
    tight_layout()
    savefig(joinpath(outdir, "scaling_breakdown_summary.png"), dpi=600)
    println("Saved: scaling_breakdown_summary.png")
end

println("Scaling analysis completed")
