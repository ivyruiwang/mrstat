#!/usr/bin/env julia

# julia --project=. scripts/problem_size_scaling/plot_problem_size.jl <results_dir>

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using JLD2
using PythonPlot

results_dir = ARGS[1]
outdir = joinpath(results_dir, "analysis")
mkpath(outdir)

struct PSResult
    N::Int
    ngpus::Int
    nnodes::Int
    wall_time::Float64
    config_tag::String   # e.g. "1GPU", "1N×2G", "2N×2G"
end

function make_tag(nnodes, ngpus)
    if ngpus == 1
        return "1GPU"
    end
    gpus_per_node = ngpus ÷ nnodes
    return "$(nnodes)N×$(gpus_per_node)G"
end

results = PSResult[]
for f in readdir(results_dir)
    endswith(f, ".jld2") || continue
    d = load(joinpath(results_dir, f))
    meta = d["meta"]
    t = d["output"].t[end]
    nnodes = meta["nnodes"]
    ngpus = meta["ngpus"]
    push!(results, PSResult(meta["N"], ngpus, nnodes, t, make_tag(nnodes, ngpus)))
end

plot_tags = ["1N×2G", "1N×4G", "2N×2G", "2N×4G"]

speedup_results = filter(r -> r.config_tag == "1GPU", results)
filter!(r -> r.config_tag in plot_tags, results)

sort!(results, by=r -> (findfirst(==(r.config_tag), plot_tags), r.N))
sort!(speedup_results, by=r -> r.N)

oom_configs = [
    (N=384, tag="1GPU"),
    (N=448, tag="1GPU"),
    (N=512, tag="1GPU"),
    (N=512, tag="1N×2G"),
]

open(joinpath(outdir, "problem_size_summary.csv"), "w") do io
    println(io, "N,N_squared,config,ngpus,nnodes,wall_time_s,speedup_vs_1gpu")

    for r in speedup_results
        println(io, "$(r.N),$(r.N^2),$(r.config_tag),$(r.ngpus),$(r.nnodes),$(round(r.wall_time, digits=2)),1.0")
    end

    for r in results
        t1 = nothing
        for r2 in speedup_results
            if r2.N == r.N
                t1 = r2.wall_time
                break
            end
        end
        speedup_str = t1 !== nothing ? "$(round(t1 / r.wall_time, digits=3))" : ""
    end
end

for r in results
    t1 = nothing
    for r2 in speedup_results
        if r2.N == r.N
            t1 = r2.wall_time
            break
        end
    end
    sp = t1 !== nothing ? string(round(t1 / r.wall_time, digits=2)) : "-"
end

config_styles = Dict(
    "1N×2G" => (marker="o-", color="tab:blue", label="1 Node × 2 GPUs (shared memory)"),
    "1N×4G" => (marker="s-", color="tab:green", label="1 Node × 4 GPUs (shared memory)"),
    "2N×2G" => (marker="d--", color="tab:orange", label="2 Nodes × 2 GPUs (TCP)"),
    "2N×4G" => (marker="^--", color="tab:purple", label="2 Nodes × 4 GPUs (TCP)"),
)

baseline_tag = "1N×2G"
p_baseline = 2

baseline_results = filter(r -> r.config_tag == baseline_tag, results)
t_baseline_by_N = Dict(r.N => r.wall_time for r in baseline_results)

all_Ns = sort([N for N in unique(r.N for r in results) if N != 128])
xticks_pos = [Float64(N^2) for N in all_Ns]
xticks_lbl = ["$(N)²" for N in all_Ns]

sp_baseline_Ns = sort([N for N in keys(t_baseline_by_N) if N != 128])
sp_xticks_pos = [Float64(N^2) for N in sp_baseline_Ns]
sp_xticks_lbl = ["$(N)²" for N in sp_baseline_Ns]

figure(figsize=(7, 5))
for tag in plot_tags
    subset = filter(r -> r.config_tag == tag && r.N != 128, results)
    isempty(subset) && continue
    sort!(subset, by=r -> r.N)
    Ns = [Float64(r.N^2) for r in subset]
    ts = [r.wall_time for r in subset]
    style = config_styles[tag]
    plot(Ns, ts, style.marker, color=style.color, label=style.label, linewidth=2, markersize=6)
end

ymax_data = maximum(r.wall_time for r in results if r.N != 128)
wt_step =
    ymax_data < 200 ? 50.0 :
    ymax_data < 500 ? 100.0 :
    ymax_data < 1500 ? 250.0 :
    ymax_data < 3000 ? 500.0 :
    1000.0
wt_ymax = ceil(ymax_data / wt_step) * wt_step
ylim(0, wt_ymax)
yticks(collect(0:wt_step:wt_ymax))

xticks(xticks_pos, xticks_lbl)
xlabel("Problem size (image dimension)");
ylabel("Wall-time (s)")
title("Wall-time vs Problem Size")
legend(fontsize=9);
grid(true, alpha=0.3)
tight_layout()
savefig(joinpath(outdir, "walltime_vs_N2.png"), dpi=600)

figure(figsize=(7, 5))
for tag in plot_tags
    tag == baseline_tag && continue    # skip baseline 
    subset = filter(r -> r.config_tag == tag, results)
    isempty(subset) && continue
    sort!(subset, by=r -> r.N)
    speedups_r = Float64[]
    Ns = Float64[]
    for r in subset
        r.N == 128 && continue
        haskey(t_baseline_by_N, r.N) || continue
        push!(speedups_r, t_baseline_by_N[r.N] / r.wall_time)
        push!(Ns, Float64(r.N^2))
    end
    isempty(Ns) && continue
    style = config_styles[tag]
    plot(Ns, speedups_r, style.marker, color=style.color, label=style.label, linewidth=2, markersize=6)
end

xticks(sp_xticks_pos, sp_xticks_lbl)
xlabel("Problem size (image dimension)")
ylabel("Relative speedup (vs 1N×2G)")
title("Speedup vs Problem Size")
legend(fontsize=9);
grid(true, alpha=0.3)
tight_layout()
savefig(joinpath(outdir, "speedup_vs_N2.png"), dpi=600)

figure(figsize=(7, 5))
for tag in plot_tags
    tag == baseline_tag && continue    # skip baseline
    subset = filter(r -> r.config_tag == tag, results)
    isempty(subset) && continue
    sort!(subset, by=r -> r.N)
    effs = Float64[]
    Ns = Float64[]
    for r in subset
        r.N == 128 && continue
        haskey(t_baseline_by_N, r.N) || continue
        t_base = t_baseline_by_N[r.N]
        speedup_r = t_base / r.wall_time
        eff_r = speedup_r / (r.ngpus / p_baseline)
        push!(effs, eff_r)
        push!(Ns, Float64(r.N^2))
    end
    isempty(Ns) && continue
    style = config_styles[tag]
    plot(Ns, effs, style.marker, color=style.color, label=style.label, linewidth=2, markersize=6)
end
axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Ideal (E=1)")

xticks(sp_xticks_pos, sp_xticks_lbl)
xlabel("Problem size (image dimension)")
ylabel("Parallel efficiency (E)")
title("Parallel Efficiency vs Problem Size")
ylim(0, 1.15);
legend(fontsize=9, loc="lower right");
grid(true, alpha=0.3)
tight_layout()
savefig(joinpath(outdir, "efficiency_vs_N2.png"), dpi=600)

