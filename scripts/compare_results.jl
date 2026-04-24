#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JLD2, Statistics, LinearAlgebra, StructArrays
using PythonPlot
using MRSTAT: optim_to_physical_pars
using QMRIColors
using ImagePhantoms



struct ReconResult
    output
    ground_truth  
    N::Int
    transmit_field
    meta::Dict
    label::String
    noise_floor::Float64
end

function load_result(path)
    d = load(path)
    meta = d["meta"]
    ngpus = meta["ngpus"]
    label = meta["solver"] == "single_gpu" ? "Single GPU" : "$(ngpus) GPU$(ngpus > 1 ? "s" : "")"
    nf = haskey(d, "noise_floor") ? Float64(d["noise_floor"]) : 0.0
    return ReconResult(d["output"], d["ground_truth"], d["N"], d["transmit_field"], meta, label, nf)
end

if isempty(ARGS)
    println("Usage: julia --project=. scripts/compare_results.jl <result1.jld2> <result2.jld2> ...")
    exit(1)
end

results = [load_result(f) for f in ARGS]
N = results[1].N
gt = results[1].ground_truth   

outdir = "/home/iwang3/mrstat_main/analysis_results"
mkpath(outdir)

println("=== Reconstruction Comparison ===")
println("    Image: $N x $N")
println("    Configs: ", join([r.label for r in results], ", "))
println("    Output:  $outdir/")


sl = shepp_logan(N, SheppLoganBrainWeb()) |> rotr90
unique_vals = unique(sl)
tissue_rois = [(val, findall(sl .== val)) for val in unique_vals if val != 0]
roi_mask = vcat([roi for (_, roi) in tissue_rois]...)
println("    ROIs: $(length(tissue_rois)) tissue types, $(length(roi_mask)) voxels")


function extract_params(x_all, transmit_field, N, iter)
    x = x_all[:, iter]
    phys = optim_to_physical_pars(x, transmit_field)
    return reshape(phys, N, N)
end

function rmsre(recon, truth, mask)
    rel_err = (recon[mask] .- truth[mask]) ./ truth[mask]
    return sqrt(mean(rel_err .^ 2))
end

function abs_rel_error_map(recon, truth, mask)
    err = zeros(size(truth))
    err[mask] = abs.(recon[mask] .- truth[mask]) ./ truth[mask]
    return err
end

function mare(recon, truth, mask)
    return mean(abs.(recon[mask] .- truth[mask]) ./ truth[mask])
end

function mean_tnr(recon_map, rois)
    tnr_vals = Float64[]
    for (_, roi) in rois
        m = mean(recon_map[roi])
        s = std(recon_map[roi])
        s > 0 && push!(tnr_vals, m / s)
    end
    return isempty(tnr_vals) ? 0.0 : mean(tnr_vals)
end

function compute_curves(res, gt, roi_mask, tissue_rois)
    n_iters = size(res.output.f, 2)
    rmsre_T1 = zeros(n_iters)
    rmsre_T2 = zeros(n_iters)
    tnr_T1   = zeros(n_iters)
    tnr_T2   = zeros(n_iters)
    for k in 1:n_iters
        p = extract_params(res.output.x, res.transmit_field, res.N, k)
        rmsre_T1[k] = rmsre(p.T₁, gt.T₁, roi_mask)
        rmsre_T2[k] = rmsre(p.T₂, gt.T₂, roi_mask)
        tnr_T1[k]   = mean_tnr(p.T₁, tissue_rois)
        tnr_T2[k]   = mean_tnr(p.T₂, tissue_rois)
    end
    return (; rmsre_T1, rmsre_T2, tnr_T1, tnr_T2)
end

all_curves = [compute_curves(r, gt, roi_mask, tissue_rois) for r in results]


println("\n RMSRE ")
for (r, c) in zip(results, all_curves)
    best_T1 = argmin(c.rmsre_T1)
    best_T2 = argmin(c.rmsre_T2)
    println("\n$(r.label):")
    println("T1 RMSRE: min=$(round(c.rmsre_T1[best_T1], digits=5)) @iter $(best_T1-1), final=$(round(c.rmsre_T1[end], digits=5))")
    println("T2 RMSRE: min=$(round(c.rmsre_T2[best_T2], digits=5)) @iter $(best_T2-1), final=$(round(c.rmsre_T2[end], digits=5))")
end

println("\nEfficiency (TnNR) ")
for (r, c) in zip(results, all_curves)
    best_T1 = argmin(c.rmsre_T1)
    best_T2 = argmin(c.rmsre_T2)
    println("\n$(r.label):")
    println(" T1 TnNR: @best=$(round(c.tnr_T1[best_T1], digits=1)), @final=$(round(c.tnr_T1[end], digits=1))")
    println(" T2 TnNR: @best=$(round(c.tnr_T2[best_T2], digits=1)), @final=$(round(c.tnr_T2[end], digits=1))")
end

markers = ["o-", "x--", "s-.", "d:", "^-", "v--"]
T_scan = 11.2   
sqrt_Tscan = sqrt(T_scan)
max_iter = maximum(size(r.output.f, 2) - 1 for r in results)

opt_T1 = argmin(all_curves[1].rmsre_T1) - 1   
opt_T2 = argmin(all_curves[1].rmsre_T2) - 1

nf = results[1].noise_floor

println("\nOptimal Iterations")
println("T1: iteration $opt_T1 (RMSRE=$(round(all_curves[1].rmsre_T1[opt_T1+1], digits=5)))")
println("T2: iteration $opt_T2 (RMSRE=$(round(all_curves[1].rmsre_T2[opt_T2+1], digits=5)))")
if nf > 0
    println("Noise floor: $nf")
end

figure(figsize=(5, 4))
for (i, r) in enumerate(results)
    costs = vec(r.output.f)
    iters = 0:(length(costs)-1)
    semilogy(iters, costs, markers[mod1(i, length(markers))], label=r.label, markersize=4)
end
if nf > 0
    axhline(y=nf, color="gray", linewidth=1.5, label="Noise floor (½‖η‖²)")
end

opt_cost = vec(results[1].output.f)[opt_T1+1]
plot(opt_T1, opt_cost, "X", color="black", markersize=8, markeredgewidth=2, label="Optimal (iter $opt_T1)", zorder=10)
xlabel("Iteration"); ylabel("Cost (a.u.)")
title("Cost vs Iteration")
xticks(0:max_iter); legend(); grid(true, alpha=0.3)
tight_layout()
savefig(joinpath(outdir, "cost_vs_iteration.png"), dpi=600)
println("\nSaved: cost_vs_iteration.png")

rmsre_all_vals = vcat([vcat(c.rmsre_T1, c.rmsre_T2) for c in all_curves]...)
rmsre_ymin = minimum(rmsre_all_vals) * 0.9
rmsre_ymax = maximum(rmsre_all_vals) * 1.1

eff_all_vals = vcat([vcat(c.tnr_T1[2:end] ./ sqrt_Tscan, c.tnr_T2[2:end] ./ sqrt_Tscan) for c in all_curves]...)
eff_ymin = minimum(eff_all_vals) * 0.9
eff_ymax = maximum(eff_all_vals) * 1.1

figure(figsize=(5, 4))
for (i, (r, c)) in enumerate(zip(results, all_curves))
    iters = 0:(length(c.rmsre_T1)-1)
    plot(iters, c.rmsre_T1, markers[mod1(i, length(markers))], label=r.label, markersize=4)
end
plot(opt_T1, all_curves[1].rmsre_T1[opt_T1+1], "X", color="black", markersize=8, markeredgewidth=2, label="Optimal (iter $opt_T1)", zorder=10)
ylim(rmsre_ymin, rmsre_ymax)
xlabel("Iteration"); ylabel("RMSRE (a.u.)")
title("T1 RMSRE vs Iteration")
xticks(0:max_iter); legend(); grid(true, alpha=0.3)
tight_layout()
savefig(joinpath(outdir, "rmsre_T1_vs_iteration.png"), dpi=600)
println("Saved: rmsre_T1_vs_iteration.png")

figure(figsize=(5, 4))
for (i, (r, c)) in enumerate(zip(results, all_curves))
    iters = 0:(length(c.rmsre_T2)-1)
    plot(iters, c.rmsre_T2, markers[mod1(i, length(markers))], label=r.label, markersize=4)
end
plot(opt_T2, all_curves[1].rmsre_T2[opt_T2+1], "X", color="black", markersize=8, markeredgewidth=2, label="Optimal (iter $opt_T2)", zorder=10)
ylim(rmsre_ymin, rmsre_ymax)
xlabel("Iteration"); ylabel("RMSRE (a.u.)")
title("T2 RMSRE vs Iteration")
xticks(0:max_iter); legend(); grid(true, alpha=0.3)
tight_layout()
savefig(joinpath(outdir, "rmsre_T2_vs_iteration.png"), dpi=600)
println("Saved: rmsre_T2_vs_iteration.png")

figure(figsize=(5, 4))
for (i, (r, c)) in enumerate(zip(results, all_curves))
    iters = 1:(length(c.tnr_T1)-1)
    plot(collect(iters), c.tnr_T1[2:end] ./ sqrt_Tscan, markers[mod1(i, length(markers))], label=r.label, markersize=4)
end
eff_opt_T1 = all_curves[1].tnr_T1[opt_T1+1] / sqrt_Tscan
plot(opt_T1, eff_opt_T1, "X", color="black", markersize=8, markeredgewidth=2, label="Optimal (iter $opt_T1)", zorder=10)
ylim(eff_ymin, eff_ymax)
xlabel("Iteration"); ylabel("Efficiency (a.u.)")
title("T1 Efficiency vs Iteration")
xticks(1:max_iter); legend(); grid(true, alpha=0.3)
tight_layout()
savefig(joinpath(outdir, "efficiency_T1_vs_iteration.png"), dpi=600)
println(" Saved: efficiency_T1_vs_iteration.png")

figure(figsize=(5, 4))
for (i, (r, c)) in enumerate(zip(results, all_curves))
    iters = 1:(length(c.tnr_T2)-1)
    plot(collect(iters), c.tnr_T2[2:end] ./ sqrt_Tscan, markers[mod1(i, length(markers))], label=r.label, markersize=4)
end
eff_opt_T2 = all_curves[1].tnr_T2[opt_T2+1] / sqrt_Tscan
plot(opt_T2, eff_opt_T2, "X", color="black", markersize=8, markeredgewidth=2, label="Optimal (iter $opt_T2)", zorder=10)
ylim(eff_ymin, eff_ymax)
xlabel("Iteration"); ylabel("Efficiency (a.u.)")
title("T2 Efficiency vs Iteration")
xticks(1:max_iter); legend(); grid(true, alpha=0.3)
tight_layout()
savefig(joinpath(outdir, "efficiency_T2_vs_iteration.png"), dpi=600)
println("Saved: efficiency_T2_vs_iteration.png")

cmap_lipari = PythonPlot.ColorMap("lipari",
    QMRIColors.relaxationColorMap("T1"),
    length(QMRIColors.relaxationColorMap("T1")), 1.0)
cmap_navia = PythonPlot.ColorMap("navia",
    QMRIColors.relaxationColorMap("T2"),
    length(QMRIColors.relaxationColorMap("T2")), 1.0)

figure(figsize=(4, 4))
imshow(gt.T₁, clim=(0, 2.5), cmap=cmap_lipari, aspect="equal")
title("Ground Truth T1 [s]"); colorbar()
tight_layout()
savefig(joinpath(outdir, "ground_truth_T1.png"), dpi=600)
println("Saved: ground_truth_T1.png")

figure(figsize=(4, 4))
imshow(gt.T₂, clim=(0, 0.35), cmap=cmap_navia, aspect="equal")
title("Ground Truth T2 [s]"); colorbar()
tight_layout()
savefig(joinpath(outdir, "ground_truth_T2.png"), dpi=600)
println(" Saved: ground_truth_T2.png")

for r in results
    n_iters = size(r.output.f, 2)
    p = extract_params(r.output.x, r.transmit_field, r.N, n_iters)

    err_T1 = abs_rel_error_map(p.T₁, gt.T₁, roi_mask)
    err_T2 = abs_rel_error_map(p.T₂, gt.T₂, roi_mask)
    mare_T1 = mare(p.T₁, gt.T₁, roi_mask)
    mare_T2 = mare(p.T₂, gt.T₂, roi_mask)

    figure(figsize=(8, 8))
    suptitle("$(r.label) — Iteration $(n_iters-1)")

    subplot(2, 2, 1); imshow(p.T₁, clim=(0, 2.5), cmap=cmap_lipari)
    title("Reconstructed T1 [s]"); colorbar()
    subplot(2, 2, 2); imshow(err_T1 .* 100, clim=(0, 10), cmap="hot")
    title("Rel. Error (MARE=$(round(mare_T1*100, digits=2))%)"); colorbar()

    subplot(2, 2, 3); imshow(p.T₂, clim=(0, 0.35), cmap=cmap_navia)
    title("Reconstructed T2 [s]"); colorbar()
    subplot(2, 2, 4); imshow(err_T2 .* 100, clim=(0, 10), cmap="hot")
    title("Rel. Error (MARE=$(round(mare_T2*100, digits=2))%)"); colorbar()

    tight_layout()
    fname = "parammap_$(replace(lowercase(r.label), " " => "_")).png"
    savefig(joinpath(outdir, fname), dpi=600)
    println("Saved: $fname")
end

# Cost vs iteration
open(joinpath(outdir, "cost_vs_iteration.csv"), "w") do io
    header = "iteration"
    for r in results
        header *= ",$(r.label)_cost"
    end
    println(io, header)
    max_iters = maximum(size(r.output.f, 2) for r in results)
    for k in 1:max_iters
        row = "$(k-1)"
        for r in results
            cost = k <= size(r.output.f, 2) ? r.output.f[k] : ""
            row *= ",$cost"
        end
        println(io, row)
    end
end
println("Saved: cost_vs_iteration.csv")

# RMSRE vs iteration 
open(joinpath(outdir, "rmsre_vs_iteration.csv"), "w") do io
    header = "iteration"
    for r in results
        header *= ",$(r.label)_T1_RMSRE,$(r.label)_T2_RMSRE"
    end
    println(io, header)
    max_iters = maximum(length(c.rmsre_T1) for c in all_curves)
    for k in 1:max_iters
        row = "$(k-1)"
        for c in all_curves
            t1 = k <= length(c.rmsre_T1) ? c.rmsre_T1[k] : ""
            t2 = k <= length(c.rmsre_T2) ? c.rmsre_T2[k] : ""
            row *= ",$t1,$t2"
        end
        println(io, row)
    end
end
println("Saved: rmsre_vs_iteration.csv")

# efficiency vs iteration
open(joinpath(outdir, "efficiency_vs_iteration.csv"), "w") do io
    header = "iteration"
    for r in results
        header *= ",$(r.label)_T1_TnNR,$(r.label)_T2_TnNR"
    end
    println(io, header)
    max_iters = maximum(length(c.tnr_T1) for c in all_curves)
    for k in 1:max_iters
        row = "$(k-1)"
        for c in all_curves
            t1 = k <= length(c.tnr_T1) ? c.tnr_T1[k] : ""
            t2 = k <= length(c.tnr_T2) ? c.tnr_T2[k] : ""
            row *= ",$t1,$t2"
        end
        println(io, row)
    end
end
println("Saved: efficiency_vs_iteration.csv")

# MARE
open(joinpath(outdir, "mare_summary.csv"), "w") do io
    println(io, "config,T1_MARE_pct,T2_MARE_pct")
    for r in results
        n_iters = size(r.output.f, 2)
        p = extract_params(r.output.x, r.transmit_field, r.N, n_iters)
        mare_T1 = mare(p.T₁, gt.T₁, roi_mask) * 100
        mare_T2 = mare(p.T₂, gt.T₂, roi_mask) * 100
        println(io, "$(r.label),$(round(mare_T1, digits=4)),$(round(mare_T2, digits=4))")
    end
end
println("Saved: mare_summary.csv")

println("\nComparison complete")
