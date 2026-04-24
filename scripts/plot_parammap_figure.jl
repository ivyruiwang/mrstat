#!/usr/bin/env julia
# ============================================================================
# Generate the noisy parameter map figure for the paper
# Usage: julia --project=. scripts/plot_parammap_figure.jl results_single_gpu.jld2 results_mpi_1n4g.jld2 results_mpi_2n8g.jld2
# ============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JLD2, Statistics, LinearAlgebra, StructArrays
using PythonPlot
using MRSTAT: optim_to_physical_pars
using QMRIColors
using ImagePhantoms

outdir = "/home/iwang3/mrstat_main/analysis_results"
mkpath(outdir)

# ── Load ──────────────────────────────────────────────────────

function load_result(path)
    d = load(path)
    meta = d["meta"]
    label = meta["solver"] == "single_gpu" ? "Single GPU" : "MPI $(meta["ngpus"])GPU"
    return (; output=d["output"], ground_truth=d["ground_truth"], N=d["N"],
              transmit_field=d["transmit_field"], meta, label)
end

res_sg = load_result(ARGS[1])
res_4g = load_result(ARGS[2])
res_8g = load_result(ARGS[3])
configs = [res_sg, res_4g, res_8g]
N = res_sg.N
gt = res_sg.ground_truth

# ROI
sl = shepp_logan(N, SheppLoganBrainWeb()) |> rotr90
tissue_rois = [(val, findall(sl .== val)) for val in unique(sl) if val != 0]
roi_mask = vcat([roi for (_, roi) in tissue_rois]...)

# ── Helpers ───────────────────────────────────────────────────

function extract_params(x_all, transmit_field, N, iter)
    x = x_all[:, iter]
    phys = optim_to_physical_pars(x, transmit_field)
    return reshape(phys, N, N)
end

function rmsre(recon, truth, mask)
    rel_err = (recon[mask] .- truth[mask]) ./ truth[mask]
    return sqrt(mean(rel_err .^ 2))
end

function error_map(recon, truth, mask)
    err = zeros(N, N)
    err[mask] = abs.(Float64.(recon[mask]) .- Float64.(truth[mask])) ./ Float64.(truth[mask])
    return err
end

function masked(img, mask)
    m = zeros(N, N)
    m[mask] = Float64.(img[mask])
    return m
end

# ── Optimal iterations ────────────────────────────────────────

n_iters = size(res_sg.output.f, 2)
rmsre_T1 = [rmsre(extract_params(res_sg.output.x, res_sg.transmit_field, N, k).T₁, gt.T₁, roi_mask) for k in 1:n_iters]
rmsre_T2 = [rmsre(extract_params(res_sg.output.x, res_sg.transmit_field, N, k).T₂, gt.T₂, roi_mask) for k in 1:n_iters]
opt_T1 = argmin(rmsre_T1)
opt_T2 = argmin(rmsre_T2)
println("T₁ optimal: iteration $(opt_T1-1), T₂ optimal: iteration $(opt_T2-1)")

# ── Extract all needed maps ──────────────────────────────────

# For each config: (T1@opt, T1@fin, T2@opt, T2@fin)
all_T1_opt = [masked(extract_params(r.output.x, r.transmit_field, N, opt_T1).T₁, roi_mask) for r in configs]
all_T1_fin = [masked(extract_params(r.output.x, r.transmit_field, N, n_iters).T₁, roi_mask) for r in configs]
all_T2_opt = [masked(extract_params(r.output.x, r.transmit_field, N, opt_T2).T₂, roi_mask) for r in configs]
all_T2_fin = [masked(extract_params(r.output.x, r.transmit_field, N, n_iters).T₂, roi_mask) for r in configs]

all_T1_opt_err = [error_map(extract_params(r.output.x, r.transmit_field, N, opt_T1).T₁, gt.T₁, roi_mask) .* 100 for r in configs]
all_T1_fin_err = [error_map(extract_params(r.output.x, r.transmit_field, N, n_iters).T₁, gt.T₁, roi_mask) .* 100 for r in configs]
all_T2_opt_err = [error_map(extract_params(r.output.x, r.transmit_field, N, opt_T2).T₂, gt.T₂, roi_mask) .* 100 for r in configs]
all_T2_fin_err = [error_map(extract_params(r.output.x, r.transmit_field, N, n_iters).T₂, gt.T₂, roi_mask) .* 100 for r in configs]

gt_T1 = masked(gt.T₁, roi_mask)
gt_T2 = masked(gt.T₂, roi_mask)

# Error clim: unified range for T₁ and T₂ (outlier voxels clipped to white)
clim_err = 30.0
println("Error clim: 0-$(clim_err)%")

# ── Colormaps ─────────────────────────────────────────────────

cmap_lipari = PythonPlot.ColorMap("lipari", QMRIColors.relaxationColorMap("T1"), length(QMRIColors.relaxationColorMap("T1")), 1.0)
cmap_navia  = PythonPlot.ColorMap("navia",  QMRIColors.relaxationColorMap("T2"), length(QMRIColors.relaxationColorMap("T2")), 1.0)

# ── Column layout ────────────────────────────────────────────
# 7 columns: GT / SG@opt / SG@fin / 4G@opt / 4G@fin / 8G@opt / 8G@fin

gpu_labels = ["1 GPU", "4 GPUs", "8 GPUs"]

fin_iter = n_iters - 1
col_titles_T1 = ["Ground Truth",
    "$(gpu_labels[1])\nIteration $(opt_T1-1)", "$(gpu_labels[1])\nIteration $fin_iter",
    "$(gpu_labels[2])\nIteration $(opt_T1-1)", "$(gpu_labels[2])\nIteration $fin_iter",
    "$(gpu_labels[3])\nIteration $(opt_T1-1)", "$(gpu_labels[3])\nIteration $fin_iter"]
col_titles_T2 = ["Ground Truth",
    "$(gpu_labels[1])\nIteration $(opt_T2-1)", "$(gpu_labels[1])\nIteration $fin_iter",
    "$(gpu_labels[2])\nIteration $(opt_T2-1)", "$(gpu_labels[2])\nIteration $fin_iter",
    "$(gpu_labels[3])\nIteration $(opt_T2-1)", "$(gpu_labels[3])\nIteration $fin_iter"]

# ── Draw figure using subplot (same API as original author) ──

nrows = 4
ncols = 7

# Row data: each row is 7 images
row1 = [gt_T1, all_T1_opt[1], all_T1_fin[1], all_T1_opt[2], all_T1_fin[2], all_T1_opt[3], all_T1_fin[3]]
row2 = [nothing, all_T1_opt_err[1], all_T1_fin_err[1], all_T1_opt_err[2], all_T1_fin_err[2], all_T1_opt_err[3], all_T1_fin_err[3]]
row3 = [gt_T2, all_T2_opt[1], all_T2_fin[1], all_T2_opt[2], all_T2_fin[2], all_T2_opt[3], all_T2_fin[3]]
row4 = [nothing, all_T2_opt_err[1], all_T2_fin_err[1], all_T2_opt_err[2], all_T2_fin_err[2], all_T2_opt_err[3], all_T2_fin_err[3]]

rows = [row1, row2, row3, row4]
cmaps = [cmap_lipari, "gray", cmap_navia, "gray"]

clims = [(0.0, 2.5), (0.0, clim_err), (0.0, 0.35), (0.0, clim_err)]
ylabels = ["T₁ Maps", "T₁ Error Maps", "T₂ Maps", "T₂ Error Maps"]

fig = figure(figsize=(18, 10))
fig.patch.set_facecolor("black")

for ri in 1:nrows
    for ci in 1:ncols
        idx = (ri - 1) * ncols + ci
        subplot(nrows, ncols, idx)

        img = rows[ri][ci]
        if img === nothing
            # Invisible placeholder to keep axes aligned
            imshow(zeros(N, N), clim=clims[ri], cmap=cmaps[ri], aspect="equal", alpha=0.0)
            gca().set_xticks(Float64[]); gca().set_yticks(Float64[])
            gca().set_facecolor("black")
            if ci == 1
                ylabel(ylabels[ri], fontsize=11, color="white", fontweight="bold")
            end
            continue
        end

        imshow(img, clim=clims[ri], cmap=cmaps[ri], aspect="equal")
        gca().set_xticks(Float64[]); gca().set_yticks(Float64[])
        gca().set_facecolor("black")

        # Column title
        if ri == 1
            title(col_titles_T1[ci], fontsize=9, color="white", fontweight="bold")
        elseif ri == 3
            title(col_titles_T2[ci], fontsize=9, color="white", fontweight="bold")
        end

        # Row label
        if ci == 1
            ylabel(ylabels[ri], fontsize=11, color="white", fontweight="bold")
        end

        # Colorbar on last column only
        if ci == ncols
            cb = colorbar()
            cb.ax.tick_params(colors="white", labelsize=8)
            cb.outline.set_edgecolor("white")
            # Set 6 ticks (5 equal parts)
            tick_vals = Float64.(collect(range(clims[ri][1], clims[ri][2], length=6)))
            cb.set_ticks(tick_vals)
            # Unit label
            unit = ri in [1, 3] ? "(s)" : "(%)"
            cb.ax.set_title(unit, fontsize=9, color="white", fontweight="bold")
        end
    end
end

suptitle("Noisy Phantom Reconstruction: 1 GPU vs 4 GPUs vs 8 GPUs",
    fontsize=16, color="#FFB347", fontweight="bold", y=0.98)
subplots_adjust(hspace=0.35, wspace=0.05, top=0.90)
savefig(joinpath(outdir, "parammap_figure.png"), dpi=600, bbox_inches="tight", facecolor="black")
println("Saved: parammap_figure.png")
