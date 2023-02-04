using EDDP: load_ensemble, load_features, load_structures, resample_mae_rmse
using EDDP: enthalpyandvolume, relativeabsoluteerror, TrainingResults
using LaTeXStrings
using Plots: scatter, plot!

using EDDP
using EDDP: ComputedRecord, PhaseDiagram, get_e_above_hull

sc = EDDP.StructureContainer(["/home/bonan/work/Fe-N_v2/search/*.res"])
phased = PhaseDiagram(sc)

"""
    plot_2d_hull(phased;threshold=0.1)

Plots 2D convex hull from a PhaseDiagram
"""
function plot_2d_hull(phased; max_above_hull=0.1)
    plot_data = EDDP.get_2d_plot_data(phased; threshold=max_above_hull)
    comp_label = L"$\mathrm{%$(phased.elements[2])_x%$(phased.elements[1])_{1-x}}$"
    p = scatter(
        plot_data.x,
        plot_data.y,
        markersize=3,
        markeralpha=0.3,
        label=nothing,
        xlabel=comp_label,
        ylabel="Formation Energy (eV / Atom)",
    )
    plot!(
        p,
        plot_data.stable_x,
        plot_data.stable_y,
        marker=(:diamond, 5),
        linecolor=:red,
        linewidth=2,
        label="Stable",
    )
    p
end
