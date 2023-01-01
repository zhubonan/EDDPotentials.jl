#=
This files contains recipies for using Plots.jl for visualization
=#

using RecipesBase
using LaTeXStrings

_get_target_and_pred(tr::TrainingResults) =  tr.H_target ./ natoms(tr), tr.H_pred ./ natoms(tr)

@recipe function f(tr::TrainingResults) 
    _get_target_and_pred(tr)
end

@recipe function f(::Type{TrainingResults}, tr::TrainingResults) 
    _get_target_and_pred(tr)
end

function resample_mae_rmse(pred, target, samples)
    rel = target .- minimum(target)
    output_rmse = zeros(length(samples))
    output_mae = similar(output_rmse)
    for (i, cutoff) in enumerate(samples)
        mask = rel .< cutoff
        err = pred[mask] .- target[mask] 
        output_rmse[i] = sqrt(sum(err .^ 2) / length(err))
        output_mae[i] = sum(abs, err) / length(err)
    end
    output_mae, output_rmse
end

resample_mae_rmse(tr::TrainingResults, samples) = resample_mae_rmse(_get_target_and_pred(tr)..., samples)

@userplot RelativeAbsoluteError

"""
    relativeabsoluteerror(t::TrainingResults, args...;kwargs...)

Plot the absolute error per atom against the relative energy (per atom).
This can be useful to check that the those with lower energy has smaller errors.
"""
relativeabsoluteerror


@recipe function f(h::RelativeAbsoluteError)

    if isa(h.args[1], TrainingResults)
        target, pred = _get_target_and_pred(h.args[1])
    else
        target, pred = h.args
    end

    error = abs.(pred .- target)
    rel = target .- minimum(target)
    @series begin
        seriestype := :scatter
        subplot := 1
        xlim := (0, 1)
        ylim := (0, 0.1)
        markersize := 1
        xlabel := "Relative energy (eV /atom)"
        ylabel := "Absolute error (eV /atom)"
        rel, error
    end
end

@userplot EnthalpyAndVolume

"""
    enthalpyandvolume(args...;kwargs...)

Plots the training/predicted enthalpy per atom verses the volume per atom.
If the first argument is a `TrainingResults` then the predicted enthalpies will
be used instead.
"""
enthalpyandvolume

@recipe function f(h::EnthalpyAndVolume)
    if isa(h.args[1], StructureContainer)
        sc = h.args[1]

        H = enthalpy_per_atom(sc)
        V = volume.(sc.structures) ./ natoms(sc)
        ytext = "Enthalpy (eV / atom)"
    elseif isa(h.args[1], TrainingResults) && isa(h.args[2], StructureContainer)
        sc_base = h.args[2]
        tr = h.args[1]
        labels = tr.fc.labels
        sc = sc_base[labels]

        H = tr.H_pred ./ natoms(tr)
        V = volume.(sc.structures) ./ natoms(sc)
        ytext = "Predicted Enthalpy (eV / atom)"
    end

    @series begin
        seriestype := :scatter
        markersize := 1
        xlabel := L"Volume ($\AA^3$ / atom)"
        ylabel := ytext
        V, H
    end
end