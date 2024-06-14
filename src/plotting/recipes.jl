#=
This files contains recipes for using Plots.jl for visualization
=#

using RecipesBase
using LaTeXStrings

_get_target_and_pred(tr::TrainingResults) =
    tr.H_target ./ natoms(tr), tr.H_pred ./ natoms(tr)

function has_multiple_comp(fc::FeatureContainer)
    length(unique([m[:formula] for m in fc.metadata])) > 1
end

function has_multiple_comp(sc::StructureContainer)
    length(unique([reduce_composition(Composition(m)) for m in sc.structures])) > 1
end

"""
Return relative per-atom energies
"""
function _get_rel_target_and_pred(tr::TrainingResults)
    comps = [m[:formula] for m in tr.fc.metadata]
    target, pred = _get_target_and_pred(tr)
    for comp in unique(comps)
        mask = map(x -> x == comp, comps)
        ref = minimum(target[mask])
        target[mask] .-= ref
        pred[mask] .-= ref
    end
    target, pred
end

@recipe function f(tr::TrainingResults)
    if has_multiple_comp(tr.fc)
        _get_rel_target_and_pred(tr)
    else
        _get_target_and_pred(tr)
    end
end

@recipe function f(::Type{TrainingResults}, tr::TrainingResults)
    if has_multiple_comp(tr.fc)
        _get_rel_target_and_pred(tr)
    else
        _get_target_and_pred(tr)
    end
end

function resample_mae_rmse(target, pred, samples)
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

resample_mae_rmse(tr::TrainingResults, samples) =
    resample_mae_rmse(_get_rel_target_and_pred(tr)..., samples)

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


@userplot InOutSample
"""
    inoutsample(builder::Builder, i::Int)

Return plots for out-of-sample prediction analysis.
This function loads the model trained up to ``i`` iteration and apply it to the ``i + 1`` iteration.
The prediction results is hence effectively "out-of-sample".

Ideally, we want the out-of-sample to behave like the in-sample results.
Unless otherwise stateed, latter refers to the training data and the model at the ``i`` iteration.
"""
inoutsample

function _get_inoutsample_data(builder, test_iter, iter_start=0)
    # Data Processing
    eiter = load_ensemble(builder, test_iter)
    enextiter = load_ensemble(builder, test_iter + 1)

    @info "Loading features"
    fcnextiter = load_features(builder, test_iter + 1, show_progress=false)
    fciter = load_features(builder, iter_start:test_iter, show_progress=false)
    @info "Features loaded"

    troutsample = TrainingResults(eiter, fcnextiter)
    trinsample = TrainingResults(eiter, fciter)
    troutsample_insample = TrainingResults(enextiter, fcnextiter)
    scnextiter = load_structures(builder, test_iter + 1)
    (; troutsample, trinsample, troutsample_insample, scnextiter)
end

@recipe function fh(h::InOutSample)

    if isa(h.args[1], Builder)
        troutsample, trinsample, troutsample_insample, scnextiter =
            _get_inoutsample_data(h.args...)
    else
        troutsample, trinsample, troutsample_insample, scnextiter = h.args[1]
    end

    # Data Processing

    # Target vs predicted enthalpies

    layout := @layout [
        scatter ev
        relabs resample
    ]

    markersize := 2
    markerstrokewidth := 1
    xlabelfontsize := 8
    ylabelfontsize := 8
    legendfontsize := 7
    xtickfontsize := 7
    ytickfontsize := 7

    @series begin
        subplot := 1
        seriestype := :scatter
        label := "In-sample"
        trinsample
    end

    @series begin
        subplot := 1
        seriestype := :scatter
        label := "Out-of-sample"
        xlabel := "Energy (eV / atom)"
        ylabel := "Energy (eV / atom)"
        aspectratio := :equal
        troutsample
    end


    function evplot_data(sc)

        H = enthalpy_per_atom(sc)
        V = volume.(sc.structures) ./ natoms(sc)
        if has_multiple_comp(sc)
            comps = [Composition(m) for m in sc.structures]
            for comp in comps
                mask = map(x -> x == comp, comps)
                min = minimum(H[mask])
                H[mask] .-= min
            end
        end

        V, H
    end

    function evplot_data(tr, sc)
        sc_base = sc
        labels = tr.fc.labels
        sc = sc_base[labels]

        if has_multiple_comp(tr.fc)
            H = _get_rel_target_and_pred(tr)[1]
        else
            H = tr.H_pred ./ natoms(tr)
        end

        V = volume.(sc.structures) ./ natoms(sc)
        V, H
    end

    subplot := 2
    @series begin
        seriestype := :scatter
        xlabel := L"Volume ($\AA^3$ / atom)"
        ylabel := "Enthalpy (eV / atom)"
        label := "Target"
        evplot_data(scnextiter)
    end

    @series begin
        seriestype := :scatter
        xlabel := L"Volume ($\AA^3$ / atom)"
        ylabel := "Predicted Enthalpy (eV / atom)"
        label := "Predicted"
        evplot_data(troutsample, scnextiter)
    end


    function relative_ae(tr, label_text)
        target, pred = _get_target_and_pred(tr)

        error = abs.(pred .- target)
        rel = target .- minimum(target)

        # Reset references energy for multiple compositions
        comps = [m[:formula] for m in tr.fc.metadata]
        for comp in unique(comps)
            mask = map(x -> x == comp, comps)
            rel[mask] .-= minimum(rel[mask])
        end

        @series begin
            seriestype := :scatter
            subplot := 3
            xlabel := "Relative energy (eV /atom)"
            ylabel := "Absolute error (eV /atom)"
            label := label_text
            rel, error
        end
    end

    relative_ae(trinsample, "In-sample")
    relative_ae(troutsample, "Out-of-sample")


    resampel_range = LinRange(0, 3, 100)
    y1, _ = resample_mae_rmse(trinsample, resampel_range)
    y2, _ = resample_mae_rmse(troutsample, resampel_range)

    subplot := 4
    @series begin
        label := "MAE: In-sample"
        xlabel := "Threshold (eV / atom)"
        ylabel := "Energy (eV / atom)"
        resampel_range, y1
    end

    @series begin
        label := "MAE: Out-of-sample"
        xlabel := "Threshold (eV / atom)"
        ylabel := "Energy (eV / atom)"
        resampel_range, y2
    end
end
