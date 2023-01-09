using EDDP: load_ensemble, load_features, load_structures, resample_mae_rmse
using EDDP: enthalpyandvolume, relativeabsoluteerror, TrainingResults
using .Plots

"""
    plot_inoutsample(builder, test_iter)

Return plots for out-of-sample prediction analysis.
This function loads the model trained up to ``X`` iteration and apply it to the ``X + 1`` iteration.
The prediction results is hence effectively "out-of-sample".

Ideally, we want the out-of-sample to behave like the in-sample results.
Unless otherwise stateed, latter refers to the training data and the model at the ``X`` iteration.
"""
function plot_inoutsample(builder, test_iter)

    eiter = load_ensemble(builder, test_iter)
    enextiter = load_ensemble(builder, test_iter + 1)

    @info "Loading features"
    fcnextiter = load_features(builder, test_iter + 1, show_progress=false)
    fciter = load_features(builder, 0:test_iter, show_progress=false)
    @info "Features loaded"

    troutsample = TrainingResults(eiter, fcnextiter)
    trinsample = TrainingResults(eiter, fciter)
    troutsample_insample = TrainingResults(enextiter, fcnextiter)

    # Target vs predicted enthalpies

    p1 = plot(
        scatter(trinsample, title="In-sample"),
        scatter(troutsample, title="Out-of-sample"),
        xlabel="Target (eV /atom)",
        ylabel="Prediction (eV /atom)",
        link=:x,
        xlabelfontsize=9,
        ylabelfontsize=9,
    )

    scnextiter = load_structures(builder, test_iter + 1)

    # Enthalpy - volume plots
    p2 = plot(
        enthalpyandvolume(
            troutsample_insample,
            scnextiter,
            title="In-sample (model X + 1)",
        ),
        enthalpyandvolume(troutsample, scnextiter, title="Out-of-sample"),
        enthalpyandvolume(scnextiter, title="Ground truth"),
        link=:y,
        layout=(1, 3),
        xlabelfontsize=9,
        ylabelfontsize=9,
    )

    # Relative absoulte errors
    p3 = plot(
        relativeabsoluteerror(troutsample, title="Out-of-sample"),
        relativeabsoluteerror(trinsample, title="In-sample"),
        xlabelfontsize=9,
        ylabelfontsize=9,
    )



    resampel_range = LinRange(0, 3, 100)
    y1, y2 = resample_mae_rmse(trinsample, resampel_range)

    begin
        p4 = scatter(resampel_range, y1, label="MAE")
        scatter!(
            p4,
            resampel_range,
            y2,
            label="RMSE",
            xlabel="Threshold (eV / atom)",
            ylabel="Error (eV / atom)",
            xlabelfontsize=9,
            ylabelfontsize=9,
            title="Resampled errors - In-sample",
        )
    end

    # Resampled errors
    y1, y2 = resample_mae_rmse(troutsample, resampel_range)
    begin
        p5 = scatter(resampel_range, y1, label="MAE")
        scatter!(
            p5,
            resampel_range,
            y2,
            label="RMSE",
            xlabel="Threshold (eV / atom)",
            ylabel="Error (eV / atom)",
            xlabelfontsize=9,
            ylabelfontsize=9,
            title="Resampled errors - Out-of-sample",
        )
    end

    p1, p2, p3, p4, p5
end
