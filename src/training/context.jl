mutable struct TrainingContext
    itf::AbstractNNInterface
    feature_spec
    elemental_energies::Dict{Symbol,Float64}
    checkpoint_every::Int
    checkpoint_prefix::AbstractString
    show_progress::Bool
    log_file::AbstractString
    rec::Vector{Tuple{Float64,Float64}}
    time_start::Float64
    last_time::Float64
    counter::Int
end

function TrainingContext(itf, feature_spec, elemental_energies, checkpoint_every, checkpoint_prefix, show_progress, log_file)
    TrainingContext(itf, feature_spec, elemental_energies, checkpoint_every, checkpoint_prefix, show_progress, log_file, Tuple{Float64,Float64}[], time(), time(), 1)
end

function progress_callback!(ctx::TrainingContext, rmse_train, rmse_val, extra_info="")
    if ctx.show_progress || !isempty(ctx.log_file)
        tnow = time()
        elapsed = tnow - ctx.time_start
        loop_time = tnow - ctx.last_time
        ctx.last_time = tnow
        line = @sprintf "Epoch: %d %3.3f %3.3f RMSE Train %10.5f eV | Val %10.5f eV%s\n" ctx.counter loop_time elapsed rmse_train rmse_val extra_info
        ctx.show_progress && print(line)
        isempty(ctx.log_file) || (open(ctx.log_file, "a") do f; write(f, line); end)
        flush(stdout)
    end
    push!(ctx.rec, (rmse_train, rmse_val))
    if ctx.checkpoint_every > 0 && !isempty(ctx.checkpoint_prefix) && ctx.feature_spec !== nothing && ctx.counter % ctx.checkpoint_every == 0
        ckpt_path = ensure_dir(@sprintf("%s_%04d.json", ctx.checkpoint_prefix, ctx.counter))
        save_model_artifact(ckpt_path, ctx.itf; feature_spec=ctx.feature_spec, elemental_energies=ctx.elemental_energies, metadata=Dict("epoch" => ctx.counter))
    end
    return
end
