#=
Function for running training as a separate process
=#

using ArgParse
using .EDDPotentialsCore:
    LinearInterface,
    ManualFluxBackPropInterface,
    clear_transient_gradients!,
    load_model,
    save_model_artifact

const TRAINING_DIR = "training"

# Symbol-typed parameters that need String→Symbol coercion from TOML
const _SYMBOL_PARAMS = (:damping_mode, :solver, :optimizer)

"""
    _symbolize_kwargs(kwargs::Dict{String,Any}) -> NamedTuple

Convert a `Dict{String,Any}` to a NamedTuple suitable for keyword splatting.
String values for known Symbol-typed parameters (e.g. `damping_mode`, `solver`)
are coerced to `Symbol`.
"""
function _symbolize_kwargs(kwargs::Dict{String,Any})
    nt = NamedTuple()
    for (k, v) in kwargs
        sk = Symbol(k)
        if v isa AbstractString && sk in _SYMBOL_PARAMS
            nt = merge(nt, NamedTuple{(sk,)}((Symbol(v),)))
        else
            nt = merge(nt, NamedTuple{(sk,)}((v,)))
        end
    end
    nt
end

"""
    dataset_name(bu::Builder, i=bu.state.iteration)

Full path to the dataset file.
"""
function dataset_name(bu::Builder, i=bu.state.iteration)
    joinpath(bu.state.workdir, TRAINING_DIR, "gen-$i-dataset.jld2")
end

function is_training_dataset_ready(bu::Builder, iter=bu.state.iteration)
    dataset_path = dataset_name(bu, iter)
    isfile(dataset_path) || return false

    dft_outputs = sort(vcat([
        glob_allow_abs(joinpath(bu.state.workdir, "gen$i-dft/*.res")) for i = 0:iter
    ]...))
    isempty(dft_outputs) && return false

    dataset_mtime = mtime(dataset_path)
    latest_dft_mtime = maximum(mtime, dft_outputs)
    return dataset_mtime >= latest_dft_mtime
end

function _is_saved_model_name(name::AbstractString, prefix::AbstractString)
    startswith(name, prefix) && (endswith(name, ".json") || endswith(name, ".jld2"))
end

function _saved_model_paths(bu::Builder, tra::TrainingOption=bu.trainer; pattern=nothing)
    if pattern !== nothing
        return sort(glob_allow_abs(pattern))
    end
    training_dir = joinpath(bu.state.workdir, TRAINING_DIR)
    isdir(training_dir) || return String[]
    prefix = tra.prefix * "gen-$(bu.state.iteration)-model-"
    sort([
        joinpath(training_dir, name) for
        name in readdir(training_dir) if _is_saved_model_name(name, prefix)
    ])
end

"""
    num_existing_models(bu::Builder, tra::TrainingOption=bu.trainer)

Return the number of existing models in the training directory.
"""
function num_existing_models(bu::Builder, tra::TrainingOption=bu.trainer)
    length(_saved_model_paths(bu, tra))
end



"""
    write_dataset(builder)

Write the dataset to the disk. These are large JLD2 archive containing the data for training, validation, and testing.

Uses memory-efficient split-then-featurize pipeline by default.
Pass a pre-built FeatureStore as the second argument to use the legacy path.
"""
function write_dataset(bu::Builder; name=dataset_name(bu))
    ensure_dir(splitdir(name)[1])
    train, val, test = split_then_featurize(bu, bu.trainer.train_split...)
    _save_split_dataset(name, train, val, test)
end

function write_dataset(bu::Builder, fs::FeatureStore; name=dataset_name(bu))
    ensure_dir(splitdir(name)[1])
    train, val, test = split(
        fs,
        bu.trainer.train_split...;
        respect_shakes=bu.trainer.respect_shakes,
    )
    _save_split_dataset(name, train, val, test)
end

function _save_split_dataset(name, train, val, test)
    jldopen(name, "w") do file
        file["format_version"] = "2"
        file["train"] = train
        file["validate"] = val
        file["test"] = test
    end
    labels = Dict{String,Vector{String}}()
    labels["train"] = train.labels
    labels["validate"] = val.labels
    labels["test"] = test.labels
    YAML.write_file(splitext(name)[1] * ".yaml", labels)
    return
end

"""
    load_training_dataset(bu::Builder, iter=bu.state.iteration;combined=false)

Load dataset saved for training
"""
function load_training_dataset(bu::Builder, iter=bu.state.iteration; combined=false)
    dataset_path = dataset_name(bu, iter)
    train, val, test = jldopen(dataset_path) do file
        if haskey(file, "format_version")
            file["train"], file["validate"], file["test"]
        else
            # Old format: "test" held monitoring data, "validate" held final eval
            file["train"], file["test"], file["validate"]
        end
    end
    if combined
        return train + val + test
    end
    return train, val, test
end


"""
    run_trainer(trainer::TrainingOption, builder::Builder)

Train the model and write the result to the disk as a JSON archive.
"""
function _train_one_model!(bu::Builder, tra::TrainingOption=bu.trainer; model_path=nothing)
    training_dir = joinpath(bu.state.workdir, "training")
    (; training_mode) = tra
    ensure_dir(training_dir)

    train, val, test = load_training_dataset(bu; combined=false)

    x_train, y_train = get_fit_data(train)
    x_val, y_val = get_fit_data(val)

    if tra.boltzmann_kt > 0
        @info "Using Boltzmann weights with kT = $(tra.boltzmann_kt)"
        y_train_at = y_train ./ size.(x_train, 2)
        y_train_at .-= minimum(y_train_at)
        weights = boltzmann.(y_train_at, tra.boltzmann_kt)
    else
        weights = nothing
    end

    @info "Model initialized"
    if training_mode == "manual_backprop"
        model = ManualFluxBackPropInterface(
            bu.cf,
            tra.n_nodes...;
            xt=train.xt,
            yt=train.yt,
            embedding=bu.cf_embedding,
        )
    elseif training_mode == "linear"
        model = LinearInterface(zeros(size(x_train[1], 1)))
    else
        throw(ErrorException("Unknown `training_mode`: $(training_mode)"))
    end

    @info "Starting training ($(tra.train_method))"
    if tra.train_method == "sgd"
        train_sgd!(
            model,
            x_train,
            y_train;
            x_val,
            y_val,
            max_epochs=tra.max_iter,
            earlystop=tra.earlystop,
            show_progress=tra.show_progress,
            keep_best=tra.keep_best,
            log_file=tra.log_file,
            weights,
            optimizer=Symbol(tra.optimizer),
            lr=tra.lr,
            momentum=tra.momentum,
            batch_size=max(1, tra.batch_size),
            weight_decay=tra.weight_decay,
            grad_clip=tra.grad_clip,
            feature_spec=bu.cf,
            elemental_energies=make_symbol_keys(bu.state.elemental_energies),
            _symbolize_kwargs(tra.training_kwargs)...,
        )
    elseif tra.batch_size > 0
        @info "Using batched LM with batch_size=$(tra.batch_size)"
        train_lm_batched!(
            model,
            x_train,
            y_train;
            x_val,
            y_val,
            max_epochs=tra.max_iter,
            earlystop=tra.earlystop,
            show_progress=tra.show_progress,
            p=tra.p,
            keep_best=tra.keep_best,
            log_file=tra.log_file,
            weights,
            batch_size=tra.batch_size,
            _symbolize_kwargs(tra.training_kwargs)...,
        )
    else
        train_lm!(
            model,
            x_train,
            y_train;
            x_val,
            y_val,
            max_epochs=tra.max_iter,
            earlystop=tra.earlystop,
            show_progress=tra.show_progress,
            p=tra.p,
            keep_best=tra.keep_best,
            log_file=tra.log_file,
            weights,
            _symbolize_kwargs(tra.training_kwargs)...,
        )
    end

    if tra.log_file != "" || tra.show_progress
        rtrain = TrainingResults(model, train)
        rval = TrainingResults(model, val)
        rtest = TrainingResults(model, test)
        if tra.log_file != ""
            open(tra.log_file, "a") do file
                println(file, "==== Training ====")
                show(file, rtrain)
                print(file, "\n")
                println(file, "=== Validation ===")
                show(file, rval)
                print(file, "\n")
                println(file, "====== Test ======")
                show(file, rtest)
                print(file, "\n")
            end
        end
        if tra.show_progress
            println("==== Training ====")
            show(rtrain)
            print("\n")
            println("=== Validation ===")
            show(rval)
            print("\n")
            println("====== Test ======")
            show(rtest)
            print("\n")
        end
    end

    clear_transient_gradients!(model)
    savepath = if model_path === nothing
        model_name =
            tra.prefix * "gen-$(bu.state.iteration)-model-" * string(uuid4())[1:8] * ".json"
        joinpath(training_dir, model_name)
    else
        ensure_dir(splitdir(model_path)[1])
        model_path
    end
    save_model_artifact(
        savepath,
        model;
        feature_spec=bu.cf,
        elemental_energies=make_symbol_keys(bu.state.elemental_energies),
    )
    @info "Model saved $(basename(savepath))"
    return savepath
end

function run_trainer_slot!(
    bu::Builder,
    tra::TrainingOption=bu.trainer;
    slot_id::Int,
    model_path::AbstractString,
)
    tmp_path = model_path * ".tmp-$(getpid())"
    isfile(tmp_path) && rm(tmp_path)
    result = _train_one_model!(bu, tra; model_path=tmp_path)
    mv(result, model_path; force=true)
    return model_path
end

function run_trainer(bu::Builder, tra::TrainingOption=bu.trainer;)
    i_trained = 0
    while num_existing_models(bu) < tra.nmodels
        _train_one_model!(bu, tra)
        i_trained += 1
    end
    @info "Trainer completed - total number of trained models: $(i_trained)"
end

"""
    load_training_data(file)

Load the training data from a JLD2 archive.
"""
function load_training_data(file)
    jldopen(file) do handle
        if haskey(handle, "format_version")
            handle["train"], handle["validate"], handle["test"]
        else
            handle["train"], handle["test"], handle["validate"]
        end
    end
end


"""
    create_ensemble(bu::Builder, tra::TrainingOption=bu.trainer;save_and_clean=false, kwargs...)

Create ensemble from resulted models. Optionally save the created ensemble model and clear
the transient data.
"""
function create_ensemble(
    bu::Builder,
    tra::TrainingOption=bu.trainer;
    save_ensemble_model=false,
    clean_individual_models=false,
    dataset_path=dataset_name(bu),
    ensemble_target=:training,
    pattern=nothing,
    alg=:fnnls,
)
    names = _saved_model_paths(bu, tra; pattern)
    @assert !isempty(names) "No model found for iteration $(bu.state.iteration)."
    # Load the models
    models = load_model.(names)
    train, val, test = load_training_data(dataset_path)

    # Select which set of data is used for ensemble creation
    if ensemble_target == :test
        total = test
    elseif ensemble_target == :all
        total = train + val + test
    elseif ensemble_target == :training
        total = train
    elseif ensemble_target == :val
        total = val
    elseif ensemble_target == :trainandval
        total = train + val
    end

    ensemble =
        create_ensemble(models, total; alg, error_threshold=tra.ensemble_error_threshold)
    if save_ensemble_model
        savepath = joinpath(bu.state.workdir, ensemble_name(bu))
        save_model_artifact(
            savepath,
            ensemble;
            feature_spec=bu.cf,
            elemental_energies=make_symbol_keys(bu.state.elemental_energies),
            metadata=Dict(
                "train_labels" => train.labels,
                "val_labels" => val.labels,
                "test_labels" => test.labels,
                "builder_uuid" => builder_uuid(bu),
            ),
        )
    end
    clean_individual_models && rm.(names)
    ensemble
end

"""
    run_trainer()

Run training through a command line interface.
"""
function run_trainer()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--prefix"
        help = "Prefix used for the trained models."
        default = ""
        "--iteration"
        help = "Override the iteration setting of the builder."
        default = -1
        arg_type = Int
        "--id"
        help = "ID of the process"
        default = ""
        arg_type = String
        "builder"
        help = "Path to the builder file."
    end
    args = parse_args(s)
    builder = Builder(args["builder"])
    if args["iteration"] >= 0
        builder.state.iteration = args["iteration"]
    end
    if args["prefix"] != ""
        builder.trainer.prefix = args["prefix"]
    end
    if args["id"] != "" && builder.trainer.log_file != ""
        builder.trainer.log_file = builder.trainer.log_file * "-" * args["id"]
    end
    run_trainer(builder)
end
