#=
Function for running training as a separate process
=#

using ArgParse
using .EDDPotentialsCore: ManualFluxBackPropInterface, LinearInterface

const TRAINING_DIR = "training"

"""
    dataset_name(bu::Builder, i=bu.state.iteration)

Full path to the dataset file.
"""
function dataset_name(bu::Builder, i=bu.state.iteration)
    joinpath(bu.state.workdir, TRAINING_DIR, "gen-$i-dataset.jld2")
end

"""
    num_existing_models(bu::Builder, tra::TrainingOption=bu.trainer)

Return the number of existing models in the training directory.
"""
function num_existing_models(bu::Builder, tra::TrainingOption=bu.trainer)
    training_dir = joinpath(bu.state.workdir, TRAINING_DIR)
    count(
        x ->
            endswith(x, ".jld2") &&
                startswith(x, tra.prefix * "gen-$(bu.state.iteration)-model"),
        readdir(training_dir),
    )
end



"""
    write_dataset(builder)

Write the dataset to the disk. These are large JLD2 archive containing the data for training, validation, and testing
"""
function write_dataset(bu::Builder, fc=load_features(bu); name=dataset_name(bu))
    # Save the dataset 
    training_dir = joinpath(bu.state.workdir, TRAINING_DIR)

    ensure_dir(splitdir(name)[1])
    train, val, test = split(fc, bu.trainer.train_split...)
    jldopen(name, "w") do file
        file["format_version"] = "2"
        file["train"] = train
        file["validate"] = val
        file["test"] = test
    end
    # Write the labels
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

Train the model and write the result to the disk as a JLD2 archive.
"""
function run_trainer(bu::Builder, tra::TrainingOption=bu.trainer;)

    training_dir = joinpath(bu.state.workdir, "training")
    (; training_mode) = tra
    ensure_dir(training_dir)

    train, val, test = load_training_dataset(bu; combined=false)

    # Linear model does not need standardisation (e.g. the interface does not have this functionality)
    if training_mode == "linear"
        reconstruct_x!(train)
        reconstruct_x!(val)
        reconstruct_x!(test)
    end

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

    i_trained = 0
    while num_existing_models(bu) < tra.nmodels

        @info "Model initialized"
        # Initialise the model
        if training_mode == "manual_backprop"
            model = ManualFluxBackPropInterface(
                bu.cf,
                tra.n_nodes...;
                xt=train.xt,
                yt=train.yt,
                apply_xt=false,
                embedding=bu.cf_embedding,
            )
        elseif training_mode == "linear"
            # Linear model
            model = LinearInterface(zeros(size(x_train[1], 1)))
        else
            throw(ErrorException("Unknown `training_mode`: $(training_mode)"))
        end

        @info "Starting training"
        # Train one model and save it...
        if tra.batch_size > 0
            @info "Using batched LM with batch_size=$(tra.batch_size)"
            train_lm_batched!(
                model,
                x_train,
                y_train;
                x_val,
                y_val,
                maxIter=tra.max_iter,
                earlystop=tra.earlystop,
                show_progress=tra.show_progress,
                p=tra.p,
                keep_best=tra.keep_best,
                log_file=tra.log_file,
                weights,
                batch_size=tra.batch_size,
            )
        else
            train_lm!(
                model,
                x_train,
                y_train;
                x_val,
                y_val,
                maxIter=tra.max_iter,
                earlystop=tra.earlystop,
                show_progress=tra.show_progress,
                p=tra.p,
                keep_best=tra.keep_best,
                log_file=tra.log_file,
                weights,
            )
        end

        # Display training results
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

        # Save the model
        clear_transient_gradients!(model)
        model_name =
            tra.prefix * "gen-$(bu.state.iteration)-model-" * string(uuid4())[1:8] * ".jld2"
        save_as_jld2(joinpath(training_dir, model_name), model)
        @info "Model save $model_name"
        i_trained += 1
    end # End the while loop
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
    ensemble_target=:test,
    pattern=joinpath(
        bu.state.workdir,
        TRAINING_DIR,
        tra.prefix * "gen-$(bu.state.iteration)-model-*.jld2",
    ),
    alg=:fnnls,
)
    names = glob_allow_abs(pattern)
    @assert !isempty(names) "No model found at $pattern"
    # Load the models
    models = load_from_jld2.(names)
    train, val, test = load_training_data(dataset_path)

    if isa(models[1], LinearInterface)
        reconstruct_x!(train)
        reconstruct_x!(val)
        reconstruct_x!(test)
    end

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
        save_as_jld2(savepath, ensemble)
        # Write additional metadata
        jldopen(savepath, "r+") do fh
            fh["train_labels"] = train.labels
            fh["val_labels"] = val.labels
            fh["test_labels"] = test.labels
            fh["builder_uuid"] = builder_uuid(bu)
            fh["cf"] = bu.cf
        end
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
