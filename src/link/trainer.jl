#=
Function for running training as a separate process
=#

using ArgParse

const TRAINING_DIR = "training"

function run_trainer(trainer::AbstractTrainer, builder::Builder) end

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
        x -> endswith(x, ".jld2") && startswith(x, tra.prefix * "model"),
        readdir(training_dir),
    )
end



"""
    write_dataset(builder)

Write the dataset to the disk. These are large JLD2 archive containing the data for training, testing and validation
"""
function write_dataset(bu::Builder, fc=load_features(bu); name=dataset_name(bu))
    # Save the dataset 
    training_dir = joinpath(bu.state.workdir, TRAINING_DIR)
    name = joinpath(training_dir, name)

    ensure_dir(training_dir)
    train, test, valid = split(fc, bu.trainer.train_split...)
    jldopen(name, "w") do file
        file["train"] = train
        file["test"] = test
        file["validate"] = valid
    end
    # Write the labels
    labels = Dict{String,Vector{String}}()
    labels["train"] = train.labels
    labels["test"] = test.labels
    labels["validate"] = valid.labels
    YAML.write_file(splitext(name)[1] * ".yaml", labels)
    return
end

"""
    load_training_dataset(bu::Builder, iter=bu.state.iteration;combined=false)

Load dataset saved for training
"""
function load_training_dataset(bu::Builder, iter=bu.state.iteration; combined=false)
    dataset_path = dataset_name(bu, iter)
    train, test, validation = jldopen(dataset_path) do file
        file["train"], file["test"], file["validate"]
    end
    if combined
        return train + test + validation
    end
    return train, test, validation
end


"""
    run_trainer(trainer::TrainingOption, builder::Builder)

Train the model and write the result to the disk as a JLD2 archive.
"""
function run_trainer(bu::Builder, tra::TrainingOption=bu.trainer;)

    training_dir = joinpath(bu.state.workdir, "training")
    (;training_mode) = tra
    ensure_dir(training_dir)

    train, test, validation = load_training_dataset(bu; combined=false)

    # Linear model does not need standardisation (e.g. the interface does not have this functionality)
    if training_mode == "linear"
        reconstruct_x!(train)        
        reconstruct_x!(test)        
        reconstruct_x!(validation)        
    end

    x_train, y_train = get_fit_data(train)
    x_test, y_test = get_fit_data(test)

    if tra.boltzmann_kt > 0
        @info "Using Boltzmann weights with kT = $(tra.boltzmann_kt)"
        y_train_at = y_train ./ size.(x_train, 2)
        y_train_at .-= minimum(y_train_at)
        weights = boltzmann.(y_train_at, tra.boltzmann_kt)
    else
        weights = nothing
    end

    i_trained = 0
    while num_existing_models(bu) < tra.nmodels || i_trained > tra.max_train

        @info "Model initialized"
        # Initialise the model
        if training_mode == "manual_backprop"
            model = EDDP.ManualFluxBackPropInterface(
                bu.cf,
                tra.n_nodes...;
                xt=train.xt,
                yt=train.yt,
                apply_xt=false,
                embedding=bu.cf_embedding,
            )
        elseif training_mode == "linear"
            # Linear model
            model = EDDP.LinearInterface(zeros(size(x_train[1], 1)))
        else
            throw(ErrorException("Unknown `training_mode`: $(training_mode)"))
        end

        @info "Starting training"
        # Train one model and save it...
        train_lm!(
            model,
            x_train,
            y_train;
            x_test,
            y_test,
            maxIter=tra.max_iter,
            earlystop=tra.earlystop,
            show_progress=tra.show_progress,
            p=tra.p,
            keep_best=tra.keep_best,
            tb_logger_dir=tra.tb_logger_dir,
            log_file=tra.log_file,
            weights,
        )

        # Display training results
        if tra.log_file != "" || tra.show_progress
            rtrain = TrainingResults(model, train)
            rtest = TrainingResults(model, test)
            rvalid = TrainingResults(model, validation)
            if tra.log_file != ""
                open(tra.log_file, "a") do file
                    println(file, "==== Training ====")
                    show(file, rtrain)
                    print(file, "\n")
                    println(file, "====== Test ======")
                    show(file, rtest)
                    print(file, "\n")
                    println(file, "=== Validation ===")
                    show(file, rvalid)
                    print(file, "\n")
                end
            end
            if tra.show_progress
                println("==== Training ====")
                show(rtrain)
                print("\n")
                println("====== Test ======")
                show(rtest)
                print("\n")
                println("=== Validation ===")
                show(rvalid)
                print("\n")
            end
        end

        # Save the model
        clear_transient_gradients!(model)
        model_name = tra.prefix * "model-" * string(uuid4())[1:8] * ".jld2"
        save_as_jld2(joinpath(training_dir, model_name), model)
        @info "Model save $model_name"
        i_trained += 1
    end # End the while loop
    @info "Trainer completed - total number of trained models: $(i_trained)"
end


"""
    create_ensemble(bu::Builder, tra::TrainingOption=bu.trainer;save_and_clean=false, kwargs...)

Create ensemble from resulted models. Optionally save the created ensemble model and clear
the transient data.
"""
function create_ensemble(
    bu::Builder,
    tra::TrainingOption=bu.trainer;
    save_and_clean=false,
    dataset_path=dataset_name(bu),
    use_validation=false,
    pattern=joinpath(bu.state.workdir, TRAINING_DIR, tra.prefix * "model-*.jld2"),
)
    names = glob_allow_abs(pattern)
    @assert !isempty(names) "No model found at $pattern"
    # Load the models
    models = load_from_jld2.(names)
    train, test, validation = jldopen(dataset_path) do file
        file["train"], file["test"], file["validate"]
    end

    if isa(models[1], LinearInterface)
        reconstruct_x!(train)
        reconstruct_x!(test)
        reconstruct_x!(validation)
    end

    if use_validation
        total = train + test + validation
    else
        total = train + test
    end
    ensemble = create_ensemble(models, total)
    if save_and_clean
        savepath = joinpath(bu.state.workdir, ensemble_name(bu))
        save_as_jld2(savepath, ensemble)
        # Write additional metadata
        jldopen(savepath, "r+") do fh
            fh["train_labels"] = train.labels
            fh["test_labels"] = test.labels
            fh["valid_labels"] = validation.labels
            fh["builder_uuid"] = builder_uuid(bu)
            fh["cf"] = bu.cf
        end
        # Remove individual models
        rm.(names)
    end
    ensemble
end

"""
    run_trainer()

Run training through a command line interface.
"""
function run_trainer()
    s = ArgParseSettings()
    @add_arg_table s begin
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
