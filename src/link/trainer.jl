#=
Function for running training as a separate process
=#

using ArgParse

const TRAINING_DIR="training"

function run_trainer(trainer::AbstractTrainer, builder::Builder) end

"""
    dataset_name(bu::Builder)
Name for the dataset file
"""
function dataset_name(bu::Builder)
    i = bu.state.iteration
    joinpath(bu.state.workdir, "gen-$i-dataset.jld2")
end

"""
    num_existing_models(bu::Builder, tra::LocalLMTrainer=bu.trainer)

Return the number of existing models in the training directory.
"""
function num_existing_models(bu::Builder, tra::LocalLMTrainer=bu.trainer)
    training_dir = joinpath(bu.state.workdir, TRAINING_DIR)
    count(x -> endswith(x, ".jld2") && startswith(x, tra.prefix * "model"), readdir(training_dir))
end



"""
    write_dataset(builder)

Write the dataset to the disk. These are large JLD2 archive containing the data for training, testing and validation
"""
function write_dataset(bu::Builder, fc=load_features(bu);name=dataset_name(bu))
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
    labels = Dict{String, Vector{String}}()
    labels["train"] = train.labels 
    labels["test"] = test.labels 
    labels["validate"] = valid.labels
    YAML.write_file(splitext(name)[1] * ".yaml", labels)
    return
end

"""
    run_trainer(trainer::LocalLMTrainer, builder::Builder)

Train the model and write the result to the disk as a JLD2 archive.
"""
function run_trainer(bu::Builder, tra::LocalLMTrainer=bu.trainer;
    dataset_path = joinpath(bu.state.workdir, TRAINING_DIR, dataset_name(bu))
    )

    training_dir = joinpath(bu.state.workdir, "training")
    ensure_dir(training_dir)

    train, test, validation = jldopen(dataset_path) do file
        file["train"], file["test"], file["validate"]
    end

    # Enter the main training loop
    x_train, y_train = get_fit_data(train)
    x_test, y_test = get_fit_data(test)

    i_trained = 0
    while num_existing_models(bu) < tra.nmodels || i_trained > tra.max_train
        
        @info "Model initialized"
        # Initialise the model
        model = EDDP.ManualFluxBackPropInterface(
            bu.cf,
            tra.n_nodes...;
            xt=train.xt,
            yt=train.yt,
            apply_xt=false,
            embedding=bu.cf_embedding,
        )

        @info "Starting training"
        # Train one model and save it...
        train_lm!(model, x_train, y_train;x_test, y_test,
        maxIter=tra.max_iter,
        earlystop=tra.earlystop,
        show_progress=tra.show_progress,
        p=tra.p,
        keep_best=tra.keep_best,
        tb_logger_dir=tra.tb_logger_dir,
        log_file=tra.log_file,
        )

        # Display training results
        if tra.log_file !== nothing || tra.show_progress
            rtrain = TrainingResults(model, train)
            rtest = TrainingResults(model, test)
            rvalid = TrainingResults(model, validation)
            if tra.log_file !== nothing
                open(tra.log_file, "a") do file
                    println(file, "==== Training ====")
                    show(file, rtrain)
                    print("\n")
                    println(file, "====== Test ======")
                    show(file, rtest)
                    print("\n")
                    println(file, "=== Validation ===")
                    show(file, rvalid)
                    print("\n")
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
    create_ensemble(bu::Builder, tra::LocalLMTrainer=bu.trainer;save_and_clean=false, kwargs...)

Create ensemble from resulted models. Optionally save the created ensemble model and clear
the transient data.
"""
function create_ensemble(bu::Builder, tra::LocalLMTrainer=bu.trainer;
    save_and_clean=false,
    dataset_path = joinpath(bu.state.workdir, TRAINING_DIR, dataset_name(bu)),
    use_validation=false,
    pattern = joinpath(bu.state.workdir, TRAINING_DIR, tra.prefix * "model-*.jld2"),
)
    names = glob(pattern)
    @assert !isempty(names) "No model found at $pattern"
    # Load the models
    models = load_from_jld2.(names, Ref(ManualFluxBackPropInterface)) 
    train, test, validation = jldopen(dataset_path) do file
        file["train"], file["test"], file["validate"]
    end
    if use_validation
        total = train + test + validation
    else
        total = train + test
    end
    ensemble = create_ensemble(models, total)
    if save_and_clean
        savepath = joinpath(bu.state.workdir, tra.prefix * "ensemble-gen$(bu.state.iteration).jld2")
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

Run training through commandline interface.
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
    if args["id"] != "" && builder.trainer.log_file !== nothing
        builder.trainer.log_file = builder.trainer.log_file  * "-" * args["id"]
    end
    run_trainer(builder)
end