#=
Function for running training as a separate process
=#


function run_trainer(trainer::AbstractTrainer, builder::Builder) end

function run_trainer(trainer::LocalLMTrainer, builder::Builder)

    training_dir = ensure_dir(builder.state.workdir, "training")
    dataset_path = joinpath(builder.state.workdir, "dataset.jl")

    train, test, validation = jldopen(dataset_path) do file
        file[:train], file[:test], file[:validate]
    end

    # Enter the main training loop
    function nexisting()
        count(x -> endswith(x, ".jld2") && startswith(x, trainer.prefix), readdir(training_dir))
    end

    x_train, y_train = get_fit_data(train)
    x_test, y_test = get_fit_data(test)

    i_trained = 0
    while nexisting() < trainer.nmodels || i_trained > trainer.max_train
        
        # Initialise the model
        model = EDDP.ManualFluxBackPropInterface(
            builder.cf,
            trainer.n_nodes...;
            xt=train.xt,
            yt=train.yt,
            apply_xt=false,
            embedding=builder.cf_embedding,
        )

        # Train one model and save it...
        train_lm!(model, x_train, y_train;x_test, y_test,
        maxIter=trainer.max_iter,
        earlystep=trainer.earlystop,
        show_progress=trainer.show_progress,
        p=trainer.p,
        keep_best=trainer.keep_best,
        tb_logger_dir=trainer.tb_logger_dir,
        log_file=trainer.log_file,
        )

        # Display training results
        if trainer.log_file !== nothing || trainer.show_progress
            rtrain = TrainingResults(model, train)
            rtest = TrainingResults(model, test)
            rvalid = TrainingResults(model, validation)
            if trainer.log_file !== nothing
                open(trainer.log_file, "a") do file
                    println(file, "==== Training ====")
                    show(file, rtrain)
                    println(file, "====== Test ======")
                    show(file, rtest)
                    println(file, "=== Validation ===")
                    show(file, rvalid)
                end
            end
            if trainer.show_progress
                println(file, "==== Training ====")
                show(stdout, rtrain)
                println(file, "====== Test ======")
                show(stdout, rtest)
                println(file, "=== Validation ===")
                show(stdout, rvalid)
            end
        end

        # Save the model
        clear_transient_gradients!(model)
        model_name = trainer.prefix * "model-" * string(uuid4())[1:4] * ".jld2"
        save_as_jld2(joinpath(training_dir, model_name), model)
        i_trained += 1
    end # End the while loop
    @info "Trainer completed - total number of trained models: $(i_trained)"
end