using Parameters

"""
    BuildOPtions

Options controlling the iterative build process.

Mandatory Keyword arguments:
  - workdir: Working directory
  - seedfile: Path to the seed file for random structure generation
"""
@with_kw mutable struct BuildOptions
    iteration::Int=0
    workdir::String
    seedfile::String
    max_iterations::Int=5
    per_generation::Int=100
    shake_per_minima::Int=10
    build_timeout::Float64=1.
    shake_amp::Float64=0.02
    shake_cell_amp::Float64=0.02
    n_parallel::Int=1
    mpinp::Int=2
    n_initial::Int=1000
    datapaths::Vector{String}=[]
    dft_mode::String="castep"
    build_only::Bool = false
end

"""
    iterative_build(state::BuildOptions, feature_opts::FeatureOptions, training_opts::TrainingOptions) 

Iteratively build the model by repetitively train the model based on existing data,
perform searches, do DFT calculations to expand the training set.

Note that the speed of slows down significantly with increasing data point (qudratic at least).
Hence the training set needs to be selected carefully. 
"""
function iterative_build(state::BuildOptions, feature_opts::FeatureOptions, 
        training_opts::TrainingOptions
    )
    while state.iteration <= state.max_iterations
        step!(state;
        training_opts, 
        feature_opts
        )
    end
end


function run_relaxation(state::BuildOptions, indir, outdir)
    @info "Relaxation from $(indir)"
    if state.dft_mode == "castep"
        run_crud(state.workdir, indir, outdir;state.mpinp, state.n_parallel)
    elseif  state.dft_mode == "pp3"
        run_pp3_many(joinpath(state.workdir, ".pp3_work"), indir, outdir, state.seedfile;n_parallel=state.n_parallel)
    else
        throw(ErrorException("Unknown dft_mode: $(state.dft_mode)"))
    end
end

"""
Run a single iteration of the building process
"""
function step!(state::BuildOptions;
        feature_opts::FeatureOptions, 
        training_opts::TrainingOptions
    )
    @info "Iterations $(state.iteration)"

    featurespec=CellFeature(feature_opts)
    subpath(x) = joinpath(state.workdir, x)

    # if the ensemble file exists, then we have trained for this iteration
    ensemble_path = subpath("iter-$(state.iteration)-ensemble.jld2")
    @info "Ensemble path $(ensemble_path)"
    if isfile(ensemble_path) && has_ensemble_model(ensemble_path)
        @info "Ensemble build completed for this iteration - no further action needed."
        state.iteration += 1
        return state
    end
    
    # Setup the base folders
    curdir = subpath("iter-$(state.iteration)")
    ensure_dir(curdir)
    dftdir  = subpath("iter-$(state.iteration)-dft")
    ensure_dir(dftdir)

    local ensemble
    if state.iteration == 0
        # build lots of structures
        @info "Generating initial dataset"
        # How many structure do we already?
        nstruct = length(glob(joinpath(curdir, "*.res")))
        @info "$(nstruct) existing structures found"
        # Build to the specified number of structure
        if state.n_initial - nstruct > 0
            build_cells(state.seedfile, curdir, state.n_initial - nstruct; save_as_res=true, build_timeout=state.build_timeout,
                        ntasks=state.mpinp * state.n_parallel)
        end
        if state.build_only
            @info "BUILD ONLY: Iteration 0 structures are generated"
            return
        end
        @info "DFT for the initial dataset"
        run_relaxation(state, curdir, dftdir)

        @info "Training with the initial dataset"
        ensemble = retrain_all_data(state, feature_opts, training_opts)

    else
        # Load ensemble of the previous iteration and carry on
        ensemble_path = subpath("iter-$(state.iteration-1)-ensemble.jld2")
        @info "Using ensemble model from $ensemble_path"
        ensemble = load_ensemble_model(ensemble_path)
        @info "Model RMSE (train): $(atomic_rmse(ensemble)) eV/atom"

        # Generate new structures
        @info "Build and relax using ensemble model"
        existing_relaxed = filter(x -> !contains(x, "shake"), glob(joinpath(curdir, "*.res")))
        nstruct = length(existing_relaxed)
        @info "$(nstruct) existing structures found"
        if state.per_generation - nstruct > 0
            build_and_relax(state.per_generation - nstruct, state.seedfile, curdir, 
                            ensemble, featurespec;timeout=state.build_timeout)
        end

        # Shake the structures that are just relaxed
        @info "Shaking relaxed structures"
        new_relaxed = filter(x -> !contains(x, "shake") && !(x in existing_relaxed), glob(joinpath(curdir, "*.res")))
        shake_res(new_relaxed, state.shake_per_minima, state.shake_amp, state.shake_cell_amp)
        if state.build_only
            @info "BUILD ONLY: Iteration $(state.iteration) structures are generated"
            return 
        end
        # Now do DFT
        @info "Running DFT calculations for singlepoint energies"
        run_relaxation(state, curdir, dftdir)

        # Update the model and save the ensemble
        ensemble = retrain_all_data(state, feature_opts, training_opts)
    end
    state.iteration += 1
    @info "Iterative build completed"
    @info "Model RMSE (train): $(atomic_rmse(ensemble)) eV/atom"
end

"""
Retrain with all existing data
"""
function retrain_all_data(state::BuildOptions, feature_opts::FeatureOptions, training_opts::TrainingOptions)
    @info "Retrain using latest data"
    # Add newly generated data for training
    subpath(x) = joinpath(state.workdir, x)
    for i in 0:state.iteration
        if !(subpath("iter-$i-dft/*.res") in state.datapaths)
            push!(state.datapaths, subpath("iter-$i-dft/*.res"))
        end
    end
    @info "Data paths $(state.datapaths)"
    # Output ensemble file
    ensemble_path = subpath("iter-$(state.iteration)-ensemble.jld2")
    @info "Ensemble archive path:  $(ensemble_path)"
    # Train the model
    ensemble = train_eddp(state.datapaths, ensemble_path, feature_opts, training_opts;)
    ensemble 
end
"""
Summarise the status of the project
"""
function summarise_project(opts::BuildOptions)
    subpath(x) = joinpath(opts.workdir, x)
    println("Workding directory: $(opts.workdir)")
    println("Seed file: $(opts.seedfile)")

    function print_res_count(path)
        nfiles = length(glob(joinpath(path, "*.res")))
        println("  $(path): $(nfiles) structures")
        nfiles
    end

    total_dft = 0
    for i=0:opts.max_iterations
        println("iteration $i")

        path = subpath("iter-$i")
        print_res_count(path)

        path = subpath("iter-$i-dft")
        total_dft += print_res_count(path)
    end
    println("Total training structures: $(total_dft)")
end

"""
    has_ensemble_model(path)

Return true if "ensemble" key exists in the archive file
"""
function has_ensemble_model(path)
    jldopen(path) do fhandle
        return "ensemble" in keys(fhandle)
    end
end