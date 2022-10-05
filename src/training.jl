#=
Training routines
=#
using Distributed
using Base.Threads
using Dates
using JLD2
using Glob
using Printf
using .NNLS
using NLSolversBase
using ProgressMeter
using Parameters

const XT_NAME="xt"
const YT_NAME="yt"
const FEATURESPEC_NAME="cf"


@with_kw struct TrainingOptions
    nmodels::Int=256
    max_iter::Int=300
    "number of hidden nodes in each layer"
    n_nodes::Vector{Int}=[8]
    yt_name::String=YT_NAME
    xt_name::String=XT_NAME
    featurespec_name::String=FEATURESPEC_NAME
    earlystop::Int=30
    show_progress::Bool=false
    "Store the data used for training in the archive"
    store_training_data::Bool=true
    rmse_threshold::Float64=0.5
end

"""
Genreate a `Chain` based on a vector specifying the number of hidden nodes in each layer
"""
function generate_chain(nfeature, nnodes)
    if length(nnodes) == 0 
        return Chain(Dense(nfeature, 1))
    end

    models = Any[Dense(nfeature, nnodes[1], tanh;bias=true)]
    # Add more layers
    if length(nnodes) > 1
        for i in 2:length(nnodes)
            push!(models, Dense(nnodes[i-1], nnodes[i]))
        end
    end
    # Output layer
    push!(models, Dense(nnodes[end], 1))
    Chain(models...)
end

"""
Fit an ensemble model from an archive of trained models.

The ensemble model is saved into the same archive
"""
function create_ensemble(fname)
    models, xt, yt, traindata  = jldopen(fname) do file
        [file[x] for x in keys(file) if contains(x, "model")], file["xt"], file["yt"], file["training_data"]
    end

    ensemble = EDDP.ModelEnsemble(models, traindata.x_train_norm, traindata.y_train_norm, xt, yt);

    jldopen(fname, "a") do file
        file["ensemble"] = ensemble
    end
    ensemble
end


"""
Load an ensemble model from an archive file
"""
function load_ensemble_model(fname)
    ensemble = jldopen(fname, ) do file
        file["ensemble"]
    end
    ensemble
end


"Load CellFeature serialized in the archive"
function load_featurespec(fname;opts=TrainingOptions())
    featurespec = jldopen(fname) do file
        file[opts.featurespec_name]
    end
    featurespec
end


"""
    nnls_weights(models, x, y)

Compute the weights for an ensemble of models using NNLS.
Args:
    - `models: a `Tuple`/`Vector` of models.
    - `x`: a `Vector` containing the features of each structure.
    - `y`: a `Vector` containing the total energy of each structure.

"""
function nnls_weights(models, x, y;threshold=1e-9)
    all_engs = zeros(length(x), length(models))
    for (i, model) in enumerate(models)
        all_engs[:, i] = predict_energy.(Ref(model), x)
    end
    wt = nnls(all_engs, y)
    wt[wt .< threshold] .= 0
    wt ./= sum(wt) 
    wt
end

"""
    create_ensemble(models::AbstractVector, x::AbstractVector, y::AbstractVector;

Create an EnsembleNNInterface from a vector of interfaces and x, y data for fitting.
"""
function create_ensemble(models::AbstractVector, x::AbstractVector, y::AbstractVector;
                         threshold=1e-9)
    weights = nnls_weights(models, x, y;threshold)
    # Trim weights
    mask = weights .!=0
    EnsembleNNInterface(Tuple(models[mask]), weights[mask])
end

predict_energy(itf::AbstractNNInterface, vec) = sum(itf(vec))

"""
Perform training for the given TrainingConfig
"""
function train!(itf::AbstractNNInterface, x, y;
                   p0=EDDP.paramvector(itf),
                   maxIter=1000,
                   show_progress=false,
                   x_test=x, y_test=y,
                   earlystop=50,
                   keep_best=true,
                   p=1.25,
                   args...
                   ) 
    rec = []
    
    train_natoms = [size(v, 2) for v in x]
    test_natoms = [size(v, 2) for v in x_test]

    function progress_tracker()
        rmse_train = rmse_per_atom(itf, x, y, train_natoms)

        if x_test === x
            rmse_test = rmse_train
        else
            rmse_test = rmse_per_atom(itf, x_test, y_test, test_natoms)
        end
        show_progress && @printf "RMSE Train %10.5f eV | Test %10.5f eV\n" rmse_train rmse_test
        flush(stdout)
        push!(rec, (rmse_train, rmse_test))
        rmse_test, paramvector(itf)
    end

    # Setting up the object for minimization
    f!, j!, fj! = setup_fj(itf, x, y);
    od2 = OnceDifferentiable(f!, j!, fj!, p0, zeros(eltype(x[1]), length(x)), inplace=true);

    callback = show_progress || (earlystop > 0) ? progress_tracker : nothing
    
    opt_res = levenberg_marquardt(od2, p0;show_trace=false, 
                                  callback=callback, p=p, 
                                  maxIter=maxIter, keep_best=keep_best, earlystop, args...)
    # Update the p0 of the training configuration
    opt_res, paramvector(itf), [map(x->x[1], rec) map(x->x[2], rec)], (f!, j!, fj!)
end

function train!(itf::AbstractNNInterface, fc_train::FeatureContainer, fc_test::FeatureContainer
                ;kwargs...)
    x_train, y_train = get_fit_data(fc_train)
    x_test, y_test = get_fit_data(fc_test)
    train!(itf, x_train, y_train;x_test, y_test, kwargs...)
end

"""
    rmse_per_atom(itf::AbstractNNInterface, x, y, nat)

Return per-atom root-mean square error.
"""
function rmse_per_atom(itf::AbstractNNInterface, x, y, nat)
    return (((predict_energy.(Ref(itf), x) .- y) ./ nat) .^ 2) |> mean |> sqrt
end

"""
    mae_per_atom(itf::AbstractNNInterface, x, y, nat)

Return per-atom-mean absolute error.
"""
function mae_per_atom(itf::AbstractNNInterface, x, y, nat)
    return abs.((predict_energy.(Ref(itf), x) .- y) ./ nat) |> mean
end

function max_ae_per_atom(itf, x, y, nat)
    return abs.((predict_energy.(Ref(itf), x) .- y) ./ nat) |> maximum
end

"""
Allow func(itf, fc) signature to be used....
"""
macro _itf_per_atom_wrap(expr)
    quote 
        function $(esc(expr))(itf::AbstractNNInterface, fc::FeatureContainer)
            x, y=  get_fit_data(fc)
            nat = size.(x, 2)
            $expr(itf, x, y, nat)
        end
    end
end

@_itf_per_atom_wrap(rmse_per_atom)
@_itf_per_atom_wrap(max_ae_per_atom)
@_itf_per_atom_wrap(mae_per_atom)

function train_multi_distributed(itf, x, y; nmodels=10, kwargs...)
                  
    results_channel = RemoteChannel(() -> Channel(nmodels))
    job_channel = RemoteChannel(() -> Channel(nmodels))

    # Put the jobs
    for i=1:nmodels
        put!(job_channel, i)
    end
    futures = []
    for p in workers()
        push!(futures, remotecall(worker_train_one, 
                                 p, reinit(itf), x, y, job_channel, results_channel;kwargs...))
    end

    # Check for any errors - works should not return until they are explicitly signaled
    sleep(0.1)
    for future in futures
        if isready(future)
            output = fetch(future)
            @error "Error detected for the worker $output"
            throw(output)
        end
    end


    # Receive the data and update the progress
    i = 1
    p = Progress(nmodels)
    all_models = []
    try
        while i <= nmodels
            itf, out = take!(results_channel)
            showvalues = [(:rmse, minimum(out[3][:, 2]))]
            ProgressMeter.next!(p;showvalues)
            push!(all_models, itf)
            i +=1
        end
    catch err
        if isa(err, InterruptException)
            Base.throwto(foreach_task, err)
        else
            throw(err)
        end
    finally
        # Send signals to stop the workers
        foreach(x -> put!(job_channel, -1), 1:length(workers()))
        sleep(1.0)
        # Close all channels
        close(job_channel)
        close(results_channel)
    end
    create_ensemble(all_models, x, y)      
end


"""
    worker_train_one(model, x, y, jobs_channel, results_channel;kwargs...)

Train one model and put the results into a channel
"""
function worker_train_one(model, x, y, jobs_channel, results_channel;kwargs...)
    while true
        job_id = take!(jobs_channel)
        # Signals no more work to do
        if job_id < 0
            @info "Worker completed"
            break
        end

        out = train!(model, x, y;kwargs...)
        # Put the output in the channel storing the results
        put!(results_channel, (model, out))
    end
end