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


"""
    predict_energy!(f, model, x::AbstractVector)

Predict the per-atom energy from a model, no pre/post-scaling of the input vectors are performed
"""
function predict_energy!(f, model, x)
    for (i, inp) in enumerate(x)
        f[i] = mean(model(inp))
    end
    f
end

predict_energy(model, x) = mean.(model.(x))


"""
Configuration for training
"""
mutable struct TrainingConfig{X, T, Z}
    "The model"
    model::T
    "Vector of the parameters"
    p0::Vector{X}
    x_train::Vector{Matrix{X}}
    y_train::Vector{X}
    xt::Z
    yt::Z
end

"""
    TrainingConfig(model;x, y, xt, yt)

Set up the data for training.
Args:
* `x`: normalised training inputs
* `y`: normalised training outputs
* `xt`: normalization transformation for x 
* `yt`: normalization transformation for y 
"""
TrainingConfig(model;x, y, xt, yt) = TrainingConfig(
    model, paramvector(model), 
    x, y,
    xt, yt
    )

update_param!(m::TrainingConfig) = m.p0 .= paramvector(m.model)

function update_param!(m::TrainingConfig, p::AbstractVector)
    m.p0 .= paramvector(m.model)
    update_param!(m.model, p)
end

paramvector(m::TrainingConfig) = paramvector(m.model)

"""
Perform training for the given TrainingConfig
"""
function train!(m::TrainingConfig;
                   p0=m.p0,
                   maxIter=1000,
                   show_progress=false,
                   x_train_norm=m.x_train, y_train_norm=m.y_train, 
                   x_test_norm=nothing, y_test_norm=nothing,
                   earlystop=50,
                   yt=m.yt,
                   keep_best=true,
                   p=1.25,
                   args...
                   )
    rec = []
    model = m.model

    function progress_tracker()
        rmse_train = atomic_rmse(predict_energy(model, x_train_norm), y_train_norm, yt)
        rmse_test = atomic_rmse(predict_energy(model, x_test_norm), y_test_norm, yt)
        show_progress && @printf "RMSE Train %10.5f eV | Test %10.5f eV\n" rmse_train rmse_test
        flush(stdout)
        push!(rec, (rmse_train, rmse_test))
        rmse_test, paramvector(model)
    end
    # Setting up the object for minimization
    fg! = setup_fg_backprop(model, x_train_norm, y_train_norm);
    od2 = OnceDifferentiable(only_fg!(fg!), p0, zeros(eltype(x_train_norm[1]), length(x_train_norm)), inplace=true);

    callback = show_progress || (earlystop > 0) ? progress_tracker : nothing
    if !isnothing(callback) && any(isnothing, (x_test_norm, y_test_norm, yt))
        callback = nothing
        throw(ErrorException("test data and transformation are required for progress display/early stopping"))
    end

    opt_res = levenberg_marquardt(od2, p0;show_trace=false, callback=callback, p=p, maxIter=maxIter, keep_best=keep_best, earlystop, args...)
    # Update the p0 of the training configuration
    update_param!(m)
    opt_res, paramvector(model), [map(x->x[1], rec) map(x->x[2], rec)]
end

function Base.show(io::IO, x::TrainingConfig)
    print(io, "TrainingConfig for:\n")
    show(io, x.model)
end

function Base.show(io::IO, m::MIME"text/plain", x::TrainingConfig)
    print(io, "TrainingConfig for:\n")
    show(io, m, x.model)
end


"""
An ensemble of models with weights
"""
mutable struct ModelEnsemble{T, N, M}
    models::Vector{T}
    weight::Vector{N}
    "Training Y - already normalised"
    x_train::Any
    "Training X - already normalised"
    y_train::Any
    xt::M
    yt::M
end

function ModelEnsemble(;model=m)
    ModelEnsemble([model], [1.0], nothing, nothing, nothing, nothing)
end

"""
Construct an ensemble of models via none-negative least squares (NNLS)
"""
function ModelEnsemble(models::AbstractVector, x, y, xt, yt;threshold=1e-7)
    engs = reduce(hcat, predict_energy.(models, Ref(x)))
    wt = nnls(engs, y);
    mask = wt .> threshold
    mwt = wt[mask]
    ModelEnsemble(models[mask], mwt, x, y, xt, yt)
end

function ModelEnsemble(tc; threshold=1e-7)
    ModelEnsemble([t.model for t in tc], tc[1].x_train, tc[1].y_train, tc[1].xt, tc[1].yt;threshold)
end 

"""
Using ModelEnsemble object for function call
"""
function (me::ModelEnsemble)(x::AbstractVecOrMat{T}) where {T} 
    out = zeros(T, 1, size(x, 2))
    for (i, m) in enumerate(me.models)
        out .+= m(x) .* me.weight[i]
    end
    out
end

"Compute RMSE from *normalised* training data stored"
rmse_train(tf::T) where {T <: Union{TrainingConfig, ModelEnsemble}} = atomic_rmse(predict_energy(tf, tf.x_train), tf.y_train, tf.yt)

"Compute RMSE from *normalised* test data"
rmse_test(tf::TrainingConfig, x_test, y_test) = atomic_rmse(predict_energy(tf, x_test), y_test, tf.yt)

predict_energy!(f, model::T) where {T <: Union{ModelEnsemble, TrainingConfig}} = predict_energy!(f, model, model.xtrain)
atomic_rmse(me::T; yt=me.yt) where {T <: Union{ModelEnsemble, TrainingConfig}} = atomic_rmse(predict_energy(me, me.x_train), me.y_train, yt)

function Base.show(io::IO, m::MIME"text/plain", me::ModelEnsemble)
    print(io, "Ensemble of $(length(me.models)) models: \n")
    for model in me.models
        show(io, m, model)
        print(io, "\n-----------\n")
    end
end

function Base.show(io::IO, me::ModelEnsemble)
    print(io, "Ensemble of $(length(me.models)) models: \n")
    for model in me.models
        show(io, model)
        print(io, "\n-----------\n")
    end
end

"""
Predict energy from feature vectors
"""
function predict_energy_from_fvecs(model::M, fvecs::Vector{Matrix{T}};) where {T, M<:Union{ModelEnsemble, TrainingConfig}}
    nfe = model.xt.len
    # Update the feature vectors consistent with model size
    # e.g. ignore the one body term if needed
    if size(fvecs[1], 1) != nfe
        vecs = map(x -> StatsBase.transform(model.xt, x[end-nfe+1:end, :]), fvecs)
    else
        vecs = map(x -> StatsBase.transform(model.xt, x), fvecs)
    end
    #Scale the concatenated feature vectors
    out = predict_energy(model, vecs)
    #Scale back  the predicted energies
    StatsBase.reconstruct!(model.yt, out)
end


function predict_energy_one(me::ModelEnsemble, mat::Matrix{T}) where {T}
    if size(mat, 1) != me.xt.len
        mat = mat[end-me.xt.len+1:end, :]
    end
    if !isnothing(me.xt)
        norm::Matrix{T} = StatsBase.transform(me.xt, mat)
    else
        norm = mat
    end
    eng = [0.]
    for (i, model) in enumerate(me.models)
        eng[] += mean(model(norm)) *  me.weight[i]
    end
    if !isnothing(me.yt)
        out = StatsBase.reconstruct(me.yt, eng)[]
    else
        out = eng
    end
    out
end

predict_energy(me::ModelEnsemble, c::Cell, featurespec) = predict_energy_one(me, feature_vector(featurespec, c))


"""
Obtain forces via finite difference

NOTE: this is very very inefficient!
"""
function cellforces(me::ModelEnsemble, cell::Cell, featurespec)
    function get_energy(pos) 
        bak = cell.positions[:]
        cell.positions[:] .= pos
        out = EDDP.predict_energy(me, cell, featurespec)
        cell.positions[:] .= bak
        out
    end
    flat_pos = cell.positions[:]
    od = NLSolversBase.OnceDifferentiable(get_energy, flat_pos; inplace=false);
    reshape(NLSolversBase.jacobian(od, flat_pos), 3, :)
end

"""
Fit ensemble model from a JLD2 archive containing multiple models
"""
function ensemble_from_archive(fname;xname="x_test_norm", yname="y_test_norm")
    stem = splitext(fname)[1]
    models, xt, yt, x_test_norm, y_test_norm = jldopen(fname) do file
        [file[x] for x in keys(file) if contains(x, "model")], file["xt"], file["yt"], file[xname], file[yname]
    end
    ensemble = EDDP.ModelEnsemble(models, x_test_norm, y_test_norm, xt, yt);
    jldopen(stem * "-ensemble.jld2", "a") do file
        file["ensemble"] = ensemble
    end
end


"Append data to an JLD2 archive and close the handle"
function appenddata(fname, name, data)
    jldopen(fname, "a") do file
        file[name] = data
    end
end

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

function train_multi(training_data, savepath;args...)
    opt = TrainingOptions(;args...)
    train_multi(training_data, savepath, opt)
end

function train_multi(training_data, savepath, opt::TrainingOptions;featurespec=nothing, ntasks=nthreads(), itf=ManualFluxBackPropInterface)


    x_train_norm = training_data.x_train_norm
    y_train_norm = training_data.y_train_norm
    x_test_norm = training_data.x_test_norm
    y_test_norm = training_data.y_test_norm
    xt = training_data.xt
    yt = training_data.yt

    nfeature = size(x_train_norm[1], 1)

    if contains(savepath, ".jld2")
        savefile = savepath
    else
        isdir(savepath) || mkdir(savepath)
        savefile="$(savepath)/$(now()).jld2"
    end

    # Write parameters
    appenddata(savefile, opt.xt_name, xt)
    appenddata(savefile, opt.yt_name, yt)

    isnothing(featurespec) || appenddata(savefile, opt.featurespec_name, featurespec)

    appenddata(savefile, "training_options", opt)

    if opt.store_training_data
        appenddata(savefile, "training_data", training_data)
    end


    nmodels = opt.nmodels
    results_channel = Channel(nmodels)

    """
    Do work for the ith object
    """
    function do_work(training_data)

        # Make copies of the data
        x_train_norm = copy(training_data.x_train_norm)
        y_train_norm = copy(training_data.y_train_norm)
        x_test_norm =copy(training_data.x_test_norm)
        y_test_norm = copy(training_data.y_test_norm)
        xt = deepcopy(training_data.xt)
        yt = deepcopy(training_data.yt)

        while true
            model = generate_chain(nfeature, opt.n_nodes)
            tf = TrainingConfig(model; x=x_train_norm, y=y_train_norm, xt, yt)
            out = train!(tf; x_test_norm, y_test_norm, yt, show_progress=opt.show_progress, earlystop=opt.earlystop, maxIter=opt.max_iter)
            # Check if RMSE is low enough, otherwise restart
            if opt.rmse_threshold > 0 && minimum(out[3][:, 2]) < opt.rmse_threshold
                put!(results_channel, (tf, out))
                break
            end
        end
        # Put the output in the channel storing the results
    end



    work = Channel(nmodels) do chn
         foreach(x -> put!(chn, training_data), 1:nmodels)
    end

    foreach_task = @async Threads.foreach(do_work, work;ntasks) 


    # Receive the data and update the progress
    i = 1
    p = Progress(nmodels)
    all_models = []
    try
        while i <= nmodels
            tf, out = take!(results_channel)
            appenddata(savefile, "model-$i", tf.model)
            showvalues = [(:rmse, minimum(out[3][:, 2]))]
            ProgressMeter.next!(p;showvalues)
            push!(all_models, tf.model)
            i +=1
        end
    catch err
        if isa(err, InterruptException)
            Base.throwto(foreach_task, err)
        else
            throw(err)
        end
    end
      
    (models=all_models, savefile=savefile)
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


rmse(x, y) = sqrt(mean((x .- y) .^2))

"""
Compute RMSE of the ensemble model based on unscaled data
"""
function rmse(ensemble::ModelEnsemble, x::AbstractVector, y::AbstractVector;scaled=false)
    if scaled == false
        x_norm = map(z -> StatsBase.transform(ensemble.xt, z), x)
        pred = StatsBase.reconstruct(ensemble.yt, predict_energy(ensemble, x_norm))
    else
        pred = StatsBase.reconstruct(ensemble.yt, predict_energy(ensemble, x))
        y = StatsBase.reconstruct(ensemble.yt, y)
    end
    rmse(pred, y)
end


function train_multi_distributed(training_data, savepath, opt::TrainingOptions;featurespec=nothing)


    x_train_norm = training_data.x_train_norm
    y_train_norm = training_data.y_train_norm
    x_test_norm = training_data.x_test_norm
    y_test_norm = training_data.y_test_norm
    xt = training_data.xt
    yt = training_data.yt

    nfeature = size(x_train_norm[1], 1)

    if contains(savepath, ".jld2")
        savefile = savepath
    else
        isdir(savepath) || mkdir(savepath)
        savefile="$(savepath)/$(now()).jld2"
    end

    # Write parameters
    appenddata(savefile, opt.xt_name, xt)
    appenddata(savefile, opt.yt_name, yt)

    isnothing(featurespec) || appenddata(savefile, opt.featurespec_name, featurespec)

    appenddata(savefile, "training_options", opt)

    if opt.store_training_data
        appenddata(savefile, "training_data", training_data)
    end


    nmodels = opt.nmodels
    results_channel = RemoteChannel(() -> Channel(nmodels))
    job_channel = RemoteChannel(() -> Channel(nmodels))

    # Put the jobs
    for i=1:nmodels
        put!(job_channel, i)
    end
    futures = []
    for p in workers()
        push!(futures, remote_do(worker_train_one, p, training_data, opt, job_channel, results_channel, nfeature))
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
            tf, out = take!(results_channel)
            appenddata(savefile, "model-$i", tf.model)
            showvalues = [(:rmse, minimum(out[3][:, 2]))]
            ProgressMeter.next!(p;showvalues)
            push!(all_models, tf.model)
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
      
    (models=all_models, savefile=savefile)
end


"""
    do_train_one(training_data, opt, results_channel)

Train one model and put the results into a channel
"""
function worker_train_one(training_data, opt, jobs, results_channel, nfeature)
    while true
        job_id = take!(jobs)
        # Signals no more work to do
        if job_id < 0
            @info "Worker completed"
            break
        end
        # Make copies of the data
        x_train_norm = training_data.x_train_norm
        y_train_norm = training_data.y_train_norm
        x_test_norm =training_data.x_test_norm
        y_test_norm = training_data.y_test_norm
        xt = training_data.xt
        yt = training_data.yt

        model = generate_chain(nfeature, opt.n_nodes)
        tf = TrainingConfig(model; x=x_train_norm, y=y_train_norm, xt, yt)
        out = train!(tf; x_test_norm, y_test_norm, yt, show_progress=opt.show_progress, earlystop=opt.earlystop, maxIter=opt.max_iter)
        # Put the output in the channel storing the results
        put!(results_channel, (tf, out))
    end
end

predict_energy(itf::AbstractNNInterface, vec) = sum(itf(vec))
"""
Perform training for the given TrainingConfig
"""
function train!(itf::T, x, y;
                   p0=EDDP.paramvector(itf),
                   maxIter=1000,
                   show_progress=false,
                   x_test=x, y_test=y,
                   earlystop=50,
                   keep_best=true,
                   p=1.25,
                   args...
                   ) where {T<:AbstractNNInterface}
    rec = []
    
    train_natoms = [size(v, 2) for v in x]
    test_natoms = [size(v, 2) for v in x_test]

    function progress_tracker()
        rmse_train = per_atom_rmse(itf, x, y, train_natoms)

        if x_test === x
            rmse_test = rmse_train
        else
            rmse_test = per_atom_rmse(itf, x_test, y_test, test_natoms)
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
    opt_res, paramvector(itf), [map(x->x[1], rec) map(x->x[2], rec)]
end

        
per_atom_rmse(itf::AbstractNNInterface, x, y, nat) = (((predict_energy.(Ref(itf), x) .- y) ./ nat) .^ 2) |> mean |> sqrt