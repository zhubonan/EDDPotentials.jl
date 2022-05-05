#=
Training routines
=#
using Base.Threads
using Dates
using JLD2
using Glob
using Printf
using NNLS
using NLSolversBase
using ProgressMeter
using Parameters


"""
    predict_energy!(f, model, x;total=reduce(hcat, x))

Predict the energy for a given model, assuming data is already normalised.
"""
function predict_energy!(f, model, x::AbstractVector;total=reduce(hcat, x))
    all_E = model(total)
    ct = 1
    for i in 1:length(f)
        lv = size(x[i], 2)
        f[i] = elemmean(all_E, ct, lv)
        ct += lv
    end
    f
end

predict_energy(model, x::AbstractMatrix) = mean(model(x))
predict_energy(model, x::Vector{Matrix{T}};total=reduce(hcat, x)) where {T} = predict_energy!(zeros(T, length(x)), model, x;total)


"Take mean to obtain atomic energy of each frame"
function elemmean(vec, i, n)
   elemsum(vec, i, n) / n
end

"Take sum to obtain total energy of each frame"
function elemsum(vec, i, n)
    val = 0.
    for j in i:i+n-1
        val += vec[j]
    end
    val
end


"""
Configuration for training
"""
mutable struct TrainingConfig{T, V, N, G, Z}
    "The model"
    model::T
    "The in Duals - used for autodiff"
    dm::V
    "Configuration for generating duals"
    cfg::N
    "Vector of the parameters"
    p0::G
    x_train::Any
    y_train::Any
    xt::Z
    yt::Z
end

TrainingConfig(model;x, y, xt, yt) = TrainingConfig(
    model, dualize_model(model), get_jacobiancfg(paramvector(model)), paramvector(model), 
    x, y,
    xt, yt
    )
update_param!(m::TrainingConfig) = m.p0 .= paramvector(m.model)

function update_param!(m::TrainingConfig, p::AbstractVector)
    m.p0 .= paramvector(m.model)
    update_param!(m.model, p)
end

paramvector(m::TrainingConfig) = paramvector(m.model)
get_jacobiancfg(m::TrainingConfig) = m.cfg
predict_energy!(f, m::TrainingConfig, x::Vector{Matrix{T}};total=reduce(hcat, x)) where {T} = predict_energy!(f, m.model, x;total)
predict_energy(m::TrainingConfig, x::Vector{Matrix{T}};total=reduce(hcat, x)) where {T} = predict_energy(m.model, x;total)

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
    dm = m.dm
    model = m.model
    cfg = m.cfg

    function progress_tracker()
        rmse_train = atomic_rmse(predict_energy(model, x_train_norm), y_train_norm, yt)
        rmse_test = atomic_rmse(predict_energy(model, x_test_norm), y_test_norm, yt)
        show_progress && @printf "RMSE Train %10.5f eV | Test %10.5f eV\n" rmse_train rmse_test
        flush(stdout)
        push!(rec, (rmse_train, rmse_test))
        rmse_test, paramvector(model)
    end
    # Setting up the object for minimization
    g2! = setup_atomic_energy_jacobian(;model, dm, x=x_train_norm, cfg);
    f2! = setup_atomic_energy_diff(;model, x=x_train_norm, y=y_train_norm);
    od2 = OnceDifferentiable(f2!,
                         g2!, p0, zeros(eltype(x_train_norm[1]), length(x_train_norm)), inplace=true);

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
    x_train::Any
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
rmse_test(tf::T, x_test, y_test) where {T<:Union{TrainingConfig, ModelEnsemble}} = atomic_rmse(predict_energy(tf, x_test), y_test, tf.yt)

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
function predict_energy_from_fvecs(model::M, fvecs::Vector{Matrix{T}};total=reduce(hcat, fvecs)) where {T, M<:Union{ModelEnsemble, TrainingConfig}}
    nfe = model.xt.len
    # Update the feature vectors consistent with model size
    # e.g. ignore the one body term if needed
    if size(total, 1) != nfe
        total = total[end-nfe+1:end, :]
    end
    f = zeros(T, length(fvecs))
    #Scale the concatenated feature vectors
    StatsBase.transform!(model.xt, total)
    predict_energy!(f, model, fvecs;total=total)
    #Scale back  the predicted energies
    StatsBase.reconstruct!(model.yt, f)
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
        out = CellTools.predict_energy(me, cell, featurespec)
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
    ensemble = CellTools.ModelEnsemble(models, x_test_norm, y_test_norm, xt, yt);
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

    x_train_norm = training_data.x_train_norm
    y_train_norm = training_data.y_train_norm
    x_test_norm = training_data.x_test_norm
    y_test_norm = training_data.y_test_norm
    xt = training_data.xt
    yt = training_data.yt

    nfeature = size(x_train_norm[1], 1)

    isdir(savepath) || mkdir(savepath)
    savefile="$(savepath)/$(now()).jld2"

    # Write parameters
    appenddata(savefile, opt.xt_name, xt)
    appenddata(savefile, opt.yt_name, yt)
    appenddata(savefile, opt.featurespec_name, cf)
    appenddata(savefile, "training_options", opt)

    if opt.store_training_data
        appenddata(savefile, "training_data", training_data)
    end

    l = ReentrantLock()

    nmodels = opt.nmodels
    p = Progress(nmodels)
    output = []
    Threads.@threads for i in 1:nmodels
        model = generate_chain(nfeature, opt.n_nodes)
        tf = TrainingConfig(model; x=x_train_norm, y=y_train_norm, xt, yt)
        out = train!(tf; x_test_norm, y_test_norm, yt, show_progress=opt.show_progress, earlystop=opt.earlystop, maxIter=opt.max_iter)
        # Write model to the archive
        lock(l)
        try
            appenddata(savefile, "model-$i", model)
            push!(output, out)
        finally
            unlock(l)
        end
        ProgressMeter.next!(p)
    end
    output
end