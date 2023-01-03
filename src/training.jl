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
using Printf
using StatsBase
using CatViews
import Base
import CellBase
using TensorBoardLogger
using Logging

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
function nnls_weights(models, x, y)
    all_engs = zeros(length(x), length(models))
    for (i, model) in enumerate(models)
        all_engs[:, i] = predict_energy.(Ref(model), x)
    end
    wt = nnls(all_engs, y)
    wt
end

"""
    create_ensemble(models::AbstractVector, x::AbstractVector, y::AbstractVector;

Create an EnsembleNNInterface from a vector of interfaces and x, y data for fitting.
"""
function create_ensemble(models, x::AbstractVector, y::AbstractVector;
                         threshold=1e-3)
    weights = nnls_weights(models, x, y)
    tmp_models = collect(models)
    mask = weights .< threshold
    # Eliminate models with weights lower than the threshold
    while sum(mask) > 0
        # Models with weights higher than the threshold
        tmp_models = tmp_models[map(!, mask)] 
        # Refit the weights
        weights = nnls_weights(tmp_models, x, y)
        # Models with small weights
        mask = weights .< threshold
    end
    EnsembleNNInterface(Tuple(tmp_models), weights)
end

EnsembleNNInterface(models, fc::FeatureContainer;threshold=1e-3) = create_ensemble(models, get_fit_data(fc)...;threshold)

predict_energy(itf::AbstractNNInterface, vec) = sum(itf(vec))

"""
Perform training for the given TrainingConfig
"""
function train_lm!(itf::AbstractNNInterface, x, y;
                   p0=EDDP.paramvector(itf),
                   maxIter=1000,
                   show_progress=false,
                   x_test=x, y_test=y,
                   earlystop=50,
                   keep_best=true,
                   tb_logger_dir=nothing,
                   p=1.25,
                   args...
                   ) 
    rec = []
    
    train_natoms = [size(v, 2) for v in x]
    test_natoms = [size(v, 2) for v in x_test]

    tb_logger = nothing
    if tb_logger_dir !== nothing
        tb_logger = TBLogger(tb_logger_dir)
    end

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

        # Tensor board logging
        if tb_logger !== nothing
            with_logger(tb_logger) do 
                @info "" rmse_test=rmse_test rmse_train=rmse_train            
            end
        end

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
                ;train_method="lm", kwargs...)
    @info "Training samples : $(length(fc_train))"
    @info "Test samples     : $(length(fc_train))"
    @info "Training method  : $(train_method)"
    @debug "Keyord arguments : $(kwargs)"

    if train_method == "lm"
        x_train, y_train = get_fit_data(fc_train)
        x_test, y_test = get_fit_data(fc_test)
        train_lm!(itf, x_train, y_train;x_test, y_test, kwargs...)
    elseif train_method == "optim"
        model = get_flux_model(itf)
        f, g!, pview, callback = EDDP.generate_f_g_optim(model, fc_train, fc_test)
        od = OnceDifferentiable(f, g!, collect(pview))
        x0 = collect(pview)
        opt_res = Optim.optimize(od, x0;callback=callback, kwargs...)
        opt_res, collect(pview)
    end
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

# function train_multi_distributed(itf, x, y; nmodels=10, kwargs...)
                  
#     results_channel = RemoteChannel(() -> Channel(nmodels))
#     job_channel = RemoteChannel(() -> Channel(nmodels))

#     # Put the jobs
#     for i=1:nmodels
#         put!(job_channel, i)
#     end
#     futures = []
#     for p in workers()
#         push!(futures, remotecall(worker_train_one, 
#                                  p, itf, x, y, job_channel, results_channel;kwargs...))
#     end

#     # Check for any errors - works should not return until they are explicitly signaled
#     sleep(0.1)
#     for future in futures
#         if isready(future)
#             output = fetch(future)
#             @error "Error detected for the worker $output"
#             throw(output)
#         end
#     end


#     # Receive the data and update the progress
#     i = 1
#     p = Progress(nmodels)
#     all_models = []
#     try
#         while i <= nmodels
#             itf, out = take!(results_channel)
#             showvalues = [(:rmse, minimum(out[3][:, 2]))]
#             ProgressMeter.next!(p;showvalues)
#             push!(all_models, itf)
#             i +=1
#         end
#     catch err
#         if isa(err, InterruptException)
#             Base.throwto(foreach_task, err)
#         else
#             throw(err)
#         end
#     finally
#         # Send signals to stop the workers
#         foreach(x -> put!(job_channel, -1), 1:length(workers()))
#         sleep(1.0)
#         # Close all channels
#         close(job_channel)
#         close(results_channel)
#     end
#     create_ensemble(all_models, x, y)      
# end

function train_multi_threaded(itf, fc_train, fc_test; 
    show_progress=true,
    nmodels=10, suffix=nothing, prefix=nothing, save_each_model=true, 
    use_test_for_ensemble=true, kwargs...)
                  
    results_channel = Channel(nmodels)
    job_channel = Channel(nmodels)

    # Put the jobs
    for i=1:nmodels
        put!(job_channel, i)
    end
    tasks = []
    for _ in 1:nthreads()
        push!(tasks, Threads.@spawn worker_train_one(itf, fc_train, fc_test, job_channel, results_channel;kwargs...))
    end

    # Check for any errors - works should not return until they are explicitly signaled
    sleep(0.1)
    for task in tasks
        if istaskfailed(task)
            output = fetch(task)
            @error "Error detected for the task $output"
            throw(output)
        end
    end

    # Receive the data and update the progress
    i = 1
    if show_progress
        p = Progress(nmodels)
    end
    all_models = []
    ts = Dates.format(now(), "yyyy-mm-dd-HH-MM-SS")
    if prefix === nothing
        fname = "models-$(ts)"
    else
        fname = "$(prefix)-models-$(ts)"
    end

    # Add suffix for saved models
    if !isnothing(suffix)
        fname = fname * "-$(suffix)"
    end

    try
        while i <= nmodels
            itf, out = take!(results_channel)
            if show_progress
                showvalues = [(:rmse, minimum(out[3][:, 2]))]
                ProgressMeter.next!(p;showvalues)
            end
            # Save to files
            save_each_model && save_as_jld2(@sprintf("%s-%03d.jld2", fname, i), itf)
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
        foreach(x -> put!(job_channel, -1), 1:length(nthreads()))
        sleep(1.0)
        # Close all channels
        close(job_channel)
        close(results_channel)
    end
    if use_test_for_ensemble
        create_ensemble(all_models, fc_train + fc_test)      
    else
        create_ensemble(all_models, fc_train)      
    end
end



"""
    create_ensemble(all_models, fc_train::FeatureContainer, args...)      

Create ensemble model from training data.
"""
function create_ensemble(all_models, fc::FeatureContainer, args...;kwargs...)      
    x, y = get_fit_data(fc)
    create_ensemble(all_models, x, y;kwargs...)
end


"""
    worker_train_one(model, x, y, jobs_channel, results_channel;kwargs...)

Train one model and put the results into a channel
"""
function worker_train_one(model, train, test, jobs_channel, results_channel;kwargs...)
    while true
        job_id = take!(jobs_channel)
        # Signals no more work to do
        if job_id < 0
            break
        end
        new_model = reinit(model)
        out = train!(new_model, train, test;kwargs...)
        # Put the output in the channel storing the results
        if isa(model, ManualFluxBackPropInterface)
            clear_transient_gradients!(model)
        end
        put!(results_channel, (new_model, out))
    end
end


struct TrainingResults{F, T}
    fc::F
    model::T
    H_pred::Vector{Float64}
    H_target::Vector{Float64}
end

CellBase.natoms(x::TrainingResults) = natoms(x.fc)

"""
    Base.getindex(v::TrainingResults, idx::Union{UnitRange, Vector{T}}) where {T<: Int}   

Allow slicing to work on TrainingResults.
"""
function Base.getindex(v::TrainingResults, idx::Union{UnitRange, Vector{T}}) where {T<: Int}   
    TrainingResults(v.fc[idx], v.model, v.H_pred[idx], v.H_target[idx])
end

function TrainingResults(model::AbstractNNInterface, fc::FeatureContainer)
    x, H_target = get_fit_data(fc)
    H_pred = predict_energy.(Ref(model), x)
    TrainingResults(fc, model, H_pred, H_target)
end

TrainingResults(tr::TrainingResults, fc::FeatureContainer) = TrainingResults(tr.model, fc)

function rmse_per_atom(tr::TrainingResults)
    ((tr.H_target .- tr.H_pred) ./ natoms(tr.fc)) .^ 2 |> mean |> sqrt
end

function mae_per_atom(tr::TrainingResults)
    abs.((tr.H_target .- tr.H_pred) ./ natoms(tr.fc)) |> mean 
end

"Absolute per-atom error"
function ae_per_atom(tr::TrainingResults)
    abs.((tr.H_target .- tr.H_pred) ./ natoms(tr.fc))
end

absolute_error(tr::TrainingResults) = abs.(tr.H_pred .- tr.H_target)

function Base.show(io::IO, ::MIME"text/plain", tr::TrainingResults)
    @printf(io, "TrainingResults\n%20s: %d\n",  "Number of structures",  length(tr.fc))
    ncomps = length(unique(m[:formula] for m in tr.fc.metadata))
    @printf(io, "%20s: %d\n",  "Number of compositions",  ncomps)
    @printf(io, "%-10s: %10.5f eV      ", "RMSE",  rmse_per_atom(tr))
    @printf(io, "%-10s: %10.5f eV\n", "MAE",  mae_per_atom(tr))
    max_mae, label_max = maximum_error(tr)
    @printf(io, "%-10s: %10.2f eV     on structure: %20s\n", "Max absolute error",  max_mae, label_max)
    @printf(io, "%-10s: %10.5f", "Average Spearman", spearman(tr))
end

function print_spearman(io, tr)
    @printf(io, "Spearman Scores:\n")
    for (f, s) in spearman_each_comp(tr)
        @printf(io, "  %-10s: %10.5f\n", f, s)
    end
end

print_spearman(tr::TrainingResults) = print_spearman(stdout, tr)
 

Base.show(io::IO, tr::TrainingResults) = Base.show(io, MIME("text/plain"), tr)

function maximum_error(tr::TrainingResults)
    ae = absolute_error(tr)
    maximum_ae = maximum(ae)
    imax = findfirst( x-> x== maximum_ae, ae)
    label_max = tr.fc.labels[imax]
    return maximum_ae, label_max
end

"""
    spearman_each_comp(tr::TrainingResults) 

Return unique reduced formula and their spearman scores. 
"""
function spearman_each_comp(tr::TrainingResults) 
    forms = [m[:formula] for m in tr.fc.metadata]
    nat = natoms(tr.fc)
    uforms = unique(forms)
    out = Dict{Symbol, eltype(tr.H_target)}()
    for fu in uforms
        idx = findall(x -> x == fu, forms)
        # Compare per-atom energy difference, otherwise having more diversity in the formula units
        # will results in overly optimistic spearman scores.
        out[fu] =  corspearman(tr.H_target[idx] ./ nat[idx], tr.H_pred[idx] ./ nat[idx])
    end
    out
end

function per_atom_scatter_each_comp(tr::TrainingResults)
    Dict(
        Pair(comp, (t.H_target ./ natoms(t.fc), t.H_pred ./ natoms(t.fc))) for (comp, t) in each_comp(tr)
    )
end

function per_atom_scatter_data(tr::TrainingResults)
    nat = natoms(tr.fc)
    (tr.H_target ./ nat, tr.H_pred ./ nat)
end


function each_comp(tr::TrainingResults)
    forms = [m[:formula] for m in tr.fc.metadata]
    uforms = unique(forms)
    Dict(Pair(form, tr[findall(x -> x == form, forms)]) for form in uforms)
end

"""
    ensemble_std(tr::TrainingResults{M, T};per_atom=true) where {M, T<:EnsembleNNInterface}

Return the standard deviation from the ensemble for each data point. Defaults to atomic energy.
"""
function ensemble_std(tr::TrainingResults{M, T};min_weight=0.05) where {M, T<:EnsembleNNInterface}
    function fvstd_atomic(fvec)
        std(mean(m(fvec)) for (m, w) in  zip(tr.model.models, tr.model.weights) if w > min_weight)
    end
    return fvstd_atomic.(tr.fc.fvecs)
end

mutable struct TrainingResultsSummary
    rmse::Vector{Float64}
    mae::Vector{Float64}
    spearman::Vector{Dict{Symbol,Float64}}
    r2::Vector{Dict{Symbol,Float64}}
    "Number of model parameters"
    nparam::Int
    "Length of the feature vector"
    nfeat::Int
    metadata::Any
end

"""
    TrainingResultsSummary(train, test ,valid)

Construct a TrainingResultsSummary object from TrainingResults for the train, test and 
validation sets.
"""
function TrainingResultsSummary(train, test ,valid)
    rmse = Float64[rmse_per_atom(x) for x in [train, test, valid]]
    mae = Float64[mae_per_atom(x) for x in [train, test, valid]]
    sp = Dict{Symbol,Float64}[spearman_each_comp(x) for x in [train,test,valid]]
    r2 = Dict{Symbol,Float64}[r2score_each_comp(x) for x in [train,test,valid]]
    np = nparams(train.model)
    nfeat = nfeatures(train.fc.feature)
    TrainingResultsSummary(rmse, mae, sp, r2, np, nfeat, nothing)
end

"""
    r2score_each_comp(tr::TrainingResults)

Compute the R2 score for each composition separately as it does not make sense to compute 
using the full dataset containing different compositions.

See also: https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score
"""
function r2score_each_comp(tr::TrainingResults)
    data = per_atom_scatter_each_comp(tr)
    out = Dict{Symbol, Float64}()
    for (comp, (target, pred)) in data
        mt = mean(target)
        sstot = sum((target .- mt) .^ 2) 
        ssres = sum((target .- pred) .^ 2)
        out[comp] = 1 - ssres / sstot
    end
    out
end

r2score(tr::TrainingResults) = mean(values(r2score_each_comp(tr)))

"""
    spearman(tr::TrainingResults)

Compute the average spearman score for each composition. 
"""
function spearman(tr::TrainingResults)
    return mean(values(spearman_each_comp(tr)))
end


"""
    generate_f_g_optim(model, train, test)

Generate f, g!, view of the parameters and the callback function for NN training using Optim.
"""
function generate_f_g_optim_alt(model, fc_train, fc_test;pow=2,earlystop=30)

    X = fc_train.fvecs
    Y = transform_y(fc_train)
    mdl = model

    ps = Flux.params(mdl)
    # View into the parameters of the model
    pview = CatView(ps.params...)

    # Per-atom data
    Ha = fc_train.H ./ natoms(fc_train)
    Ha_test = fc_test.H ./ natoms(fc_test)

    iter :: Int = 1
    min_test :: Float64 = floatmax(Float64)
    min_iter :: Int = 1

    function f(x)
        pview .= x 
        return loss_all(mdl, X, Y;pow)
    end

    function g!(g, x)
        pview .= x 
        grad = Zygote.gradient(ps) do 
            loss_all(mdl, X, Y;pow)
        end
        g .= CatView([grad.grads[x] for x in ps.params]...)
        return g
    end

    """
    Callback for progress display and early stopping
    """
    function callback(args...;kwargs...)
        rmse = ((mean.(mdl.(fc_train.fvecs)) .* fc_train.yt.scale[1]) .+ fc_train.yt.mean[1] .- Ha) .^ 2 |> mean |> sqrt
        rmse_test = ((mean.(mdl.(fc_test.fvecs)) .* fc_test.yt.scale[1]) .+ fc_test.yt.mean[1] .- Ha_test) .^ 2 |> mean |> sqrt
        @info "Iter $(iter) - RMSE $(round(rmse, digits=5)) eV / $(round(rmse_test, digits=5)) eV"

        if rmse_test < min_test
            min_iter = iter
        end

        if iter - min_iter > earlystop
            @info "Early stop condition triggered - last best test was $(earlystop) iterations ago"
            return true
        end

        iter += 1
        false
    end

    return f, g!, pview, callback
end

raw"""
Compute the loss as absolute difference in total energy

```math
L = \sum_i |y_i - y_i^p|^l
```
"""
function loss_all(model, fvecs, H;pow=2)
    if pow == 2
        (sum.(model.(fvecs)) .- H) .^ pow  |> sum
    else
        abs.(sum.(model.(fvecs)) .- H)  .^ pow  |> sum
    end
end

"""
    generate_f_g_optim_batch(model, train, test)

Generate f, g!, view of the parameters and the callback function for NN training using Optim.

This is for 'batch' training where all of the data are included.
"""
function generate_f_g_optim(model, fc_train, fc_test;pow=2,earlystop=30)

    X = fc_train.fvecs
    nfeat = size(X[1], 1)
    Y = transform_y(fc_train)
    xsizes = size.(X, 2)

    # Generate dataset by sizes, each dataset has X as a rank 3 tensor
    datasets = Tuple{Array{eltype(X[1]), 3}, Vector{eltype(Y)}}[]
    for usize in unique(xsizes)
        mask = findall(x -> x==usize, xsizes)
        this_size = X[mask]
        # A rank 3 tensor
        this_tensor = Array{eltype(X[1]), 3}(undef, nfeat, usize, length(mask))
        # Copy data to the tensor
        for (idx, i) in enumerate(mask)
            this_tensor[:, :, idx] .= X[i]
        end
        this_H = Y[mask]
        push!(datasets, (this_tensor, this_H))
    end
    
    mdl = model

    ps = Flux.params(mdl)
    # View into the parameters of the model
    pview = CatView(ps.params...)

    # Per-atom data
    Ha = fc_train.H ./ natoms(fc_train)
    Ha_test = fc_test.H ./ natoms(fc_test)

    iter :: Int = 1
    min_test :: Float64 = floatmax(Float64)
    min_iter :: Int = 1

    function f(x)
        pview .= x 
        #return loss_all(mdl, X, Y;pow)
        return loss_stacked(mdl, datasets;pow)
    end

    function g!(g, x)
        pview .= x 
        grad = Zygote.gradient(ps) do 
            loss_stacked(mdl, datasets;pow)
        end
        g .= CatView([grad.grads[x] for x in ps.params]...)
        return g
    end

    """
    Callback for progress display and early stopping
    """
    function callback(args...;kwargs...)
        rmse = ((mean.(mdl.(fc_train.fvecs)) .* fc_train.yt.scale[1]) .+ fc_train.yt.mean[1] .- Ha) .^ 2 |> mean |> sqrt
        rmse_test = ((mean.(mdl.(fc_test.fvecs)) .* fc_test.yt.scale[1]) .+ fc_test.yt.mean[1] .- Ha_test) .^ 2 |> mean |> sqrt
        @info "Iter $(iter) - RMSE $(round(rmse, digits=5)) eV / $(round(rmse_test, digits=5)) eV"

        if rmse_test < min_test
            min_iter = iter
        end

        if iter - min_iter > earlystop
            @info "Early stop condition triggered - last best test was $(earlystop) iterations ago"
            return true
        end

        iter += 1
        false
    end

    return f, g!, pview, callback
end

raw"""
Compute the loss as absolute difference in total energy to certain power

```math
L = \sum_i |y_i - y_i^p|^l
```

This function requires a dataset that consisted of rank3 tensors so the total energy can be obtained
straightforwardly using a single call of `sum`.
Each element of the dataset is consisted of structures with the same number of atoms.
"""
function loss_stacked(model, datasets;pow=2)
    sum(datasets) do (x, y)
        pred = sum(model(x), dims=2)[:]
        abs.(pred .- y) .^ pow |> sum
    end
end

