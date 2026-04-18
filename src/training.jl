#=
Training routines
=#
using Base.Threads
using Dates
using JLD2
using Glob
using Printf
using .EDDPotentialsCore.NNLS: nnls
using NLSolversBase
using ProgressMeter: Progress
using Parameters
using Printf
using StatsBase
using Flux: Dense, Chain, tanh
import Flux
import Base
import CellBase
using Logging
using NonNegLeastSquares
using .EDDPotentialsCore: paramvector
import .EDDPotentialsCore: EnsembleNNInterface

const XT_NAME = "xt"
const YT_NAME = "yt"
const FEATURESPEC_NAME = "cf"


"""
Genreate a `Chain` based on a vector specifying the number of hidden nodes in each layer
"""
function generate_chain(nfeature, nnodes)
    if length(nnodes) == 0
        return Chain(Dense(nfeature, 1))
    end

    models = Any[Dense(nfeature, nnodes[1], tanh; bias=true)]
    # Add more layers
    if length(nnodes) > 1
        for i = 2:length(nnodes)
            push!(models, Dense(nnodes[i-1], nnodes[i]))
        end
    end
    # Output layer
    push!(models, Dense(nnodes[end], 1))
    Chain(models...)
end


"""
    boltzmann(x, kT, x0=0.)

Computes the boltzmann distribution: ``e^{\\frac{-(x - x0)}{kT}}``.
"""
function boltzmann(x, kT, x0=0.0)
    exp(-(x - x0) / kT)
end


"""
    nnls_weights(models, x, y; alg=:fnnls, error_threshold=0.5) -> Vector{Float64}

Compute optimal non-negative weights for an ensemble of models using NNLS.

This function determines how much each model should contribute to the final
ensemble prediction by solving a non-negative least squares problem. The weights
minimize the total energy prediction error across all structures.

# Arguments
- `models`: Tuple or vector of trained models (e.g., multiple `AbstractNNInterface`)
- `x`: Vector of feature matrices, one per structure
- `y`: Vector of target total energies (in eV), one per structure

# Keyword Arguments
- `alg=:fnnls`: NNLS algorithm (`:fnnls` for fast NNLS is recommended)
- `error_threshold=0.5`: Exclude structures where individual models have error > threshold (eV/atom)

# Returns
- `Vector{Float64}`: Non-negative weight for each model (same length as `models`)

# Examples
```julia
# Compute weights for 5 trained models
models = [model1, model2, model3, model4, model5]
x_train, y_train = get_fit_data(fc_train)
weights = nnls_weights(models, x_train, y_train)

# Use weights manually
ensemble_energy = sum(w * predict_energy(m, fvec) for (w, m) in zip(weights, models))
```

# Notes
- Weights sum to minimize mean squared error of total energies
- Structures with very high errors from any model can be excluded to improve robustness
- At least 100 structures are always included, even if error threshold is exceeded
- Zero or near-zero weights indicate a model does not improve ensemble predictions

# See also
[`create_ensemble`](@ref), [`EnsembleNNInterface`](@ref)
"""
function nnls_weights(models, x, y; alg=:fnnls, error_threshold=0.5)
    all_engs = zeros(length(x), length(models))
    nat = map(x -> size(x, 2), x)
    y_pa = y ./ nat
    mask = ones(Bool, length(x))

    for (i, model) in enumerate(models)
        y_tmp = predict_energy.(Ref(model), x)
        all_engs[:, i] = y_tmp
        ae = abs.(y_tmp ./ nat .- y_pa)
        @. mask = mask & (ae < error_threshold)
    end
    nselected = sum(mask)
    if nselected != length(x)
        @info "Using $(nselected)/$(length(x)) structures with AE < $(error_threshold) eV/atom."
    end
    if sum(mask) < 100
        mask .= true
        @info "Less than 100 observations selected, using all structures instead."
    end
    #wt = nnls(all_engs, y)
    wt = nonneg_lsq(all_engs[mask, :], y[mask]; alg)
    wt
end


"""
    create_ensemble(models, x::AbstractVector, y::AbstractVector; kwargs...) -> EnsembleNNInterface

Create an ensemble model by combining multiple trained neural networks.

Ensemble models improve prediction accuracy and provide uncertainty estimates by
combining predictions from multiple independently trained models. The optimal
weights for each model are determined using non-negative least squares (NNLS)
to minimize prediction error on the provided data.

# Arguments
- `models`: Vector or tuple of trained `AbstractNNInterface` models
- `x::AbstractVector`: Vector of feature matrices for fitting ensemble weights
- `y::AbstractVector`: Vector of target energies for fitting ensemble weights

# Keyword Arguments
- `threshold=1e-13`: Minimum weight threshold - models with lower weights are excluded
- `alg=:fnnls`: NNLS algorithm (`:fnnls` for fast NNLS, `:nnls` for standard)
- `error_threshold=0.5`: Exclude structures with individual model errors > this value (eV/atom)

# Returns
- `EnsembleNNInterface`: Ensemble model containing selected models and their weights

# Examples
```julia
# Train multiple models with different initializations
models = []
for i in 1:10
    itf = FluxInterface(Chain(Dense(10, 20, tanh), Dense(20, 1)))
    train!(itf, fc_train, fc_test; maxIter=300)
    push!(models, itf)
end

# Create ensemble using training data
x_train, y_train = get_fit_data(fc_train)
ensemble = create_ensemble(models, x_train, y_train)

# Ensemble automatically weights models and removes low-performing ones
println("Ensemble contains ", length(ensemble.models), " models")
println("Weights: ", ensemble.weights)
```

# Notes
- Models are weighted to minimize total energy error, not per-atom error
- Models with NNLS weights below `threshold` are automatically removed
- Structures where any individual model has high error can be excluded via `error_threshold`
- Using ensemble models improves robustness and enables uncertainty quantification

# See also
[`EnsembleNNInterface`](@ref), [`nnls_weights`](@ref), [`train!`](@ref)
"""
function create_ensemble(
    models,
    x::AbstractVector,
    y::AbstractVector;
    threshold=1e-13,
    alg=:fnnls,
    error_threshold=0.5,
)
    weights = nnls_weights(models, x, y; alg, error_threshold)[:]
    tmp_models = collect(models)
    mask = weights .> threshold
    EnsembleNNInterface(Tuple(tmp_models[mask]), weights[mask])
end

EnsembleNNInterface(models, fc::FeatureContainer; kwargs...) =
    create_ensemble(models, get_fit_data(fc)...; kwargs...)

"""
    predict_energy(itf::AbstractNNInterface, vec) -> Float64

Predict the total energy for a structure from its feature vector.

This function evaluates the neural network on the feature vector and sums
the atomic energy contributions to get the total energy.

# Arguments
- `itf::AbstractNNInterface`: Trained neural network interface
- `vec`: Feature vector matrix (nfeatures × natoms)

# Returns
- `Float64`: Total predicted energy in eV

# Examples
```julia
# Predict energy for a structure
fvec = compute_fv(calc, structure)
energy = predict_energy(model, fvec)
```

# See also
[`AbstractNNInterface`](@ref), [`compute_fv`](@ref)
"""
predict_energy(itf::AbstractNNInterface, vec) = sum(itf(vec))

"""
    train_lm!(itf::AbstractNNInterface, x, y; kwargs...) -> (opt_res, params, history, funcs)

Train a neural network using the Levenberg-Marquardt algorithm.

This is the core training function that optimizes the neural network parameters
to minimize the difference between predicted and target energies. The algorithm
uses second-order optimization (Levenberg-Marquardt) which is efficient for
fitting potential energy surfaces.

# Arguments
- `itf::AbstractNNInterface`: Neural network interface to train (will be modified in-place)
- `x`: Vector of feature matrices, one per structure
- `y`: Vector of target total energies (in eV), one per structure

# Keyword Arguments
- `p0=paramvector(itf)`: Initial parameter vector for optimization
- `maxIter=1000`: Maximum number of iterations
- `show_progress=false`: Print progress to stdout during training
- `x_test=x`: Test set feature vectors (defaults to training set)
- `y_test=y`: Test set target energies (defaults to training set)
- `earlystop=50`: Stop if test error doesn't improve for this many iterations
- `keep_best=true`: Keep parameters with best test error instead of final iteration
- `log_file=""`: Path to file for logging progress (empty string = no logging)
- `p=1.25`: Levenberg-Marquardt damping parameter
- `weights=nothing`: Optional weights for each structure in the loss function

# Returns
- `opt_res`: Optimization result object
- `params`: Final parameter vector
- `history`: Matrix with columns [train_rmse, test_rmse] for each iteration
- `funcs`: Tuple of (f!, j!, fj!) functions used for optimization

# Examples
```julia
# Basic training on carbon structures
itf = FluxInterface(Chain(Dense(10, 20, tanh), Dense(20, 1)))
x_train, y_train = get_fit_data(fc_train)
x_test, y_test = get_fit_data(fc_test)

# Train with progress display and early stopping
opt_res, params, history = train_lm!(
    itf, x_train, y_train;
    x_test=x_test, y_test=y_test,
    maxIter=500,
    show_progress=true,
    earlystop=30
)

# Check final RMSE
final_train_rmse = history[end, 1]
final_test_rmse = history[end, 2]
```

# Notes
- The network parameters in `itf` are updated in-place during training
- Training minimizes RMSE of total energies, not per-atom energies
- Early stopping uses test set RMSE to prevent overfitting
- GPU acceleration is used automatically if CUDA is available

# See also
[`train!`](@ref), [`AbstractNNInterface`](@ref), [`create_ensemble`](@ref)
"""
function train_lm!(
    itf::AbstractNNInterface,
    x,
    y;
    p0=paramvector(itf),
    maxIter=1000,
    show_progress=false,
    x_test=x,
    y_test=y,
    earlystop=50,
    keep_best=true,
    log_file="",
    p=1.25,
    weights=nothing,
    args...,
)
    rec = []

    train_natoms = [size(v, 2) for v in x]
    test_natoms = [size(v, 2) for v in x_test]

    time_start = time()
    last_time = time()
    iter_count = 1
    function progress_tracker()
        rmse_train = rmse_per_atom(itf, x, y, train_natoms)

        if x_test === x
            rmse_test = rmse_train
        else
            rmse_test = rmse_per_atom(itf, x_test, y_test, test_natoms)
        end
        if show_progress || (log_file != "")
            tnow = time()
            elapsed = tnow - time_start
            loop = tnow - last_time
            last_time = tnow
            logline =
                @sprintf "Iter: %d %3.3f %3.3f RMSE Train %10.5f eV | Test %10.5f eV\n" iter_count loop elapsed rmse_train rmse_test
            show_progress && print(logline)
            if log_file != ""
                open(log_file, "a") do file
                    write(file, logline)
                end
            end
        end

        flush(stdout)
        push!(rec, (rmse_train, rmse_test))

        iter_count += 1

        rmse_test, paramvector(itf)
    end

    # Setting up the object for minimization
    f!, j!, fj! = setup_fj(itf, x, y, weights)
    od2 = OnceDifferentiable(f!, j!, fj!, p0, zeros(eltype(x[1]), length(x)), inplace=true)

    callback = show_progress || (earlystop > 0) ? progress_tracker : nothing

    func = levenberg_marquardt
    if EDDPotentials.USE_CUDA[]
        func = levenberg_marquardt_gpu
    end

    opt_res = @timeit to "lm solve" func(
        od2,
        p0;
        show_trace=false,
        callback=callback,
        p=p,
        maxIter=maxIter,
        keep_best=keep_best,
        earlystop,
        args...,
    )
    # Update the p0 of the training configuration
    opt_res, paramvector(itf), [map(x -> x[1], rec) map(x -> x[2], rec)], (f!, j!, fj!)
end

"""
    train_lm_batched!(itf::AbstractNNInterface, x, y; kwargs...) -> (opt_res, params, history, funcs)

Train a neural network using the batched Levenberg-Marquardt algorithm.

Each LM step operates on a random mini-batch of the training data rather than
the full dataset, reducing per-iteration cost from O(m·n²) to O(B·n²) where
B is the batch size.  Data is shuffled at the start of each epoch and the
callback evaluates on the full dataset for reliable early stopping.

# Arguments
- `itf::AbstractNNInterface`: Neural network interface to train (modified in-place)
- `x`: Vector of feature matrices, one per structure
- `y`: Vector of target total energies (in eV), one per structure

# Keyword Arguments
- `p0=paramvector(itf)`: Initial parameter vector
- `maxIter=1000`: Maximum total LM iterations (across all batches/epochs)
- `show_progress=false`: Print progress each epoch
- `x_test=x`: Test set feature vectors
- `y_test=y`: Test set target energies
- `earlystop=50`: Stop if test RMSE hasn't improved for this many epochs
- `keep_best=true`: Keep parameters with best test RMSE
- `log_file=""`: Path to log file (empty = no logging)
- `p=1.25`: Levenberg-Marquardt damping parameter
- `weights=nothing`: Optional per-structure weights
- `batch_size=512`: Number of samples per mini-batch
- `shuffle_seed=42`: RNG seed for reproducible shuffling

# Returns
- `opt_res`: Optimization result object
- `params`: Final parameter vector
- `history`: Matrix with columns [train_rmse, test_rmse] per epoch
- `funcs`: Tuple of (fj_batch!, f_full!) closures

# See also
[`train_lm!`](@ref), [`train!`](@ref)
"""
function train_lm_batched!(
    itf::AbstractNNInterface,
    x,
    y;
    p0=paramvector(itf),
    maxIter=1000,
    show_progress=false,
    x_test=x,
    y_test=y,
    earlystop=50,
    keep_best=true,
    log_file="",
    p=1.25,
    weights=nothing,
    batch_size=512,
    shuffle_seed=42,
    args...,
)
    rec = []

    train_natoms = [size(v, 2) for v in x]
    test_natoms = [size(v, 2) for v in x_test]

    time_start = time()
    last_time = time()
    epoch_count = 1
    function progress_tracker()
        rmse_train = rmse_per_atom(itf, x, y, train_natoms)

        if x_test === x
            rmse_test = rmse_train
        else
            rmse_test = rmse_per_atom(itf, x_test, y_test, test_natoms)
        end
        if show_progress || (log_file != "")
            tnow = time()
            elapsed = tnow - time_start
            loop = tnow - last_time
            last_time = tnow
            logline =
                @sprintf "Epoch: %d %3.3f %3.3f RMSE Train %10.5f eV | Test %10.5f eV (batch=%d)\n" epoch_count loop elapsed rmse_train rmse_test batch_size
            show_progress && print(logline)
            if log_file != ""
                open(log_file, "a") do file
                    write(file, logline)
                end
            end
        end

        flush(stdout)
        push!(rec, (rmse_train, rmse_test))

        epoch_count += 1

        rmse_test, paramvector(itf)
    end

    # Set up batched closures
    fj_batch!, f_full! = setup_fj_batched(itf, x, y, weights)

    callback = show_progress || (earlystop > 0) ? progress_tracker : nothing

    opt_res = @timeit to "batched lm solve" levenberg_marquardt_batched(
        fj_batch!,
        f_full!,
        p0;
        n_data=length(x),
        batch_size=batch_size,
        maxIter=maxIter,
        callback=callback,
        p=p,
        keep_best=keep_best,
        earlystop=earlystop,
        shuffle_seed=shuffle_seed,
        args...,
    )

    opt_res, paramvector(itf), [map(x -> x[1], rec) map(x -> x[2], rec)], (fj_batch!, f_full!)
end

"""
    train!(itf::AbstractNNInterface, fc_train::FeatureContainer, fc_test::FeatureContainer; kwargs...)

Train a neural network model using training and test datasets.

This is the main high-level training function that users should call. It automatically
extracts features and energies from FeatureContainers and dispatches to the appropriate
training algorithm (Levenberg-Marquardt by default).

# Arguments
- `itf::AbstractNNInterface`: Neural network interface to train
- `fc_train::FeatureContainer`: Training dataset with features and energies
- `fc_test::FeatureContainer`: Test dataset for validation

# Keyword Arguments
- `train_method="lm"`: Training algorithm ("lm" for Levenberg-Marquardt, "lm_batched" for batched LM, "optim" for Optim.jl)
- Additional kwargs are passed to the underlying training function

# Returns
Returns from underlying training method (`train_lm!` or Optim optimization)

# Examples
```julia
# Load training and test data
using ExtXYZ
structures_train = read("carbon_train.xyz", ExtXYZ)
structures_test = read("carbon_test.xyz", ExtXYZ)

# Create FeatureContainers
cf = CellFeature([:C])  # Define features for carbon
sc_train = StructureContainer(structures_train)
sc_test = StructureContainer(structures_test)
fc_train = FeatureContainer(cf, sc_train)
fc_test = FeatureContainer(cf, sc_test)

# Create and train model
itf = FluxInterface(Chain(Dense(10, 20, tanh), Dense(20, 1)))
train!(itf, fc_train, fc_test; maxIter=500, show_progress=true)

# Evaluate performance
train_results = TrainingResults(itf, fc_train)
println("Train RMSE: ", rmse_per_atom(train_results), " eV/atom")
```

# Notes
- The default Levenberg-Marquardt method (`train_method="lm"`) is recommended for EDDP
- Features and energies are automatically extracted from FeatureContainers
- All kwargs are passed through to the underlying training function

# See also
[`train_lm!`](@ref), [`FeatureContainer`](@ref), [`TrainingResults`](@ref), [`create_ensemble`](@ref)
"""
function train!(
    itf::AbstractNNInterface,
    fc_train::FeatureContainer,
    fc_test::FeatureContainer;
    train_method="lm",
    kwargs...,
)

    if train_method == "lm"
        x_train, y_train = get_fit_data(fc_train)
        x_test, y_test = get_fit_data(fc_test)
        train_lm!(itf, x_train, y_train; x_test, y_test, kwargs...)
    elseif train_method == "lm_batched"
        x_train, y_train = get_fit_data(fc_train)
        x_test, y_test = get_fit_data(fc_test)
        train_lm_batched!(itf, x_train, y_train; x_test, y_test, kwargs...)
    elseif train_method == "optim"
        model = get_flux_model(itf)
        f, g!, pview, callback = EDDPotentials.generate_f_g_optim(model, fc_train, fc_test)
        od = OnceDifferentiable(f, g!, collect(pview))
        x0 = collect(pview)
        opt_res = Optim.optimize(od, x0; callback=callback, kwargs...)
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

"""
    max_ae_per_atom(itf::AbstractNNInterface, x, y, nat)

Return the maximum per-atom absolute error.
"""
function max_ae_per_atom(itf, x, y, nat)
    return abs.((predict_energy.(Ref(itf), x) .- y) ./ nat) |> maximum
end

"""
Allow func(itf, fc) signature to be used....
"""
macro _itf_per_atom_wrap(expr)
    quote
        function $(esc(expr))(itf::AbstractNNInterface, fc::FeatureContainer)
            x, y = get_fit_data(fc)
            nat = size.(x, 2)
            $expr(itf, x, y, nat)
        end
    end
end

@_itf_per_atom_wrap(rmse_per_atom)
@_itf_per_atom_wrap(max_ae_per_atom)
@_itf_per_atom_wrap(mae_per_atom)



"""
    create_ensemble(all_models, fc_train::FeatureContainer, args...)      

Create ensemble model from training data.
"""
function create_ensemble(all_models, fc::FeatureContainer, args...; kwargs...)
    x, y = get_fit_data(fc)
    create_ensemble(all_models, x, y; kwargs...)
end


"""
    worker_train_one(model, x, y, jobs_channel, results_channel;kwargs...)

Train one model and put the results into a channel
"""
function worker_train_one(model, train, test, jobs_channel, results_channel; kwargs...)
    while true
        job_id = take!(jobs_channel)
        # Signals no more work to do
        if job_id < 0
            break
        end
        new_model = reinit(model)
        out = train!(new_model, train, test; kwargs...)
        # Put the output in the channel storing the results
        if isa(model, ManualFluxBackPropInterface)
            clear_transient_gradients!(model)
        end
        put!(results_channel, (new_model, out))
    end
end

"""
    TrainingResults{F,T}

Container for the results of a training run.
"""
struct TrainingResults{F,T}
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
function Base.getindex(v::TrainingResults, idx::Union{UnitRange,Vector{T}}) where {T<:Int}
    TrainingResults(v.fc[idx], v.model, v.H_pred[idx], v.H_target[idx])
end

"""
    TrainingResults(model::AbstractNNInterface, fc::FeatureContainer)

Create a `TrainingResults` object from a model and a feature container.
"""
function TrainingResults(model::AbstractNNInterface, fc::FeatureContainer)
    x, H_target = get_fit_data(fc)
    H_pred = predict_energy.(Ref(model), x)
    TrainingResults(fc, model, H_pred, H_target)
end

"""
    TrainingResults(tr::TrainingResults, fc::FeatureContainer)


Create a `TrainingResults` object from a `TrainingResults` object and a feature container.
"""
TrainingResults(tr::TrainingResults, fc::FeatureContainer) = TrainingResults(tr.model, fc)

"""
    absolute_error(tr::TrainingResults)

Absolute error.
"""
absolute_error(tr::TrainingResults) = abs.(tr.H_pred .- tr.H_target)

"""
    ae_per_atom(tr::TrainingResults)

Absolute error per atom.
"""
ae_per_atom(tr::TrainingResults) = absolute_error(tr) ./ natoms(tr.fc)

"""
    rmse_per_atom(tr::TrainingResults)

Root-mean squared error per atom.
"""
rmse_per_atom(tr::TrainingResults) = ae_per_atom(tr) .^ 2 |> mean |> sqrt


"""
    mae_per_atom(tr::TrainingResults)

Mean absolute error per atom.
"""
mae_per_atom(tr::TrainingResults) = ae_per_atom(tr) |> mean


function Base.show(io::IO, ::MIME"text/plain", tr::TrainingResults)
    @printf(io, "TrainingResults\n%20s: %d\n", "Number of structures", length(tr.fc))
    ncomps = length(unique(m[:formula] for m in tr.fc.metadata))
    @printf(io, "%20s: %d\n", "Number of compositions", ncomps)
    @printf(io, "%-10s: %10.5f eV      ", "RMSE", rmse_per_atom(tr))
    @printf(io, "%-10s: %10.5f eV\n", "MAE", mae_per_atom(tr))
    max_mae, label_max = maximum_error(tr)
    @printf(
        io,
        "%-10s: %10.2f eV     on structure: %20s\n",
        "Max absolute error",
        max_mae,
        label_max
    )
    @printf(io, "%-10s: %10.5f", "Average Spearman", spearman(tr))
end


"""
Print the spearman scores for each composition.
"""
function print_spearman(io, tr)
    @printf(io, "Spearman Scores:\n")
    for (f, s) in spearman_each_comp(tr)
        @printf(io, "  %-10s: %10.5f\n", f, s)
    end
end

print_spearman(tr::TrainingResults) = print_spearman(stdout, tr)


Base.show(io::IO, tr::TrainingResults) = Base.show(io, MIME("text/plain"), tr)


"""
Find the maximum absolute error per atom and the corresponding structure label.
"""
function maximum_error(tr::TrainingResults)
    ae = absolute_error(tr)
    maximum_ae = maximum(ae)
    imax = findfirst(x -> x == maximum_ae, ae)
    label_max = tr.fc.labels[imax]
    return maximum_ae / natoms(tr.fc)[imax], label_max
end

"""
    spearman_each_comp(tr::TrainingResults) 

Return unique reduced formula and their spearman scores. 
"""
function spearman_each_comp(tr::TrainingResults)
    forms = [m[:formula] for m in tr.fc.metadata]
    nat = natoms(tr.fc)
    uforms = unique(forms)
    out = Dict{Symbol,eltype(tr.H_target)}()
    for fu in uforms
        idx = findall(x -> x == fu, forms)
        # Compare per-atom energy difference, otherwise having more diversity in the formula units
        # will results in overly optimistic spearman scores.
        out[fu] = corspearman(tr.H_target[idx] ./ nat[idx], tr.H_pred[idx] ./ nat[idx])
    end
    out
end

"""
    per_atom_scatter_each_comp(tr::TrainingResults)

Return a dictionary of per-atom scatter data for each composition.
"""
function per_atom_scatter_each_comp(tr::TrainingResults)
    Dict(
        Pair(comp, (t.H_target ./ natoms(t.fc), t.H_pred ./ natoms(t.fc))) for
        (comp, t) in each_comp(tr)
    )
end

"""
    per_atom_scatter_data(tr::TrainingResults)

Return the per-atom scatter data as a tuple of vectors does not take compositions into account.
"""
function per_atom_scatter_data(tr::TrainingResults)
    nat = natoms(tr.fc)
    (tr.H_target ./ nat, tr.H_pred ./ nat)
end


"""
    each_comp(tr::TrainingResults)

Return a dictionary of `TrainingResults` objects for each composition.
"""
function each_comp(tr::TrainingResults)
    forms = [m[:formula] for m in tr.fc.metadata]
    uforms = unique(forms)
    Dict(Pair(form, tr[findall(x -> x == form, forms)]) for form in uforms)
end

"""
    ensemble_std(tr::TrainingResults{M, T};per_atom=true) where {M, T<:EnsembleNNInterface}

Return the standard deviation from the ensemble for each data point. Defaults to atomic energy.
"""
function ensemble_std(
    tr::TrainingResults{M,T};
    min_weight=0.05,
) where {M,T<:EnsembleNNInterface}
    function fvstd_atomic(fvec)
        std(
            mean(m(fvec)) for
            (m, w) in zip(tr.model.models, tr.model.weights) if w > min_weight
        )
    end
    return fvstd_atomic.(tr.fc.fvecs)
end

"""
    r2score_each_comp(tr::TrainingResults)

Compute the R2 score for each composition separately as it does not make sense to compute 
using the full dataset containing different compositions.

See also: https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score
"""
function r2score_each_comp(tr::TrainingResults)
    data = per_atom_scatter_each_comp(tr)
    out = Dict{Symbol,Float64}()
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

raw"""
Compute the loss as absolute difference in total energy

```math
L = \sum_i |y_i - y_i^p|^l
```
"""
function loss_all(model, fvecs, H; pow=2)
    if pow == 2
        (sum.(model.(fvecs)) .- H) .^ pow |> sum
    else
        abs.(sum.(model.(fvecs)) .- H) .^ pow |> sum
    end
end

"""
    generate_f_g_optim(model, fc_train, fc_test; pow=2, earlystop=30)

Generate f, g!, view of the parameters and the callback function for neuron network training using Optim.

This is for 'batch' training where all of the data are included.
"""
function generate_f_g_optim(model, fc_train, fc_test; pow=2, earlystop=30)

    X = fc_train.fvecs
    nfeat = size(X[1], 1)
    Y = transform_y(fc_train)
    xsizes = size.(X, 2)

    # Generate dataset by sizes, each dataset has X as a rank 3 tensor
    datasets = Tuple{Array{eltype(X[1]),3},Vector{eltype(Y)}}[]
    for usize in unique(xsizes)
        mask = findall(x -> x == usize, xsizes)
        this_size = X[mask]
        # A rank 3 tensor
        this_tensor = Array{eltype(X[1]),3}(undef, nfeat, usize, length(mask))
        # Copy data to the tensor
        for (idx, i) in enumerate(mask)
            this_tensor[:, :, idx] .= X[i]
        end
        this_H = Y[mask]
        push!(datasets, (this_tensor, this_H))
    end

    mdl = model.chain

    # View into the parameters of the model
    flat, rebuild = Flux.destructure(mdl)

    # Per-atom data
    Ha = fc_train.H ./ natoms(fc_train)
    Ha_test = fc_test.H ./ natoms(fc_test)

    iter::Int = 1
    min_test::Float64 = floatmax(Float64)
    min_iter::Int = 1

    function f(x)
        mdl = rebuild(x)
        #return loss_all(mdl, X, Y;pow)
        return loss_stacked(mdl, datasets; pow)
    end

    function g!(g, x)
        grad = Zygote.gradient(x) do _
            mdl = rebuild(x)
            loss_stacked(mdl, datasets; pow)
        end
        g .= grad[1]
        return g
    end

    """
    Callback for progress display and early stopping
    """
    function callback(args...; kwargs...)
        rmse =
            (
                (mean.(mdl.(fc_train.fvecs)) .* fc_train.yt.scale[1]) .+
                fc_train.yt.mean[1] .- Ha
            ) .^ 2 |>
            mean |>
            sqrt
        rmse_test =
            (
                (mean.(mdl.(fc_test.fvecs)) .* fc_test.yt.scale[1]) .+ fc_test.yt.mean[1] .-
                Ha_test
            ) .^ 2 |>
            mean |>
            sqrt
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

    return f, g!, flat, callback
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
function loss_stacked(model, datasets; pow=2)
    sum(datasets) do (x, y)
        pred = sum(model(x), dims=2)[:]
        abs.(pred .- y) .^ pow |> sum
    end
end
