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
using Flux: Dense, Chain
using .EDDPotentialsCore: leaky_tanh
import Flux
import Base
import CellBase
import Zygote
using Logging
using NonNegLeastSquares
using .EDDPotentialsCore: paramvector, save_model_artifact, ThreadBufferPool, rmse_per_atom_threaded,
    nfeatures, EDDPCalculator, predict, nparams
import .EDDPotentialsCore: EnsembleNNInterface


const XT_NAME = "xt"
const YT_NAME = "yt"
const FEATURESPEC_NAME = "cf"


"""
Genreate a `Chain` based on a vector specifying the number of hidden nodes in each layer
"""
function generate_chain(nfeature, nnodes; init=EDDPotentialsCore.ginit)
    if length(nnodes) == 0
        return Chain(Dense(nfeature => 1; init))
    end

    models = Any[Dense(nfeature => nnodes[1], leaky_tanh; bias=true, init)]
    # Add more layers
    if length(nnodes) > 1
        for i = 2:length(nnodes)
            push!(models, Dense(nnodes[i-1] => nnodes[i]; init))
        end
    end
    # Output layer
    push!(models, Dense(nnodes[end] => 1; init))
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
    train!(itf, fc_train, fc_val; maxIter=300)
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
- `x_val=x`: Validation set feature vectors (defaults to training set)
- `y_val=y`: Validation set target energies (defaults to training set)
- `earlystop=50`: Stop if validation error doesn't improve for this many iterations
- `keep_best=true`: Keep parameters with best validation error instead of final iteration
- `log_file=""`: Path to file for logging progress (empty string = no logging)
- `p=1.25`: Levenberg-Marquardt damping parameter
- `weights=nothing`: Optional weights for each structure in the loss function

# Returns
- `opt_res`: Optimization result object
- `params`: Final parameter vector
- `history`: Matrix with columns [train_rmse, val_rmse] for each iteration
- `funcs`: Tuple of (f!, j!, fj!) functions used for optimization

# Examples
```julia
# Basic training on carbon structures
itf = FluxInterface(Chain(Dense(10, 20, tanh), Dense(20, 1)))
x_train, y_train = get_fit_data(fc_train)
x_val, y_val = get_fit_data(fc_val)

# Train with progress display and early stopping
opt_res, params, history = train_lm!(
    itf, x_train, y_train;
    x_val=x_val, y_val=y_val,
    maxIter=500,
    show_progress=true,
    earlystop=30
)

# Check final RMSE
final_train_rmse = history[end, 1]
final_val_rmse = history[end, 2]
```

# Notes
- The network parameters in `itf` are updated in-place during training
- Training minimizes RMSE of total energies, not per-atom energies
- Early stopping uses validation set RMSE to prevent overfitting
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
    x_val=x,
    y_val=y,
    earlystop=50,
    keep_best=true,
    log_file="",
    p=1.25,
    weights=nothing,
    checkpoint_every::Int=0,
    checkpoint_prefix::AbstractString="",
    feature_spec=nothing,
    elemental_energies::Dict=Dict{Symbol,Float64}(),
    solver::Symbol=:cholesky,
    args...,
)
    rec = []

    train_natoms = [size(v, 2) for v in x]
    val_natoms = [size(v, 2) for v in x_val]

    time_start = time()
    last_time = time()
    iter_count = 1
    function progress_tracker()
        rmse_train = rmse_per_atom(itf, x, y, train_natoms)

        if x_val === x
            rmse_val = rmse_train
        else
            rmse_val = rmse_per_atom(itf, x_val, y_val, val_natoms)
        end
        if show_progress || (log_file != "")
            tnow = time()
            elapsed = tnow - time_start
            loop = tnow - last_time
            last_time = tnow
            logline =
                @sprintf "Iter: %d %3.3f %3.3f RMSE Train %10.5f eV | Val %10.5f eV\n" iter_count loop elapsed rmse_train rmse_val
            show_progress && print(logline)
            if log_file != ""
                open(log_file, "a") do file
                    write(file, logline)
                end
            end
        end

        flush(stdout)
        push!(rec, (rmse_train, rmse_val))

        # Checkpoint
        if checkpoint_every > 0 && checkpoint_prefix != "" && feature_spec !== nothing && iter_count % checkpoint_every == 0
            ckpt_path = _ensure_dir(@sprintf("%s_%04d.json", checkpoint_prefix, iter_count))
            save_model_artifact(ckpt_path, itf; feature_spec, elemental_energies, metadata=Dict("epoch" => iter_count))
        end

        iter_count += 1

        rmse_val, paramvector(itf)
    end

    # Setting up the object for minimization
    f!, j!, fj! = setup_fj(itf, x, y, weights)
    od2 = OnceDifferentiable(f!, j!, fj!, p0, zeros(eltype(x[1]), length(x)), inplace=true)

    callback = show_progress || (earlystop > 0) || checkpoint_every > 0 ? progress_tracker : nothing

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
        solver,
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
- `max_epochs=300`: Maximum number of training epochs
- `show_progress=false`: Print progress each epoch
- `x_val=x`: Validation set feature vectors
- `y_val=y`: Validation set target energies
- `earlystop=50`: Stop if validation RMSE hasn't improved for this many epochs
- `keep_best=true`: Keep parameters with best validation RMSE
- `log_file=""`: Path to log file (empty = no logging)
- `p=1.25`: Levenberg-Marquardt damping parameter
- `weights=nothing`: Optional per-structure weights
- `batch_size=512`: Number of samples per mini-batch
- `shuffle_seed=42`: RNG seed for reproducible shuffling

# Returns
- `opt_res`: Optimization result object
- `params`: Final parameter vector
- `history`: Matrix with columns [train_rmse, val_rmse] per epoch
- `funcs`: Tuple of (fj_batch!, f_full!) closures

# See also
[`train_lm!`](@ref), [`train!`](@ref)
"""
function train_lm_batched!(
    itf::AbstractNNInterface,
    x,
    y;
    p0=paramvector(itf),
    max_epochs=300,
    maxIter::Union{Integer,Nothing}=nothing,
    show_progress=false,
    x_val=x,
    y_val=y,
    earlystop=50,
    keep_best=true,
    log_file="",
    p=1.25,
    weights=nothing,
    batch_size=512,
    shuffle_seed=42,
    threaded::Bool=(Threads.nthreads() > 1),
    damping_mode::Symbol=:fletcher,
    attempts_per_step::Int=1,
    solver::Symbol=:direct,
    batch_ranges::Union{Nothing,Vector{<:UnitRange{Int}}}=nothing,
    checkpoint_every::Int=0,
    checkpoint_prefix::AbstractString="",
    feature_spec=nothing,
    elemental_energies::Dict=Dict{Symbol,Float64}(),
    args...,
)
    # --- Deprecated alias ---
    if maxIter !== nothing
        @warn "maxIter is deprecated, use max_epochs instead" maxIter max_epochs
        max_epochs = maxIter
    end
    # Filter maxIter from args to avoid double deprecation warning downstream
    args = Base.structdiff(NamedTuple(args), NamedTuple{(:maxIter,)})

    rec = []

    train_natoms = x isa FeatureStore ? natoms(x) : [size(v, 2) for v in x]
    val_natoms = x_val isa FeatureStore ? natoms(x_val) : [size(v, 2) for v in x_val]

    time_start = time()
    last_time = time()
    epoch_count = 1
    val_pool = nothing
    function progress_tracker()
        if val_pool !== nothing
            rmse_train = rmse_per_atom_threaded(val_pool, x, y, train_natoms)
        else
            rmse_train = rmse_per_atom(itf, x, y, train_natoms)
        end

        if x_val === x
            rmse_val = rmse_train
        else
            if val_pool !== nothing
                rmse_val = rmse_per_atom_threaded(val_pool, x_val, y_val, val_natoms)
            else
                rmse_val = rmse_per_atom(itf, x_val, y_val, val_natoms)
            end
        end
        if show_progress || (log_file != "")
            tnow = time()
            elapsed = tnow - time_start
            loop = tnow - last_time
            last_time = tnow
            logline =
                @sprintf "Epoch: %d %3.3f %3.3f RMSE Train %10.5f eV | Val %10.5f eV (batch=%d)\n" epoch_count loop elapsed rmse_train rmse_val batch_size
            show_progress && print(logline)
            if log_file != ""
                open(log_file, "a") do file
                    write(file, logline)
                end
            end
        end

        flush(stdout)
        push!(rec, (rmse_train, rmse_val))

        # Checkpoint
        if checkpoint_every > 0 && checkpoint_prefix != "" && feature_spec !== nothing && epoch_count % checkpoint_every == 0
            ckpt_path = _ensure_dir(@sprintf("%s_%04d.json", checkpoint_prefix, epoch_count))
            save_model_artifact(ckpt_path, itf; feature_spec, elemental_energies, metadata=Dict("epoch" => epoch_count))
        end

        epoch_count += 1

        rmse_val, paramvector(itf)
    end

    # Set up batched closures
    if threaded
        fj_batch!, f_full!, val_pool = setup_fj_batched_threaded(itf, x, y, weights)
    else
        fj_batch!, f_full! = setup_fj_batched(itf, x, y, weights)
    end

    callback = show_progress || (earlystop > 0) || checkpoint_every > 0 ? progress_tracker : nothing

    opt_res = @timeit to "batched lm solve" levenberg_marquardt_batched(
        fj_batch!,
        f_full!,
        p0;
        n_data=length(x),
        batch_size=batch_size,
        batch_ranges=batch_ranges,
        max_epochs=max_epochs,
        callback=callback,
        p=p,
        keep_best=keep_best,
        earlystop=earlystop,
        damping_mode=damping_mode,
        attempts_per_step=attempts_per_step,
        solver=solver,
        shuffle_seed=shuffle_seed,
        args...,
    )

    opt_res, paramvector(itf), [map(x -> x[1], rec) map(x -> x[2], rec)], (fj_batch!, f_full!)
end

"""
    train!(itf::AbstractNNInterface, fc_train::FeatureContainer, fc_val::FeatureContainer; kwargs...)

Train a neural network model using training and validation datasets.

This is the main high-level training function that users should call. It automatically
extracts features and energies from FeatureContainers and dispatches to the appropriate
training algorithm (Levenberg-Marquardt by default).

# Arguments
- `itf::AbstractNNInterface`: Neural network interface to train
- `fc_train::FeatureContainer`: Training dataset with features and energies
- `fc_val::FeatureContainer`: Validation dataset for monitoring during training

# Keyword Arguments
- `train_method="lm"`: Training algorithm ("lm" for Levenberg-Marquardt, "lm_batched" for batched LM, "optim" for Optim.jl, "sgd" for SGD/Adam)
- Additional kwargs are passed to the underlying training function

# Returns
Returns from underlying training method (`train_lm!` or Optim optimization)

# Examples
```julia
# Load training and validation data
using ExtXYZ
structures_train = read("carbon_train.xyz", ExtXYZ)
structures_val = read("carbon_val.xyz", ExtXYZ)

# Create FeatureContainers
cf = CellFeature([:C])  # Define features for carbon
sc_train = StructureContainer(structures_train)
sc_val = StructureContainer(structures_val)
fc_train = FeatureContainer(cf, sc_train)
fc_val = FeatureContainer(cf, sc_val)

# Create and train model
itf = FluxInterface(Chain(Dense(10, 20, tanh), Dense(20, 1)))
train!(itf, fc_train, fc_val; maxIter=500, show_progress=true)

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
    fc_val::FeatureContainer;
    train_method="lm",
    kwargs...,
)
    # Inject feature_spec and elemental_energies from the FeatureContainer if not provided
    kw = NamedTuple(kwargs)
    if !haskey(kw, :feature_spec)
        kw = merge(kw, (feature_spec=fc_train.feature,))
    end
    if !haskey(kw, :elemental_energies)
        kw = merge(kw, (elemental_energies=fc_train.elemental_energies,))
    end

    if train_method == "lm"
        x_train, y_train = get_fit_data(fc_train)
        x_val, y_val = get_fit_data(fc_val)
        train_lm!(itf, x_train, y_train; x_val, y_val, kw...)
    elseif train_method == "lm_batched"
        x_train, y_train = get_fit_data(fc_train)
        x_val, y_val = get_fit_data(fc_val)
        train_lm_batched!(itf, x_train, y_train; x_val, y_val, kw...)
    elseif train_method == "sgd"
        x_train, y_train = get_fit_data(fc_train)
        x_val, y_val = get_fit_data(fc_val)
        train_sgd!(itf, x_train, y_train; x_val, y_val, kw...)
    elseif train_method == "optim"
        model = get_flux_model(itf)
        f, g!, pview, callback = EDDPotentials.generate_f_g_optim(model, fc_train, fc_val)
        od = OnceDifferentiable(f, g!, collect(pview))
        x0 = collect(pview)
        opt_res = Optim.optimize(od, x0; callback=callback, kwargs...)
        opt_res, collect(pview)
    end
end

"""
    train!(itf::AbstractNNInterface, store::FeatureStore, fc_val; kwargs...)

Train a neural network using a `FeatureStore` (disk-backed features) as the
training data.  Uses the batched LM algorithm with fixed-batch shuffle for
efficient sequential I/O.

When `store` is a `FeatureStore`, the training data is loaded from disk on
demand with per-batch caching.  The `batch_ranges` from the store are passed
to the LM solver so that only the batch order is shuffled each epoch (not the
per-structure indices), enabling sequential reads within each batch.

# Keyword Arguments
- `train_method`: `"lm_batched"` or `"sgd"` for `FeatureStore`.
- `batch_size`: Ignored — the store's pre-computed `batch_ranges` are used
  directly.  To change the batch size, regenerate the `FeatureStore` with
  a different `batch_size`.
- All other kwargs are passed to `train_lm_batched!`.

# See also
[`FeatureStore`](@ref), [`train_lm_batched!`](@ref), [`train!`](@ref)
"""
function train!(
    itf::AbstractNNInterface,
    store::FeatureStore,
    fc_val::FeatureContainer;
    train_method="lm_batched",
    kwargs...,
)
    # Inject feature_spec and elemental_energies from the FeatureStore if not provided
    kw = NamedTuple(kwargs)
    if !haskey(kw, :feature_spec)
        kw = merge(kw, (feature_spec=store.feature,))
    end
    if !haskey(kw, :elemental_energies)
        kw = merge(kw, (elemental_energies=store.elemental_energies,))
    end

    if train_method == "lm_batched"
        x_train, y_train = get_fit_data(store)
        x_val, y_val = get_fit_data(fc_val)
        train_lm_batched!(
            itf, x_train, y_train;
            x_val=x_val, y_val=y_val,
            batch_ranges=store.batch_ranges,
            kw...,
        )
    elseif train_method == "sgd"
        x_train, y_train = get_fit_data(store)
        x_val, y_val = get_fit_data(fc_val)
        train_sgd!(itf, x_train, y_train; x_val, y_val, kw...)
    else
        error("FeatureStore only supports train_method=\"lm_batched\" or \"sgd\"")
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
function worker_train_one(model, train, val, jobs_channel, results_channel; kwargs...)
    while true
        job_id = take!(jobs_channel)
        # Signals no more work to do
        if job_id < 0
            break
        end
        new_model = reinit(model)
        out = train!(new_model, train, val; kwargs...)
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
    generate_f_g_optim(model, fc_train, fc_val; pow=2, earlystop=30)

Generate f, g!, view of the parameters and the callback function for neuron network training using Optim.

This is for 'batch' training where all of the data are included.
"""
function generate_f_g_optim(model, fc_train, fc_val; pow=2, earlystop=30)

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
    Ha_val = fc_val.H ./ natoms(fc_val)

    iter::Int = 1
    min_val::Float64 = floatmax(Float64)
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
        rmse_val =
            (
                (mean.(mdl.(fc_val.fvecs)) .* fc_val.yt.scale[1]) .+ fc_val.yt.mean[1] .-
                Ha_val
            ) .^ 2 |>
            mean |>
            sqrt
        @info "Iter $(iter) - RMSE $(round(rmse, digits=5)) eV / $(round(rmse_val, digits=5)) eV"

        if rmse_val < min_val
            min_iter = iter
        end

        if iter - min_iter > earlystop
            @info "Early stop condition triggered - last best validation was $(earlystop) iterations ago"
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

# =====================================================================
# TrainedModel — high-level training result container
# =====================================================================

"""
    TrainedModel{M,F}

Result of [`train_model`](@ref). Bundles a trained model, feature
specification, elemental energy offsets, and evaluation results.

Can be:
- Saved with `save_model_artifact(path, tm)`
- Converted to a calculator with `EDDPCalculator(tm; kwargs...)`
- Used for one-shot prediction with `predict(tm, cell; kwargs...)`
"""
struct TrainedModel{M,F}
    model::M
    feature_spec::F
    elemental_energies::Dict{Symbol,Float64}
    train_results::TrainingResults
    val_results::TrainingResults
    test_results::TrainingResults
end

function Base.show(io::IO, ::MIME"text/plain", tm::TrainedModel)
    println(io, "TrainedModel")
    println(io, "  Architecture: ", typeof(tm.model).name.name)
    println(io, "  Feature spec: ", typeof(tm.feature_spec).name.name)
    if !isempty(tm.elemental_energies)
        print(io, "  Elemental energies: ")
        for (k, v) in sort(collect(tm.elemental_energies), by=first)
            print(io, "$k=", round(v; digits=4), " ")
        end
        println(io)
    end
    println(io, "--- Training ---")
    show(io, MIME("text/plain"), tm.train_results)
    println(io, "\n--- Validation ---")
    show(io, MIME("text/plain"), tm.val_results)
    println(io, "\n--- Test ---")
    show(io, MIME("text/plain"), tm.test_results)
end

function EDDPCalculator(tm::TrainedModel; kwargs...)
    EDDPCalculator(tm.feature_spec, tm.model, tm.elemental_energies; kwargs...)
end

function save_model_artifact(path::AbstractString, tm::TrainedModel; metadata=nothing)
    save_model_artifact(
        path, tm.model;
        feature_spec=tm.feature_spec,
        elemental_energies=tm.elemental_energies,
        metadata,
    )
end

"""
    predict(tm::TrainedModel, cell::Cell; ...) -> NamedTuple{(:energy, :forces, :stress),...}

One-shot energy, forces, and stress prediction. Internally constructs a
temporary calculator from the trained model and feature spec.
"""
function predict(
    tm::TrainedModel,
    cell::Cell;
    forces::Bool=true,
    stress::Bool=true,
    shell_size::Float64=2.0,
    nmax::Int=500,
)
    calc = EDDPCalculator(tm; shell_size, nmax)
    predict(calc, cell; forces, stress)
end

# =====================================================================
# train_model — single-call training
# =====================================================================

function _print_train_config(fc_train, ntrain, nval, ntest, nnodes, ninp, nparam, train_method, kwargs)
    cf = fc_train.feature
    ntotal = ntrain + nval + ntest
    printstyled("Training Configuration\n"; bold=true)
    println("─"^50)
    println("  Structures : $ntotal total  →  $ntrain train / $nval val / $ntest test")
    println("  Species    : $(join(string.(cf.elements), ", "))")
    if cf isa CellFeature
        if !isempty(cf.two_body)
            tbf = cf.two_body[1]
            println("  2-body     : p=$(tbf.p)  rcut=$(tbf.rcut)")
        end
        if !isempty(cf.three_body)
            tbf3 = cf.three_body[1]
            println("  3-body     : p=$(tbf3.p)  q=$(tbf3.q)  rcut=$(tbf3.rcut)")
        end
    end
    println("  Features   : $(ninp) per atom  ($(typeof(cf).name.name))")
    println("  Network    : $(nnodes) hidden  →  $nparam parameters")
    bs = get(kwargs, :batch_size, nothing)
    if bs !== nothing
        println("  Method     : $train_method  batch_size=$bs")
    else
        println("  Method     : $train_method")
    end
    maxe = get(kwargs, :max_epochs, get(kwargs, :maxIter, nothing))
    if maxe !== nothing
        println("  Max epochs : $maxe")
    end
    estop = get(kwargs, :earlystop, nothing)
    if estop !== nothing && estop > 0
        println("  Early stop : $estop")
    end
    println("─"^50)
end

function _print_train_results(train_results, val_results, test_results)
    println()
    printstyled("Training Results\n"; bold=true)
    println("─"^50)
    @printf("  Train RMSE/atom : %.5f eV\n", rmse_per_atom(train_results))
    @printf("  Train MAE/atom  : %.5f eV\n", mae_per_atom(train_results))
    @printf("  Val   RMSE/atom : %.5f eV\n", rmse_per_atom(val_results))
    @printf("  Val   MAE/atom  : %.5f eV\n", mae_per_atom(val_results))
    @printf("  Test  RMSE/atom : %.5f eV\n", rmse_per_atom(test_results))
    @printf("  Test  MAE/atom  : %.5f eV\n", mae_per_atom(test_results))
    println("─"^50)
end

# --- Internal helpers ---

function _unique_symbols(structures)
    symbols = Set{Symbol}()
    for s in structures
        union!(symbols, Set(CellBase.species(s)))
    end
    sort!(collect(symbols))
end

function _infer_feature_spec(sc, feature_kwargs)
    symbols = _unique_symbols(sc.structures)
    CellFeature(symbols; feature_kwargs...)
end

function _sc_from_paths(data; threshold=10.0, pressure_gpa=0.0)
    paths = data isa Vector ? data : [data]
    StructureContainer(paths; threshold, pressure_gpa)
end

function _prepare_split(sc::StructureContainer, feature_spec;
                        val_ratio::Float64, test_ratio::Float64, seed::Int,
                        elemental_energies, show_progress, nmax)
    val_ratio > 0 || throw(ArgumentError("val_ratio must be > 0"))
    test_ratio > 0 || throw(ArgumentError("test_ratio must be > 0"))
    N = length(sc)
    nval = Int(floor(N * val_ratio))
    ntest = Int(floor(N * test_ratio))
    nval > 0 || throw(ArgumentError(
        "val_ratio=$val_ratio yields 0 validation structures from $N total"))
    ntest > 0 || throw(ArgumentError(
        "test_ratio=$test_ratio yields 0 test structures from $N total"))
    ntrain = N - nval - ntest
    ntrain > 0 || throw(ArgumentError(
        "Not enough structures ($N) for val_ratio=$val_ratio + test_ratio=$test_ratio"))
    split_and_featurize(sc, feature_spec, ntrain, nval, ntest;
        seed, elemental_energies, show_progress, nmax)
end

function _prepare_presplit(scs, feature_spec;
                           elemental_energies, show_progress, nmax)
    fcs = [FeatureContainer(sc, feature_spec;
            elemental_energies, show_progress, nmax) for sc in scs]
    standardize!(fcs...)
    Tuple(fcs)
end

function _prepare_fc_split(fc::FeatureContainer; val_ratio::Float64, test_ratio::Float64, seed::Int)
    val_ratio > 0 || throw(ArgumentError("val_ratio must be > 0"))
    test_ratio > 0 || throw(ArgumentError("test_ratio must be > 0"))
    N = length(fc)
    nval = Int(floor(N * val_ratio))
    ntest = Int(floor(N * test_ratio))
    nval > 0 || throw(ArgumentError(
        "val_ratio=$val_ratio yields 0 validation structures from $N total"))
    ntest > 0 || throw(ArgumentError(
        "test_ratio=$test_ratio yields 0 test structures from $N total"))
    ntrain = N - nval - ntest
    ntrain > 0 || throw(ArgumentError(
        "Not enough structures ($N) for val_ratio=$val_ratio + test_ratio=$test_ratio"))
    fc_clean = _reset_transforms(fc)
    split(fc_clean, ntrain, nval, ntest; standardize=true, seed)
end

function _reset_transforms(fc::FeatureContainer)
    FeatureContainer(
        fc.fvecs,
        fc.feature,
        fc.H,
        fc.labels,
        fc.metadata,
        nothing,
        nothing,
        fc.elemental_energies,
    )
end

# =====================================================================
# Core method: train_model on 3 pre-split FeatureContainers
# =====================================================================

"""
    train_model(fc_train, fc_val, fc_test; ...) -> TrainedModel

Core training entry point operating on three pre-split, pre-standardized
`FeatureContainer`s. All other `train_model` overloads delegate here.

Training uses `fc_train` and `fc_val`; `fc_test` is evaluated once after
training completes. The model, feature spec, elemental energies, and
evaluation results on all three splits are bundled into the returned
[`TrainedModel`](@ref).

# Arguments

- `fc_train::FeatureContainer`: training set
- `fc_val::FeatureContainer`: validation set (used for early stopping)
- `fc_test::FeatureContainer`: test set (evaluated once post-training)

# Keyword arguments

- `nnodes=[10, 10]`: hidden layer sizes passed to [`generate_chain`](@ref).
- `train_method="lm_batched"`: training algorithm (`"lm"`, `"lm_batched"`,
  `"sgd"`, or `"optim"`).
- `verbose=true`: print training configuration and results.
- `show_progress=true`: show progress during feature computation and training.
- All other kwargs are forwarded to [`train!`](@ref).

# See also

[`train!`](@ref), [`TrainedModel`](@ref), [`FeatureContainer`](@ref)
"""
function train_model(
    fc_train::FeatureContainer,
    fc_val::FeatureContainer,
    fc_test::FeatureContainer;
    nnodes::Vector{Int}=[10, 10],
    train_method::String="lm_batched",
    verbose::Bool=true,
    show_progress::Bool=true,
    kwargs...,
)
    ninp = nfeatures(fc_train.feature)
    chain = generate_chain(ninp, nnodes)
    itf = ManualFluxBackPropInterface(chain, xt=fc_train.xt, yt=fc_train.yt)

    ntrain = length(fc_train)
    nval = length(fc_val)
    ntest = length(fc_test)

    if verbose
        _print_train_config(fc_train, ntrain, nval, ntest, nnodes, ninp, nparams(itf), train_method, kwargs)
    end

    train!(itf, fc_train, fc_val; train_method, show_progress, kwargs...)

    train_results = TrainingResults(itf, fc_train)
    val_results = TrainingResults(itf, fc_val)
    test_results = TrainingResults(itf, fc_test)

    if verbose
        _print_train_results(train_results, val_results, test_results)
    end

    TrainedModel(
        itf,
        fc_train.feature,
        fc_train.elemental_energies,
        train_results,
        val_results,
        test_results,
    )
end

# =====================================================================
# Single-data wrappers (built-in 3-way split)
# =====================================================================

"""
    train_model(data, feature_spec; val_ratio=0.1, test_ratio=0.1, ...) -> TrainedModel
    train_model(data; val_ratio=0.1, test_ratio=0.1, p2=2:8, ...) -> TrainedModel
    train_model(sc, feature_spec; ...) -> TrainedModel
    train_model(sc; p2=2:8, ...) -> TrainedModel
    train_model(fc; val_ratio=0.1, test_ratio=0.1, ...) -> TrainedModel

Train an EDDP model from data in a single call.  Handles structure loading,
feature computation, 3-way data splitting (train / val / test), standardisation,
model construction, and training.

The test set is evaluated once after training completes — it is never used
during the training loop.

# Arguments

- `data`: file paths (glob patterns or `@:N` slice syntax), a
  `StructureContainer`, or a pre-computed `FeatureContainer`.
- `feature_spec::AbstractCellFeatureSpec`: the atomic environment
  descriptor. **Optional** — if omitted, species are inferred from the
  loaded structures and default feature parameters are used.  Feature
  parameters (`p2`, `p3`, `q3`, `rcut2`, `rcut3`) can be passed as
  keyword arguments instead.

# Keyword arguments

- `val_ratio=0.1`: fraction of data reserved for validation.
- `test_ratio=0.1`: fraction of data reserved for final testing.
- `seed=42`: RNG seed for reproducible shuffling of the split.
- `nnodes=[10, 10]`: hidden layer sizes.
- `train_method="lm_batched"`: training algorithm (`"lm"`, `"lm_batched"`,
  `"sgd"`, or `"optim"`).
- `verbose=true`: print training configuration and results.
- `elemental_energies`: per-element reference energy offsets (subtracted
  from target energies during training, added back at inference).
- `threshold=10.0`: per-atom energy ceiling for filtering (path-based).
- `pressure_gpa=0.0`: external pressure for enthalpy conversion (path-based).
- `show_progress=true`: show progress bar during feature computation.
- `nmax=500`: maximum neighbour list entries.
- `p2=2:8`, `p3=2:8`, `q3=2:8`: feature exponents (auto-inferred spec only).
- `rcut2=4.0`, `rcut3=4.0`: feature cutoff radii (auto-inferred spec only).
- All other kwargs are forwarded to [`train!`](@ref).

# Pre-split data

You can also pass pre-split data directly, bypassing the built-in split:

```julia
# From pre-split StructureContainers
train_model(sc_train, sc_val, sc_test, feature_spec; ...)
train_model(sc_train, sc_val, sc_test; p2=2:12, rcut2=5.0, ...)

# From pre-split file paths
train_model(path_train, path_val, path_test, feature_spec; ...)
train_model(path_train, path_val, path_test; p2=2:12, ...)

# From pre-split FeatureContainers
train_model(fc_train, fc_val, fc_test; ...)
```

# Example
```julia
# With explicit feature spec
tm = train_model("data/*.res",
    CellFeature([:Al, :N, :Sc]; p2=2:12, p3=2:4, q3=2:4);
    nnodes=[20, 20], max_epochs=500)

# Auto-inferred species, custom feature parameters
tm = train_model("data/*.res";
    p2=2:12, p3=2:4, q3=2:4, rcut2=5.0, rcut3=4.0,
    nnodes=[20, 20], max_epochs=500)

save_model_artifact("model.json", tm)
calc = EDDPCalculator(tm)
result = predict(tm, my_cell)
```
"""

# Path with explicit feature_spec
function train_model(
    data::Union{AbstractString,Vector{<:AbstractString}},
    feature_spec::AbstractCellFeatureSpec;
    val_ratio::Float64=0.1,
    test_ratio::Float64=0.1,
    seed::Int=42,
    nnodes::Vector{Int}=[10, 10],
    train_method::String="lm_batched",
    elemental_energies::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
    threshold::Float64=10.0,
    pressure_gpa::Float64=0.0,
    show_progress::Bool=true,
    nmax::Int=500,
    verbose::Bool=true,
    kwargs...,
)
    sc = _sc_from_paths(data; threshold, pressure_gpa)
    verbose && println("Loaded $(length(sc)) structures after filtering.")
    fc_train, fc_val, fc_test = _prepare_split(
        sc, feature_spec; val_ratio, test_ratio, seed, elemental_energies, show_progress, nmax)
    return train_model(fc_train, fc_val, fc_test;
        nnodes, train_method, verbose, show_progress, kwargs...)
end

# Path, auto-infer feature spec
function train_model(
    data::Union{AbstractString,Vector{<:AbstractString}};
    val_ratio::Float64=0.1,
    test_ratio::Float64=0.1,
    seed::Int=42,
    nnodes::Vector{Int}=[10, 10],
    train_method::String="lm_batched",
    elemental_energies::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
    threshold::Float64=10.0,
    pressure_gpa::Float64=0.0,
    show_progress::Bool=true,
    nmax::Int=500,
    verbose::Bool=true,
    p2=2:8,
    p3=2:8,
    q3=2:8,
    rcut2=4.0,
    rcut3=4.0,
    kwargs...,
)
    sc = _sc_from_paths(data; threshold, pressure_gpa)
    verbose && println("Loaded $(length(sc)) structures after filtering.")
    feature_spec = _infer_feature_spec(sc, (; p2, p3, q3, rcut2, rcut3))
    fc_train, fc_val, fc_test = _prepare_split(
        sc, feature_spec; val_ratio, test_ratio, seed, elemental_energies, show_progress, nmax)
    return train_model(fc_train, fc_val, fc_test;
        nnodes, train_method, verbose, show_progress, kwargs...)
end

# StructureContainer with explicit feature_spec
function train_model(
    sc::StructureContainer,
    feature_spec::AbstractCellFeatureSpec;
    val_ratio::Float64=0.1,
    test_ratio::Float64=0.1,
    seed::Int=42,
    nnodes::Vector{Int}=[10, 10],
    train_method::String="lm_batched",
    elemental_energies::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
    show_progress::Bool=true,
    nmax::Int=500,
    verbose::Bool=true,
    kwargs...,
)
    fc_train, fc_val, fc_test = _prepare_split(
        sc, feature_spec; val_ratio, test_ratio, seed, elemental_energies, show_progress, nmax)
    return train_model(fc_train, fc_val, fc_test;
        nnodes, train_method, verbose, show_progress, kwargs...)
end

# StructureContainer, auto-infer feature spec
function train_model(
    sc::StructureContainer;
    val_ratio::Float64=0.1,
    test_ratio::Float64=0.1,
    seed::Int=42,
    nnodes::Vector{Int}=[10, 10],
    train_method::String="lm_batched",
    elemental_energies::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
    show_progress::Bool=true,
    nmax::Int=500,
    verbose::Bool=true,
    p2=2:8,
    p3=2:8,
    q3=2:8,
    rcut2=4.0,
    rcut3=4.0,
    kwargs...,
)
    feature_spec = _infer_feature_spec(sc, (; p2, p3, q3, rcut2, rcut3))
    fc_train, fc_val, fc_test = _prepare_split(
        sc, feature_spec; val_ratio, test_ratio, seed, elemental_energies, show_progress, nmax)
    return train_model(fc_train, fc_val, fc_test;
        nnodes, train_method, verbose, show_progress, kwargs...)
end

# FeatureContainer with built-in split
function train_model(
    fc::FeatureContainer;
    val_ratio::Float64=0.1,
    test_ratio::Float64=0.1,
    seed::Int=42,
    nnodes::Vector{Int}=[10, 10],
    train_method::String="lm_batched",
    verbose::Bool=true,
    show_progress::Bool=true,
    kwargs...,
)
    fc_train, fc_val, fc_test = _prepare_fc_split(fc; val_ratio, test_ratio, seed)
    return train_model(fc_train, fc_val, fc_test;
        nnodes, train_method, verbose, show_progress, kwargs...)
end

# =====================================================================
# Pre-split wrappers
# =====================================================================

# --- Pre-split StructureContainers with explicit feature_spec ---

function train_model(
    sc_train::StructureContainer,
    sc_val::StructureContainer,
    sc_test::StructureContainer,
    feature_spec::AbstractCellFeatureSpec;
    nnodes::Vector{Int}=[10, 10],
    train_method::String="lm_batched",
    elemental_energies::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
    show_progress::Bool=true,
    nmax::Int=500,
    verbose::Bool=true,
    kwargs...,
)
    fcs = _prepare_presplit([sc_train, sc_val, sc_test], feature_spec;
        elemental_energies, show_progress, nmax)
    return train_model(fcs[1], fcs[2], fcs[3];
        nnodes, train_method, verbose, show_progress, kwargs...)
end

# --- Pre-split StructureContainers, auto-infer feature spec ---

function train_model(
    sc_train::StructureContainer,
    sc_val::StructureContainer,
    sc_test::StructureContainer;
    nnodes::Vector{Int}=[10, 10],
    train_method::String="lm_batched",
    elemental_energies::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
    show_progress::Bool=true,
    nmax::Int=500,
    verbose::Bool=true,
    p2=2:8,
    p3=2:8,
    q3=2:8,
    rcut2=4.0,
    rcut3=4.0,
    kwargs...,
)
    feature_spec = _infer_feature_spec(sc_train, (; p2, p3, q3, rcut2, rcut3))
    fcs = _prepare_presplit([sc_train, sc_val, sc_test], feature_spec;
        elemental_energies, show_progress, nmax)
    return train_model(fcs[1], fcs[2], fcs[3];
        nnodes, train_method, verbose, show_progress, kwargs...)
end

# --- Pre-split paths with explicit feature_spec ---

function train_model(
    path_train::Union{AbstractString,Vector{<:AbstractString}},
    path_val::Union{AbstractString,Vector{<:AbstractString}},
    path_test::Union{AbstractString,Vector{<:AbstractString}},
    feature_spec::AbstractCellFeatureSpec;
    nnodes::Vector{Int}=[10, 10],
    train_method::String="lm_batched",
    elemental_energies::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
    threshold::Float64=10.0,
    pressure_gpa::Float64=0.0,
    show_progress::Bool=true,
    nmax::Int=500,
    verbose::Bool=true,
    kwargs...,
)
    sc_train = _sc_from_paths(path_train; threshold, pressure_gpa)
    sc_val = _sc_from_paths(path_val; threshold, pressure_gpa)
    sc_test = _sc_from_paths(path_test; threshold, pressure_gpa)
    verbose && println("Loaded $(length(sc_train)) train / $(length(sc_val)) val / $(length(sc_test)) test structures.")
    fcs = _prepare_presplit([sc_train, sc_val, sc_test], feature_spec;
        elemental_energies, show_progress, nmax)
    return train_model(fcs[1], fcs[2], fcs[3];
        nnodes, train_method, verbose, show_progress, kwargs...)
end

# --- Pre-split paths, auto-infer feature spec ---

function train_model(
    path_train::Union{AbstractString,Vector{<:AbstractString}},
    path_val::Union{AbstractString,Vector{<:AbstractString}},
    path_test::Union{AbstractString,Vector{<:AbstractString}};
    nnodes::Vector{Int}=[10, 10],
    train_method::String="lm_batched",
    elemental_energies::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
    threshold::Float64=10.0,
    pressure_gpa::Float64=0.0,
    show_progress::Bool=true,
    nmax::Int=500,
    verbose::Bool=true,
    p2=2:8,
    p3=2:8,
    q3=2:8,
    rcut2=4.0,
    rcut3=4.0,
    kwargs...,
)
    sc_train = _sc_from_paths(path_train; threshold, pressure_gpa)
    sc_val = _sc_from_paths(path_val; threshold, pressure_gpa)
    sc_test = _sc_from_paths(path_test; threshold, pressure_gpa)
    verbose && println("Loaded $(length(sc_train)) train / $(length(sc_val)) val / $(length(sc_test)) test structures.")
    feature_spec = _infer_feature_spec(sc_train, (; p2, p3, q3, rcut2, rcut3))
    fcs = _prepare_presplit([sc_train, sc_val, sc_test], feature_spec;
        elemental_energies, show_progress, nmax)
    return train_model(fcs[1], fcs[2], fcs[3];
        nnodes, train_method, verbose, show_progress, kwargs...)
end

#=
Training with learnable species embedding.

These entry points co-optimize the NN parameters and the species embedding
matrix in a single LM loop. Features are recomputed every step using the
current embedding. The combined parameter vector is [nn_params; vec(embedding)].
=#

"""
    train_lm_with_embedding!(itf, structures, cf, y; kwargs...) ->
        (opt_res, params, history, funcs)

Train a model using full-dataset LM with learnable species embedding.

The embedding matrix in `cf` is updated in-place during training. The combined
parameter vector `[nn_params; vec(cf.embedding)]` is optimized jointly.

# Arguments
- `itf::AbstractNNInterface`: Neural network interface
- `structures::AbstractVector{<:Cell}`: Training structures
- `cf::RandomEmbeddedCellFeature`: Feature spec with mutable embedding
- `y::AbstractVector`: Target total energies

# Keyword Arguments
- `p0_nn=paramvector(itf)`: Initial NN parameters
- `structures_val=structures`: Validation structures
- `y_val=y`: Validation target energies
- `normalize_embedding=false`: Normalize embedding rows after training
- `recompute_xt=false`: Recompute feature standardization each iteration
- Remaining kwargs passed to `levenberg_marquardt`
"""
function train_lm_with_embedding!(
    itf::AbstractNNInterface,
    structures::AbstractVector{<:Cell},
    cf::Union{RandomEmbeddedCellFeature,MultiRandomEmbeddedCellFeature},
    y::AbstractVector;
    p0_nn=paramvector(itf),
    structures_val=structures,
    y_val=y,
    normalize_embedding::Bool=false,
    recompute_xt::Bool=false,
    maxIter=1000,
    show_progress=false,
    earlystop=50,
    keep_best=true,
    log_file="",
    p=1.25,
    weights=nothing,
    checkpoint_every::Int=0,
    checkpoint_prefix::AbstractString="",
    elemental_energies::Dict=Dict{Symbol,Float64}(),
    args...,
)
    rec = []
    n_nn = length(p0_nn)
    n_emb = EDDPotentialsCore._emb_param_count(cf)
    n_total = n_nn + n_emb

    # Combined initial parameter vector
    p0 = vcat(p0_nn, EDDPotentialsCore._emb_param_vec(cf))

    train_natoms = [natoms(s) for s in structures]
    val_natoms = [natoms(s) for s in structures_val]

    # Pre-build NLs/workspaces for RMSE callback (avoids rebuilding each iteration)
    train_nls, train_wks = _precompute_nl_workspace(structures, cf)
    if structures_val !== structures
        val_nls, val_wks = _precompute_nl_workspace(structures_val, cf)
    else
        val_nls, val_wks = train_nls, train_wks
    end

    time_start = time()
    last_time = time()
    iter_count = 1

    function progress_tracker()
        rmse_train = rmse_per_atom_with_embedding(itf, structures, y, train_natoms, cf, train_nls, train_wks)
        if structures_val === structures
            rmse_val = rmse_train
        else
            rmse_val = rmse_per_atom_with_embedding(itf, structures_val, y_val, val_natoms, cf, val_nls, val_wks)
        end
        if show_progress || (log_file != "")
            tnow = time()
            elapsed = tnow - time_start
            loop = tnow - last_time
            last_time = tnow
            logline = @sprintf "Iter: %d %3.3f %3.3f RMSE Train %10.5f eV | Val %10.5f eV\n" iter_count loop elapsed rmse_train rmse_val
            show_progress && print(logline)
            if log_file != ""
                open(log_file, "a") do file
                    write(file, logline)
                end
            end
        end

        flush(stdout)
        push!(rec, (rmse_train, rmse_val))

        # Checkpoint
        if checkpoint_every > 0 && checkpoint_prefix != "" && iter_count % checkpoint_every == 0
            ckpt_path = _ensure_dir(@sprintf("%s_%04d.json", checkpoint_prefix, iter_count))
            save_model_artifact(ckpt_path, itf; feature_spec=cf, elemental_energies, metadata=Dict("epoch" => iter_count))
        end

        iter_count += 1

        rmse_val, paramvector(itf)
    end

    # Setup closures for training data
    f!, j!, fj! = EDDPotentialsCore.setup_fj_with_embedding(
        itf, structures, cf, y, weights; recompute_xt
    )

    # For validation RMSE, use the training closures (they share model + cf state)
    od2 = OnceDifferentiable(f!, j!, fj!, p0, zeros(eltype(p0), length(y)), inplace=true)

    callback = show_progress || (earlystop > 0) || checkpoint_every > 0 ? progress_tracker : nothing

    opt_res = @timeit EDDPotentialsCore.to "lm solve with embedding" levenberg_marquardt(
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

    # Update model and embedding from final parameters
    final_params = keep_best ? opt_res.minimizer : paramvector(itf)
    # Also update from the final paramvector
    final_nn = view(final_params, 1:n_nn)
    final_emb = view(final_params, n_nn+1:n_total)
    EDDPotentialsCore.setparamvector!(itf, final_nn)
    EDDPotentialsCore._emb_param_set!(cf, final_emb)

    if normalize_embedding
        EDDPotentialsCore._emb_normalize!(cf)
    end

    opt_res, vcat(paramvector(itf), EDDPotentialsCore._emb_param_vec(cf)), [map(x -> x[1], rec) map(x -> x[2], rec)], (f!, j!, fj!)
end

"""
    train_lm_batched_with_embedding!(itf, structures, cf, y; kwargs...) ->
        (opt_res, params, history, funcs)

Train a model using batched LM with learnable species embedding.
"""
function train_lm_batched_with_embedding!(
    itf::AbstractNNInterface,
    structures::AbstractVector{<:Cell},
    cf::Union{RandomEmbeddedCellFeature,MultiRandomEmbeddedCellFeature},
    y::AbstractVector;
    p0_nn=paramvector(itf),
    structures_val=structures,
    y_val=y,
    normalize_embedding::Bool=false,
    recompute_xt::Bool=false,
    max_epochs=300,
    maxIter::Union{Integer,Nothing}=nothing,
    show_progress=false,
    earlystop=50,
    keep_best=true,
    log_file="",
    p=1.25,
    weights=nothing,
    batch_size=512,
    shuffle_seed=42,
    damping_mode::Symbol=:fletcher,
    attempts_per_step::Int=1,
    solver::Symbol=:direct,
    checkpoint_every::Int=0,
    checkpoint_prefix::AbstractString="",
    elemental_energies::Dict=Dict{Symbol,Float64}(),
    args...,
)
    if maxIter !== nothing
        @warn "maxIter is deprecated, use max_epochs instead" maxIter max_epochs
        max_epochs = maxIter
    end
    args = Base.structdiff(NamedTuple(args), NamedTuple{(:maxIter,)})

    rec = []
    n_nn = length(p0_nn)
    n_emb = EDDPotentialsCore._emb_param_count(cf)
    n_total = n_nn + n_emb

    p0 = vcat(p0_nn, EDDPotentialsCore._emb_param_vec(cf))

    train_natoms = [natoms(s) for s in structures]
    val_natoms = [natoms(s) for s in structures_val]

    # Pre-build NLs/workspaces for RMSE callback
    train_nls, train_wks = _precompute_nl_workspace(structures, cf)
    if structures_val !== structures
        val_nls, val_wks = _precompute_nl_workspace(structures_val, cf)
    else
        val_nls, val_wks = train_nls, train_wks
    end

    time_start = time()
    last_time = time()
    epoch_count = 1

    function progress_tracker()
        rmse_train = rmse_per_atom_with_embedding(itf, structures, y, train_natoms, cf, train_nls, train_wks)
        if structures_val === structures
            rmse_val = rmse_train
        else
            rmse_val = rmse_per_atom_with_embedding(itf, structures_val, y_val, val_natoms, cf, val_nls, val_wks)
        end
        if show_progress || (log_file != "")
            tnow = time()
            elapsed = tnow - time_start
            loop = tnow - last_time
            last_time = tnow
            logline = @sprintf "Epoch: %d %3.3f %3.3f RMSE Train %10.5f eV | Val %10.5f eV (batch=%d)\n" epoch_count loop elapsed rmse_train rmse_val batch_size
            show_progress && print(logline)
            if log_file != ""
                open(log_file, "a") do file
                    write(file, logline)
                end
            end
        end

        flush(stdout)
        push!(rec, (rmse_train, rmse_val))

        # Checkpoint
        if checkpoint_every > 0 && checkpoint_prefix != "" && epoch_count % checkpoint_every == 0
            ckpt_path = _ensure_dir(@sprintf("%s_%04d.json", checkpoint_prefix, epoch_count))
            save_model_artifact(ckpt_path, itf; feature_spec=cf, elemental_energies, metadata=Dict("epoch" => epoch_count))
        end

        epoch_count += 1

        rmse_val, vcat(paramvector(itf), EDDPotentialsCore._emb_param_vec(cf))
    end

    # Set up batched closures
    fj_batch!, f_full! = EDDPotentialsCore.setup_fj_batched_with_embedding(
        itf, structures, cf, y, weights; recompute_xt
    )

    callback = show_progress || (earlystop > 0) || checkpoint_every > 0 ? progress_tracker : nothing

    opt_res = @timeit EDDPotentialsCore.to "batched lm solve with embedding" levenberg_marquardt_batched(
        fj_batch!,
        f_full!,
        p0;
        n_data=length(structures),
        batch_size=batch_size,
        max_epochs=max_epochs,
        callback=callback,
        p=p,
        keep_best=keep_best,
        earlystop=earlystop,
        damping_mode=damping_mode,
        attempts_per_step=attempts_per_step,
        solver=solver,
        shuffle_seed=shuffle_seed,
        args...,
    )

    # Update model and embedding from final parameters
    final_params = keep_best ? opt_res.minimizer : p0
    final_nn = view(final_params, 1:n_nn)
    final_emb = view(final_params, n_nn+1:n_total)
    EDDPotentialsCore.setparamvector!(itf, final_nn)
    EDDPotentialsCore._emb_param_set!(cf, final_emb)

    if normalize_embedding
        EDDPotentialsCore._emb_normalize!(cf)
    end

    opt_res, vcat(paramvector(itf), EDDPotentialsCore._emb_param_vec(cf)), [map(x -> x[1], rec) map(x -> x[2], rec)], (fj_batch!, f_full!)
end

"""
    rmse_per_atom_with_embedding(itf, structures, y, natoms_vec, cf)
    rmse_per_atom_with_embedding(itf, structures, y, natoms_vec, cf, nls, workspaces)

Compute RMSE per atom using the current embedding to recompute features.
When `nls` and `workspaces` are provided, they are reused instead of rebuilt.
"""
function rmse_per_atom_with_embedding(itf, structures, y, natoms_vec, cf, nls, workspaces)
    total_sq = 0.0
    total_nat = 0
    for i in eachindex(structures)
        @timeit to "rmse: feature recompute" recompute_features!(workspaces[i].fvec, cf, structures[i], workspaces[i], nls[i])
        @timeit to "rmse: forward!" out = EDDPotentialsCore.forward!(itf, workspaces[i].fvec)
        pred = sum(out)
        total_sq += (pred - y[i])^2
        total_nat += natoms_vec[i]
    end
    sqrt(total_sq / total_nat)
end

function rmse_per_atom_with_embedding(itf, structures, y, natoms_vec, cf)
    rcut = suggest_rcut(cf; shell=0.0)
    nls = [CellBase.NeighbourList(s, rcut, 500; savevec=true) for s in structures]
    workspaces = [EDDPotentialsCore.get_workspace(cf, nl, false) for nl in nls]
    rmse_per_atom_with_embedding(itf, structures, y, natoms_vec, cf, nls, workspaces)
end

"""Pre-build neighbour lists and workspaces for a set of structures."""
function _precompute_nl_workspace(structures, cf; nmax=500)
    rcut = suggest_rcut(cf; shell=0.0)
    nls = [CellBase.NeighbourList(s, rcut, nmax; savevec=true) for s in structures]
    workspaces = [EDDPotentialsCore.get_workspace(cf, nl, false) for nl in nls]
    nls, workspaces
end
