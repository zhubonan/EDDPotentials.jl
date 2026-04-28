#=
Stochastic Gradient Descent training for EDDP neural networks.

Uses Flux.destructure/rebuild with Zygote.gradient on a flat parameter vector,
and implements Adam and SGD-with-momentum update rules directly on that vector.

This follows the same pattern as `generate_f_g_optim` in training.jl.
=#

using Printf
using Random
using Flux
using Zygote: withgradient
using .EDDPotentialsCore:
    AbstractNNInterface,
    get_flux_model,
    transformed_inp,
    paramvector,
    setparamvector!,
    nparams,
    forward!,
    ThreadBufferPool,
    rmse_per_atom_threaded
using .EDDPotentialsCore: to
using .EDDPotentialsCore: save_model_artifact
using Base.Threads: nthreads
using TimerOutputs

# ---------- Optimizer state types ----------

mutable struct SGDState{T<:AbstractFloat}
    velocity::Vector{T}
    lr::T
    momentum::T
end

mutable struct AdamState{T<:AbstractFloat}
    m::Vector{T}   # first moment
    v::Vector{T}   # second moment
    t::Int               # timestep
    lr::T
    beta1::T
    beta2::T
    eps::T
end

function SGDState(n::Int; lr=1e-2, momentum=0.9)
    SGDState(zeros(n), Float64(lr), Float64(momentum))
end

function SGDState{T}(n::Int; lr=1e-2, momentum=0.9) where {T<:AbstractFloat}
    SGDState{T}(zeros(T, n), T(lr), T(momentum))
end

function AdamState(n::Int; lr=1e-3, beta=(0.9, 0.999), eps=1e-8)
    AdamState(zeros(n), zeros(n), 0, Float64(lr), Float64(beta[1]), Float64(beta[2]), Float64(eps))
end

function AdamState{T}(n::Int; lr=1e-3, beta=(0.9, 0.999), eps=1e-8) where {T<:AbstractFloat}
    AdamState{T}(zeros(T, n), zeros(T, n), 0, T(lr), T(beta[1]), T(beta[2]), T(eps))
end

# ---------- Update rules ----------

function sgd_update!(state::SGDState, params, grad)
    @. state.velocity = state.momentum * state.velocity + grad
    @. params -= state.lr * state.velocity
    params
end

function adam_update!(state::AdamState, params, grad)
    state.t += 1
    @. state.m = state.beta1 * state.m + (1 - state.beta1) * grad
    @. state.v = state.beta2 * state.v + (1 - state.beta2) * grad^2
    bc1 = 1 - state.beta1^state.t
    bc2 = 1 - state.beta2^state.t
    m_hat = state.m ./ bc1
    v_hat = state.v ./ bc2
    @. params -= state.lr * m_hat / (sqrt(v_hat) + state.eps)
    params
end

# ---------- Learning rate schedule closures ----------

"""
    constant_lr(lr)

Return a constant learning rate schedule: `(epoch) -> lr`.
"""
constant_lr(lr) = (_ -> lr)

"""
    step_decay_lr(initial_lr; step_size=100, factor=0.5)

Return a step-decay schedule that multiplies the learning rate by `factor`
every `step_size` epochs.
"""
function step_decay_lr(initial_lr; step_size=100, factor=0.5)
    epoch -> initial_lr * factor^(epoch ÷ step_size)
end

"""
    cosine_annealing_lr(initial_lr, min_lr, max_epochs)

Return a cosine annealing schedule from `initial_lr` to `min_lr` over `max_epochs`.
"""
function cosine_annealing_lr(initial_lr, min_lr, max_epochs)
    epoch -> min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * epoch / max_epochs))
end

# ---------- Internal helpers ----------

"""
    _transform_targets(y, natoms_vec, yt)

Transform target total energies to the same space as the model output.
When `yt` is a ZScoreTransform, applies: `y_std = (y/nat - mean) / scale * nat`.
When `yt` is `nothing`, returns `y` unchanged.
"""
function _transform_targets(y, natoms_vec, yt)
    if yt === nothing
        return y
    end
    map(y, natoms_vec) do yi, nat
        ((yi / nat - yt.mean[1]) / yt.scale[1]) * nat
    end
end

"""
    _scatter_matrix(batch_indices, natoms_vec, ref)

Build a scatter matrix `S` of shape `(length(batch_indices), total_atoms)` where
`S[k, j] = 1` if atom `j` belongs to structure `batch_indices[k]`. The matrix is
allocated on the same device as `ref` (CPU or GPU).
"""
function _scatter_matrix(batch_indices, natoms_vec, ref)
    n_batch = length(batch_indices)
    total_atoms = sum(natoms_vec[i] for i in batch_indices)
    S = similar(ref, n_batch, total_atoms)
    fill!(S, zero(eltype(ref)))
    offset = 1
    for (k, i) in enumerate(batch_indices)
        nat = natoms_vec[i]
        S[k, offset:offset+nat-1] .= one(eltype(ref))
        offset += nat
    end
    S
end

"""
    _hcat_batch(x_t, batch_indices)

Concatenate feature matrices for structures in `batch_indices` into a single
`(nfeatures, total_atoms)` matrix. Allocated on the same device as the input arrays.
"""
function _hcat_batch(x_t, batch_indices)
    total_atoms = sum(size(x_t[i], 2) for i in batch_indices)
    nfeat = size(first(x_t), 1)
    X = similar(first(x_t), nfeat, total_atoms)
    fill!(X, 0)
    offset = 1
    for i in batch_indices
        nat = size(x_t[i], 2)
        X[:, offset:offset+nat-1] = x_t[i]
        offset += nat
    end
    X
end

"""
    _batch_loss_and_grad!(grad_out, flat, rebuild, x_t, y, batch_indices, natoms_vec, weights, pow)

Compute the average loss and gradient over `batch_indices` (a mini-batch).
All atoms in the batch are concatenated into a single matrix for one forward pass.
Per-structure energies are computed via a scatter matrix: `E_pred = S * vec(Y)`.
Works with any `AbstractArray` type (CPU or GPU).
"""
function _batch_loss_and_grad!(grad_out, flat, rebuild, x_t, y, batch_indices, natoms_vec, weights, pow)
    _batch_loss_and_grad_impl!(grad_out, flat, rebuild, x_t, y, batch_indices, natoms_vec, weights, pow, nothing)
end

function _batch_loss_and_grad_store!(grad_out, flat, rebuild, store, y, batch_indices, natoms_vec, weights, pow, itf)
    _batch_loss_and_grad_impl!(grad_out, flat, rebuild, store, y, batch_indices, natoms_vec, weights, pow, itf)
end

function _batch_loss_and_grad_impl!(grad_out, flat, rebuild, x_source, y, batch_indices, natoms_vec, weights, pow, itf)
    T = eltype(flat)
    fill!(grad_out, zero(T))

    # Build the concatenated batch matrix, applying Z-score transform on-the-fly if needed
    X_batch = if itf !== nothing
        # Disk-backed store path: transform each matrix as we load it
        nfeat = size(x_source[batch_indices[1]], 1)
        total_atoms = sum(natoms_vec[i] for i in batch_indices)
        X = similar(flat, nfeat, total_atoms)
        fill!(X, zero(T))
        offset = 1
        for i in batch_indices
            xi = transformed_inp(itf, x_source[i])
            nat = size(xi, 2)
            X[:, offset:offset+nat-1] = xi
            offset += nat
        end
        X
    else
        _hcat_batch(x_source, batch_indices)
    end

    # Scatter matrix for per-structure energy summation (same device as flat)
    S = _scatter_matrix(batch_indices, natoms_vec, flat)

    # Per-structure targets on the same device as flat
    y_batch = similar(flat, length(batch_indices))
    for (k, i) in enumerate(batch_indices)
        y_batch[k] = y[i]
    end
    w_batch = if weights !== nothing
        wb = similar(flat, length(batch_indices))
        for (k, i) in enumerate(batch_indices)
            wb[k] = weights[i]
        end
        wb
    else
        nothing
    end

    res = Zygote.withgradient(flat) do p
        m = rebuild(p)
        Y = m(X_batch)                    # (1, total_atoms) — per-atom energies
        E_pred = S * vec(Y)               # (n_batch,) — per-structure energies via matmul
        residuals = E_pred .- y_batch
        r = w_batch !== nothing ? residuals .* w_batch : residuals
        mean(abs.(r) .^ pow)
    end

    grad_out .= res.grad[1]
    return Float64(res.val)
end

"""
    _make_batches(n_data, batch_size, perm)

Split the permuted indices `perm` into batches of at most `batch_size`.
Returns a vector of index vectors.
"""
function _make_batches(n_data, batch_size, perm)
    batches = Vector{Vector{Int}}()
    for start in 1:batch_size:n_data
        stop = min(start + batch_size - 1, n_data)
        push!(batches, collect(perm[start:stop]))
    end
    batches
end

# ---------- Main training function ----------

"""
    train_sgd!(itf, x, y; kwargs...) -> (nothing, params, rmse_history, nothing)

Train a neural network using stochastic gradient descent (SGD) or Adam optimizer.

Uses `Flux.destructure`/`rebuild` with `Zygote.gradient` on a flat parameter vector.
Optimizer update rules (Adam, SGD with momentum) are implemented directly on the
flat vector, independent of Flux's internal optimizer API.

The return value has the same shape as `train_lm_batched!` for compatibility.

# Arguments
- `itf::AbstractNNInterface`: Neural network interface (must support `get_flux_model`)
- `x`: Feature matrices (Vector{Matrix{Float64}}), one per structure
- `y`: Target total energies (Vector{Float64})

# Keyword Arguments
- `optimizer=:adam`: Optimizer to use (`:adam`, `:sgd`, or `:sgd_momentum`)
- `lr=1e-3`: Initial learning rate
- `lr_schedule=nothing`: A closure `(epoch) -> lr`; defaults to `constant_lr(lr)`
- `momentum=0.9`: Momentum coefficient for `:sgd_momentum`
- `batch_size=32`: Mini-batch size (structures per gradient step)
- `max_epochs=300`: Maximum training epochs
- `earlystop=50`: Stop if validation RMSE hasn't improved for this many epochs
- `keep_best=true`: Restore best parameters at the end
- `show_progress=false`: Print epoch-by-epoch progress
- `log_file=""`: Append progress to this file
- `shuffle_seed=42`: RNG seed for reproducible shuffling
- `pow=2`: Loss power (2 = MSE, 1 = MAE)
- `weights=nothing`: Per-structure loss weights
- `x_val=x, y_val=y`: Validation data
- `beta=(0.9, 0.999)`: Adam β₁, β₂
- `eps=1e-8`: Adam ε
- `weight_decay=0.0`: L2 regularization coefficient
- `grad_clip=1.0`: Clip gradient L2 norm to this value (0 = disable clipping)
```julia
# Adam training (default)
train!(itf, fc_train, fc_val; train_method=:sgd, optimizer=:adam, lr=1e-3, max_epochs=500)

# SGD with momentum and cosine annealing
schedule = cosine_annealing_lr(1e-2, 1e-4, 300)
train!(itf, fc_train, fc_val; train_method=:sgd, optimizer=:sgd_momentum,
       lr=1e-2, lr_schedule=schedule, momentum=0.9)
```
"""
function train_sgd!(
    itf::AbstractNNInterface,
    x,
    y;
    x_val=x,
    y_val=y,
    optimizer::Symbol=:adam,
    lr::Real=1e-3,
    lr_schedule=nothing,
    momentum::Real=0.9,
    batch_size::Int=32,
    max_epochs::Int=300,
    earlystop::Int=50,
    keep_best::Bool=true,
    show_progress::Bool=false,
    log_file::AbstractString="",
    shuffle_seed::Int=42,
    pow::Int=2,
    weights=nothing,
    beta::Tuple{Real,Real}=(0.9, 0.999),
    eps::Real=1e-8,
    weight_decay::Real=0.0,
    grad_clip::Real=1.0,
    checkpoint_every::Int=0,
    checkpoint_prefix::AbstractString="",
    feature_spec=nothing,
    elemental_energies::Dict=Dict{Symbol,Float64}(),
    kwargs...,
)

    # --- GPU dispatch ---
    if EDDPotentials.USE_CUDA[]
        return EDDPotentials.train_sgd_gpu!(itf, x, y;
            x_val, y_val, optimizer, lr, lr_schedule, momentum,
            batch_size, max_epochs, earlystop, keep_best, show_progress,
            log_file, shuffle_seed, pow, weights, beta, eps,
            weight_decay, grad_clip, checkpoint_every, checkpoint_prefix,
            feature_spec, elemental_energies, kwargs...)
    end

    # --- Setup ---
    model = get_flux_model(itf)
    flat, rebuild = Flux.destructure(model)
    nparam = length(flat)

    # Atom counts for per-atom RMSE
    train_natoms = natoms(x)
    val_natoms = natoms(x_val)

    # Pre-transform inputs (apply Z-score normalization from itf.xt)
    # For disk-backed FeatureStores, defer transformation to per-batch on-the-fly
    # to avoid materializing the entire dataset.
    x_is_disk_store = x isa FeatureStore && x.data === nothing
    if x_is_disk_store
        x_t = x   # will be transformed per-batch
    else
        x_t = [transformed_inp(itf, xi) for xi in x]
    end
    x_val_is_disk_store = x_val isa FeatureStore && x_val.data === nothing
    if x_val === x
        x_val_t = x_t
    elseif x_val_is_disk_store
        x_val_t = x_val
    else
        x_val_t = [transformed_inp(itf, xi) for xi in x_val]
    end

    # Transform targets to match model output space.
    # When yt is set, the model outputs per-atom standardized energies.
    # sum(model(x)) gives standardized total energy, so we must also
    # transform y to standardized total energy for the loss to be correct.
    # transform_y: y_std = (y/nat - mean) / scale * nat
    y_train = _transform_targets(y, train_natoms, itf.yt)
    y_val_transformed = x_val === x ? y_train : _transform_targets(y_val, val_natoms, itf.yt)

    n_data = length(x)

    # Learning rate schedule
    actual_schedule = lr_schedule !== nothing ? lr_schedule : constant_lr(lr)

    # Create optimizer state
    if optimizer == :adam
        opt_state = AdamState(nparam; lr=Float64(lr), beta=beta, eps=Float64(eps))
    elseif optimizer == :sgd
        opt_state = SGDState(nparam; lr=Float64(lr), momentum=0.0)
    elseif optimizer == :sgd_momentum
        opt_state = SGDState(nparam; lr=Float64(lr), momentum=Float64(momentum))
    else
        error(
            "Unknown optimizer: $optimizer. Supported: :adam, :sgd, :sgd_momentum",
        )
    end

    # Gradient accumulation buffer
    grad_buf = zeros(nparam)

    # Tracking state
    ctx = TrainingContext(itf, feature_spec, elemental_energies, checkpoint_every, checkpoint_prefix, show_progress, log_file)
    best_flat = copy(flat)
    min_val_rmse = floatmax(Float64)
    min_epoch = 0

    rng = MersenneTwister(shuffle_seed)

    # Pre-allocate thread pool for validation RMSE (forward-only, no gradients).
    # pool.chain references itf.chain, so setparamvector!(itf, flat) keeps it in sync.
    val_pool = ThreadBufferPool(itf, nthreads() * 10 + 1)

    # --- Epoch loop ---
    for epoch in 1:max_epochs
        # Update learning rate
        current_lr = Float64(actual_schedule(epoch))
        if optimizer == :adam
            opt_state.lr = current_lr
        else
            opt_state.lr = current_lr
        end

        # Shuffle data
        perm = randperm(rng, n_data)
        batches = _make_batches(n_data, batch_size, perm)

        # Mini-batch updates
        for batch_indices in batches
            @timeit to "sgd: grad" begin
                if x_is_disk_store
                    loss_val = _batch_loss_and_grad_store!(
                        grad_buf, flat, rebuild, x_t, y_train, batch_indices, train_natoms, weights, pow, itf
                    )
                else
                    loss_val = _batch_loss_and_grad!(
                        grad_buf, flat, rebuild, x_t, y_train, batch_indices, train_natoms, weights, pow
                    )
                end
            end

            # L2 regularization (weight decay)
            if weight_decay > 0.0
                @. grad_buf += weight_decay * flat
            end

            # Gradient clipping by global L2 norm
            if grad_clip > 0.0
                gnorm = sqrt(sum(abs2, grad_buf))
                if gnorm > grad_clip
                    grad_buf .*= grad_clip / gnorm
                end
            end

            # Skip update if gradient is not finite
            if !all(isfinite, grad_buf)
                @warn "Non-finite gradient at epoch $epoch, skipping update"
                continue
            end

            # Apply optimizer update
            @timeit to "sgd: update" if optimizer == :adam
                adam_update!(opt_state, flat, grad_buf)
            else
                sgd_update!(opt_state, flat, grad_buf)
            end

            # Sync the underlying model so interface methods work correctly
            model = rebuild(flat)
        end

        # Rebuild model with final parameters for this epoch
        model = rebuild(flat)

        # --- Epoch callback: evaluate RMSE, log, early stopping ---
        @timeit to "sgd: eval" begin
            # Sync interface parameters for RMSE evaluation
            setparamvector!(itf, flat)

            rmse_train = rmse_per_atom_threaded(val_pool, x, y, train_natoms)
            if x_val === x
                rmse_val = rmse_train
            else
                rmse_val = rmse_per_atom_threaded(val_pool, x_val, y_val, val_natoms)
            end
        end

        ctx.counter = epoch
        progress_callback!(ctx, rmse_train, rmse_val,
            @sprintf(" (%s, bs=%d, lr=%.2e)", optimizer, batch_size, current_lr))

        # Track best
        if rmse_val < min_val_rmse
            min_val_rmse = rmse_val
            min_epoch = epoch
            best_flat = copy(flat)
        end

        # Early stopping
        if earlystop > 0 && (epoch - min_epoch) >= earlystop
            if show_progress
                @info "Early stopping at epoch $epoch (no improvement for $earlystop epochs)"
            end
            break
        end
    end

    # Restore best parameters
    if keep_best
        flat .= best_flat
        model = rebuild(flat)
        setparamvector!(itf, flat)
    end

    return (nothing, paramvector(itf), [map(x -> x[1], ctx.rec) map(x -> x[2], ctx.rec)], nothing)
end

"""
    _rmse_per_atom_raw(model, x_t, y, natoms, itf)

Compute per-atom RMSE using the raw model (Chain) and pre-transformed inputs.
Handles the `yt` output transform from the interface.
"""
function _rmse_per_atom_raw(model, x_t, y, natoms_vec, itf)
    n = length(y)
    preds = Vector{Float64}(undef, n)
    for i in 1:n
        out = model(x_t[i])
        # Apply yt inverse transform if present
        if itf.yt !== nothing
            # The model outputs in standardized space, reconstruct to physical space
            out_phys = StatsBase.reconstruct(itf.yt, out)
            preds[i] = sum(out_phys)
        else
            preds[i] = sum(out)
        end
    end
    sqrt(mean(((preds .- y) ./ natoms_vec) .^ 2))
end
