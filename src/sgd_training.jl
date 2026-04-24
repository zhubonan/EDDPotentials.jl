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
    forward!
using .EDDPotentialsCore: to
using .EDDPotentialsCore: save_model_artifact
using TimerOutputs

"""Create parent directories for `path` if they don't exist."""
function _ensure_dir(path::AbstractString)
    dir = dirname(path)
    !isempty(dir) && !isdir(dir) && mkpath(dir)
    path
end

# ---------- Optimizer state types ----------

mutable struct SGDState
    velocity::Vector{Float64}
    lr::Float64
    momentum::Float64
end

mutable struct AdamState
    m::Vector{Float64}   # first moment
    v::Vector{Float64}   # second moment
    t::Int               # timestep
    lr::Float64
    beta1::Float64
    beta2::Float64
    eps::Float64
end

function SGDState(n::Int; lr=1e-2, momentum=0.9)
    SGDState(zeros(n), lr, momentum)
end

function AdamState(n::Int; lr=1e-3, beta=(0.9, 0.999), eps=1e-8)
    AdamState(zeros(n), zeros(n), 0, lr, beta[1], beta[2], eps)
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
    _batch_loss_and_grad!(grad_out, flat, rebuild, x_t, y, indices, weights, pow)

Compute the average loss and accumulated gradient over `indices` (a mini-batch).
`x_t` are pre-transformed feature matrices, `y` are target total energies
(in the same space as the model output).
"""
function _batch_loss_and_grad!(grad_out, flat, rebuild, x_t, y, indices, weights, pow)
    fill!(grad_out, 0.0)
    total_loss = 0.0

    for i in indices
        res = Zygote.withgradient(flat) do p
            m = rebuild(p)
            E_pred = sum(m(x_t[i]))
            residual = E_pred - y[i]
            if weights !== nothing
                residual *= weights[i]
            end
            abs(residual)^pow
        end
        grad_out .+= res.grad[1]
        total_loss += res.val
    end

    grad_out ./= length(indices)
    return total_loss / length(indices)
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
train!(itf, fc_train, fc_val; train_method="sgd", optimizer=:adam, lr=1e-3, max_epochs=500)

# SGD with momentum and cosine annealing
schedule = cosine_annealing_lr(1e-2, 1e-4, 300)
train!(itf, fc_train, fc_val; train_method="sgd", optimizer=:sgd_momentum,
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

    # --- Setup ---
    model = get_flux_model(itf)
    flat, rebuild = Flux.destructure(model)
    nparam = length(flat)

    # Pre-transform inputs (apply Z-score normalization from itf.xt)
    x_t = [transformed_inp(itf, xi) for xi in x]
    x_val_t = x_val === x ? x_t : [transformed_inp(itf, xi) for xi in x_val]

    # Atom counts for per-atom RMSE
    train_natoms = [size(v, 2) for v in x]
    val_natoms = [size(v, 2) for v in x_val]

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
    rec = []
    best_flat = copy(flat)
    min_val_rmse = floatmax(Float64)
    min_epoch = 0

    time_start = time()
    last_time = time()
    rng = MersenneTwister(shuffle_seed)

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
                loss_val = _batch_loss_and_grad!(
                    grad_buf, flat, rebuild, x_t, y_train, batch_indices, weights, pow
                )
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

            rmse_train = _rmse_per_atom_raw(model, x_t, y, train_natoms, itf)
            if x_val === x
                rmse_val = rmse_train
            else
                rmse_val = _rmse_per_atom_raw(model, x_val_t, y_val, val_natoms, itf)
            end
        end

        tnow = time()
        elapsed = tnow - time_start
        loop_time = tnow - last_time
        last_time = tnow

        logline = @sprintf(
            "Epoch: %d %3.3f %3.3f RMSE Train %10.5f eV | Val %10.5f eV (%s, bs=%d, lr=%.2e)\n",
            epoch, loop_time, elapsed, rmse_train, rmse_val, optimizer, batch_size, current_lr
        )
        show_progress && print(logline)
        if log_file != ""
            open(log_file, "a") do file
                write(file, logline)
            end
        end
        flush(stdout)

        push!(rec, (rmse_train, rmse_val))

        # Checkpoint
        if checkpoint_every > 0 && checkpoint_prefix != "" && feature_spec !== nothing && epoch % checkpoint_every == 0
            ckpt_path = _ensure_dir(@sprintf("%s_%04d.json", checkpoint_prefix, epoch))
            save_model_artifact(ckpt_path, itf; feature_spec, elemental_energies, metadata=Dict("epoch" => epoch))
        end

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

    return (nothing, paramvector(itf), [map(x -> x[1], rec) map(x -> x[2], rec)], nothing)
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
