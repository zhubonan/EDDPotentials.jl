#=
Batched Levenberg-Marquardt solver for stochastic/mini-batch training.

Operates on random subsets (batches) of the training data per LM step,
organised in epochs.  Each epoch shuffles the data and runs one LM step per
batch.  The callback (RMSE evaluation, early stopping) is called once per
epoch on the full dataset.

The normal equation (J'J + λ·diag(DtD))·Δx = -J'·f is computed from the
current batch only, reducing per-iteration cost from O(m·n²) to O(B·n²) where
B is the batch size and n the number of parameters.
=#

using LinearAlgebra
using Random
using OptimBase
using Logging
import StatsBase
import NLSolversBase: value, jacobian

using .EDDPotentialsCore: to
using TimerOutputs

struct BatchedLevenbergMarquardt <: Optimizer end
Base.summary(::BatchedLevenbergMarquardt) = "Batched Levenberg-Marquardt"

"""
    levenberg_marquardt_batched(fj_batch!, f_full!, initial_x; kwargs...)

Batched variant of the Levenberg-Marquardt algorithm for large datasets.

Instead of computing the full Jacobian over all data points, each LM step
operates on a random mini-batch.  Data is shuffled at the start of each
epoch.  The callback (if provided) is called once per epoch and should
evaluate on the **full** dataset for reliable early stopping.

# Arguments
- `fj_batch!`:  Function `(fvec, jmat, param, batch_indices) -> (fvec, jmat)`.
  Computes the residual vector and Jacobian for the data points indexed by
  `batch_indices`.  Either `fvec` or `jmat` may be `nothing`.
- `f_full!`:    Function `(fvec, param) -> fvec`.  Computes the residual
  vector for the **full** dataset (used by the callback).
- `initial_x`:  Initial parameter vector.

# Keyword arguments
- `n_data`:          Total number of data points.
- `batch_size`:      Number of samples per mini-batch (default 512).
- `maxIter`:         Maximum total LM iterations **across all batches and epochs** (default 1000).
- `p`:               Power for iterative weight update (default 2.0).
- `update_weights`:  Whether to update weights (default `true`).
- `x_tol`:           Step-size convergence tolerance (default `1e-8`).
- `g_tol`:           Gradient convergence tolerance (default `1e-12`).
- `lambda`:          Initial damping parameter (default `10`).
- `tau`:             Heuristic for initial lambda from J'J (default `Inf`).
- `lambda_increase`: Factor to increase lambda on bad steps (default `2.0`).
- `lambda_decrease`: Factor to decrease lambda on good steps (default `0.2`).
- `min_step_quality`:  Threshold below which trust region shrinks (default `1e-3`).
- `good_step_quality`: Threshold above which trust region expands (default `0.75`).
- `show_trace`:      Print status each epoch (default `false`).
- `callback`:        Called once per epoch, should return `(rmse_test, param_vector)`.
- `keep_best`:       Rewind to best parameter vector at end (default `true`).
- `earlystop`:       Stop if callback RMSE hasn't improved for this many epochs (default `0`).
- `shuffle_seed`:    RNG seed for reproducibility (default `42`).
"""
function levenberg_marquardt_batched(
    fj_batch!,
    f_full!,
    initial_x::AbstractVector{T};
    n_data::Int,
    batch_size::Int=512,
    maxIter::Integer=1000,
    p=2.0,
    update_weights=true,
    x_tol::Real=1e-8,
    g_tol::Real=1e-12,
    lambda=T(10),
    tau=T(Inf),
    lambda_increase::Real=2.0,
    lambda_decrease::Real=0.2,
    min_step_quality::Real=1e-3,
    good_step_quality::Real=0.75,
    show_trace::Bool=false,
    callback=nothing,
    keep_best=true,
    earlystop=0,
    shuffle_seed::Int=42,
) where {T}

    # --- Validation ---
    batch_size > 0 || throw(ArgumentError("batch_size must be positive, got $batch_size"))

    # --- Constants (same as full LM) ---
    MAX_LAMBDA = 1e16
    MIN_LAMBDA = 1e-16
    MIN_DIAGONAL = 1e-6
    WEIGHT_SHIFT = 1e-4

    # --- State ---
    n = length(initial_x)
    x = copy(initial_x)
    best_x = copy(initial_x)
    delta_x = similar(x)
    x_trial = similar(x)
    converged = false
    x_converged = false
    g_converged = false
    iterCt = 0
    epoch = 0

    rng = MersenneTwister(shuffle_seed)

    # Buffers – sized for a single batch (allocated once, reused)
    B_max = min(batch_size, n_data)
    J_b = Matrix{T}(undef, B_max, n)
    f_b = Vector{T}(undef, B_max)
    wt_b = ones(T, B_max)

    # Normal-equation buffers (n × n, independent of batch size)
    JJ = Matrix{T}(undef, n, n)
    J2 = similar(J_b)
    n_buffer = Vector{T}(undef, n)
    DtD = zeros(T, n)
    trial_f_b = Vector{T}(undef, B_max)

    # Full-dataset residual (for tau initialisation and callback)
    f_full = Vector{T}(undef, n_data)
    f_full!(f_full, initial_x)
    best_residual = residual_full = sum(abs2, f_full)

    test_rmse = T[]

    # Tau heuristic: scale initial lambda by max diagonal of J'J
    # Use first batch to estimate
    first_batch = collect(1:min(B_max, n_data))
    fj_batch!(f_b, J_b, initial_x, first_batch)
    B0 = length(first_batch)
    if isfinite(tau)
        lambda = tau * maximum(sum(abs2, @view(J_b[1:B0, :]); dims=1))
    end

    # Trace
    tr = OptimBase.OptimizationTrace{BatchedLevenbergMarquardt}()

    @debug "Batched LM starting" n_data batch_size nparams=n maxIter lambda

    while ~converged && iterCt < maxIter
        epoch += 1
        x_converged = false
        g_converged = false
        perm = randperm(rng, n_data)
        num_batches = cld(n_data, batch_size)

        @debug "Epoch $epoch" num_batches lambda

        for bi in 1:num_batches
            iterCt >= maxIter && break

            batch_start = (bi - 1) * batch_size + 1
            batch_end = min(bi * batch_size, n_data)
            batch_idx = perm[batch_start:batch_end]
            B = length(batch_idx)

            # ---- Evaluate batch Jacobian and residual ----
            @timeit to "batch fj" fj_batch!(
                (B <= B_max) ? view(f_b, 1:B) : f_b,
                (B <= B_max) ? view(J_b, 1:B, :) : J_b,
                x,
                batch_idx,
            )
            fv = (B <= B_max) ? view(f_b, 1:B) : f_b
            Jv = (B <= B_max) ? view(J_b, 1:B, :) : J_b
            wv = (B <= B_max) ? view(wt_b, 1:B) : wt_b

            # Initialise batch weights from residuals
            if p != 2.0 && update_weights
                for i in 1:B
                    wv[i] = abs(fv[i] + WEIGHT_SHIFT)^(p - 2.0)
                end
            else
                fill!(wv, one(T))
            end

            residual_b = sum(i -> wv[i] * fv[i]^2, 1:B)

            # ---- Normal equations ----
            # DtD = diag(J' W J)
            @timeit to "ATWA_DIAG!" _atwa_diag_batched!(DtD, Jv, wv)
            wt_mean = mean(wv)
            for i in 1:n
                if DtD[i] <= MIN_DIAGONAL * wt_mean
                    DtD[i] = MIN_DIAGONAL * wt_mean
                end
            end

            # J'WJ via sqrt(W).*J trick
            @timeit to "JJ" begin
                J2v = (B <= B_max) ? view(J2, 1:B, :) : J2
                J2v .= sqrt.(wv) .* Jv
                mul!(JJ, transpose(J2v), J2v)
            end

            # Add damping diagonal
            for i in 1:n
                @inbounds JJ[i, i] += lambda * DtD[i]
            end

            # Right-hand side: -J' W f
            mul!(n_buffer, transpose(Jv), wv .* fv)
            rmul!(n_buffer, -one(T))

            # Solve
            @timeit to "matinv" delta_x .= JJ \ n_buffer

            # ---- Predicted residual ----
            Jdelta = Jv * delta_x .+ fv
            predicted_residual = dot(Jdelta, wv .* Jdelta)

            # ---- Trial step ----
            x_trial .= x .+ delta_x

            # Evaluate trial on same batch
            trial_fv = (B <= B_max) ? view(trial_f_b, 1:B) : trial_f_b
            @timeit to "trial eval" fj_batch!(trial_fv, nothing, x_trial, batch_idx)
            trial_residual = sum(i -> wv[i] * trial_fv[i]^2, 1:B)

            # Step quality
            denom = predicted_residual - residual_b
            rho = abs(denom) < eps(T) ? zero(T) : (trial_residual - residual_b) / denom

            @debug "  Batch $bi" B lambda rho residual_b trial_residual

            if trial_residual < residual_b && rho > min_step_quality
                x .= x_trial

                if rho > good_step_quality
                    lambda = max(lambda_decrease * lambda, MIN_LAMBDA)
                    @debug "  Lambda decreased: $lambda"
                end
            else
                lambda = min(lambda_increase * lambda, MAX_LAMBDA)
                @debug "  Step rejected, lambda increased: $lambda"
            end

            iterCt += 1

            # Check convergence (batch gradient)
            g_norm = norm(transpose(Jv) * (wv .* fv), Inf)
            if norm(delta_x) < x_tol * (x_tol + norm(x))
                x_converged = true
            end
            if g_norm < g_tol
                g_converged = true
            end

            @debug "  Convergence" x_converged g_converged g_norm iterCt

            converged = x_converged | g_converged
            converged && break
        end # batch loop

        # ---- Per-epoch callback ----
        # Always evaluate full residual for tracking
        @timeit to "full eval" f_full!(f_full, x)
        residual_full = sum(abs2, f_full)
        if residual_full < best_residual && isnothing(callback)
            best_residual = residual_full
            best_x .= x
            @debug "Epoch $epoch: new best residual: $best_residual"
        end

        if !isnothing(callback)
            @timeit to "callback eval" rmse_val, xtmp = callback()
            if length(test_rmse) > 0 && rmse_val < minimum(test_rmse)
                best_x .= xtmp
            end
            push!(test_rmse, rmse_val)
            @info "Epoch $epoch complete" test_rmse=rmse_val lambda iterCt
        end

        if earlystop > 0 && !isempty(test_rmse)
            earlystopcheck(test_rmse, earlystop) && (converged = true)
        end

        # Show trace
        if show_trace
            g_norm = isnan(residual_full) ? NaN : sqrt(residual_full / n_data)
            d = Dict("lambda" => lambda, "epoch" => epoch)
            os = OptimBase.OptimizationState{BatchedLevenbergMarquardt}(
                iterCt, residual_full, g_norm, d
            )
            push!(tr, os)
            println(os)
        end
    end # main loop

    # Rewind to best x
    if keep_best
        f_full!(f_full, best_x)
        x .= best_x
    end

    MultivariateOptimizationResults(
        BatchedLevenbergMarquardt(),
        initial_x,
        x,
        sum(abs2, f_full),
        iterCt,
        !converged,
        x_converged,
        0.0,
        0.0,
        false,
        0.0,
        0.0,
        g_converged,
        g_tol,
        0.0,
        false,
        tr,
        0,    # f_calls
        iterCt, # g_calls (approx: one per batch step)
        0,
    )
end

"""
    _atwa_diag_batched!(out, A, W)

Compute diag(A' W A) for a batch Jacobian and weight vector.
"""
function _atwa_diag_batched!(out, A, W)
    m, n = size(A)
    fill!(out, 0)
    for j in 1:n
        @simd for i in 1:m
            @inbounds out[j] += A[i, j] * A[i, j] * W[i]
        end
    end
    out
end
