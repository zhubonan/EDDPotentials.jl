#=
levenberg-Marquardt solver adapted for solving EDDPotentials potential

The source code is adapted from that of LsqFit to allow more features.
=#

using Base.Threads
using Distributions
using OptimBase
using LinearAlgebra
import NLSolversBase: value, jacobian
import StatsBase
import StatsBase: coef, dof, mean, nobs, rss, stderror, weights, residuals
import LsqFit

import Base.summary

struct LevenbergMarquardt <: Optimizer end
Base.summary(::LevenbergMarquardt) = "Levenberg-Marquardt"

"""
    _apply_damping_full!(JJ, lambda, DtD, mode)

Add LM damping to the full-batch normal matrix in-place.

# Modes
- `:fletcher`       — `JJ + λ·diag(DtD)` (current default)
- `:fletcher_floor` — `JJ + λ·diag(max(1, DtD))` (Fortran `forge` style)
"""
function _apply_damping_full!(JJ, lambda, DtD, mode::Symbol)
    if mode === :fletcher
        @simd for i in eachindex(DtD)
            @inbounds JJ[i, i] += lambda * DtD[i]
        end
    elseif mode === :fletcher_floor
        floor = one(eltype(DtD))
        @simd for i in eachindex(DtD)
            @inbounds JJ[i, i] += lambda * max(floor, DtD[i])
        end
    else
        throw(ArgumentError("Unknown damping_mode: $mode. Use :fletcher or :fletcher_floor."))
    end
    JJ
end

function _damping_diagonal_full!(damping_diag, DtD, mode::Symbol)
    if mode === :fletcher
        copyto!(damping_diag, DtD)
    elseif mode === :fletcher_floor
        floor = one(eltype(DtD))
        @simd for i in eachindex(DtD)
            @inbounds damping_diag[i] = max(floor, DtD[i])
        end
    else
        throw(ArgumentError("Unknown damping_mode: $mode. Use :fletcher or :fletcher_floor."))
    end
    damping_diag
end

Base.@kwdef mutable struct CGSolveStats
    converged::Bool = false
    iterations::Int = 0
    relres::Float64 = Inf
end

function _reset_cg_stats!(stats::Union{Nothing,CGSolveStats})
    isnothing(stats) && return nothing
    stats.converged = false
    stats.iterations = 0
    stats.relres = Inf
    return stats
end

function _finish_cg_stats!(
    stats::Union{Nothing,CGSolveStats},
    converged::Bool,
    iterations::Int,
    relres,
)
    isnothing(stats) && return converged
    stats.converged = converged
    stats.iterations = iterations
    stats.relres = Float64(relres)
    return converged
end

function _weighted_jtr_neg!(rhs, J, wt, rlike, tmp)
    @. tmp = wt * rlike
    mul!(rhs, transpose(J), tmp)
    rmul!(rhs, -1)
    return rhs
end

function _adaptive_cg_tol(
    cg_tol,
    cg_tol_max,
    grad_norm,
    initial_grad_norm,
    adaptive::Bool,
    ::Type{T},
) where {T}
    !adaptive && return T(cg_tol)
    denom = max(T(initial_grad_norm), eps(T))
    ratio = max(T(grad_norm) / denom, eps(T))
    return clamp(T(cg_tol_max) * sqrt(ratio), T(cg_tol), T(cg_tol_max))
end

function _build_cg_preconditioner!(
    M,
    DtD,
    lambda,
    damping_diag,
    preconditioner_floor,
)
    floor = eltype(M)(preconditioner_floor)
    @. M = max(DtD + lambda * damping_diag, floor)
    for value in M
        (isfinite(value) && value > zero(value)) || return false
    end
    return true
end

function _apply_cg_preconditioner!(z, r, M)
    @. z = r / M
    return z
end

"""
    _cg_solve_full!(delta_x, J, wt, lambda, DtD, rhs, tmp, Ap, r, z, p, M, damping_diag; kwargs...) -> Bool

Solve `(J'WJ + λ·diag(d)) * delta_x = rhs` with matrix-free CG.

Returns `true` only when CG converges. Returns `false` on invalid Jacobi
preconditioner entries, non-finite arithmetic, non-positive curvature, or
max-iteration non-convergence.
"""
function _cg_solve_full!(
    delta_x,
    J,
    wt,
    lambda,
    DtD,
    rhs,
    tmp,
    Ap,
    r,
    z,
    p,
    M,
    damping_diag;
    damping_mode::Symbol=:fletcher,
    maxiter::Int=500,
    tol::Real=1e-6,
    preconditioner_floor::Real=eps(eltype(delta_x)),
    stats::Union{Nothing,CGSolveStats}=nothing,
)
    _reset_cg_stats!(stats)
    maxiter > 0 || return _finish_cg_stats!(stats, false, 0, Inf)

    _damping_diagonal_full!(damping_diag, DtD, damping_mode)
    _build_cg_preconditioner!(
        M,
        DtD,
        lambda,
        damping_diag,
        preconditioner_floor,
    ) || return _finish_cg_stats!(stats, false, 0, Inf)

    fill!(delta_x, zero(eltype(delta_x)))
    copyto!(r, rhs)
    rhs_norm = norm(rhs)
    isfinite(rhs_norm) || return _finish_cg_stats!(stats, false, 0, Inf)
    rhs_norm == 0 && return _finish_cg_stats!(stats, true, 0, 0.0)

    _apply_cg_preconditioner!(z, r, M)
    copyto!(p, z)
    rho = dot(r, z)
    (isfinite(rho) && rho > zero(rho)) || return _finish_cg_stats!(stats, false, 0, Inf)

    for iter in 1:maxiter
        mul!(tmp, J, p)
        @. tmp = wt * tmp
        mul!(Ap, transpose(J), tmp)
        @. Ap += lambda * damping_diag * p

        pAp = dot(p, Ap)
        (isfinite(pAp) && pAp > zero(pAp)) ||
            return _finish_cg_stats!(stats, false, iter, Inf)

        alpha = rho / pAp
        isfinite(alpha) || return _finish_cg_stats!(stats, false, iter, Inf)
        @. delta_x += alpha * p
        @. r -= alpha * Ap

        r_norm = norm(r)
        isfinite(r_norm) || return _finish_cg_stats!(stats, false, iter, Inf)
        relres = r_norm / rhs_norm
        relres <= tol && return _finish_cg_stats!(stats, true, iter, relres)

        _apply_cg_preconditioner!(z, r, M)
        rho_next = dot(r, z)
        (isfinite(rho_next) && rho_next > zero(rho_next)) ||
            return _finish_cg_stats!(stats, false, iter, relres)

        beta = rho_next / rho
        isfinite(beta) || return _finish_cg_stats!(stats, false, iter, relres)
        @. p = z + beta * p
        rho = rho_next
    end

    relres = norm(r) / rhs_norm
    return _finish_cg_stats!(stats, false, maxiter, relres)
end

function _build_normal_full!(JJ, J2, J, wt, lambda, DtD, damping_mode::Symbol)
    @. J2 = sqrt(wt) * J
    mul!(JJ, transpose(J2), J2)
    return _apply_damping_full!(JJ, lambda, DtD, damping_mode)
end

function _factor_normal_full!(JJ, solver::Symbol)
    if solver === :direct
        return lu!(JJ), true
    elseif solver === :cholesky
        try
            return cholesky!(Hermitian(JJ)), true
        catch err
            err isa PosDefException || rethrow()
            return nothing, false
        end
    elseif solver === :qr
        return qr(JJ), true
    elseif solver === :cg
        throw(ArgumentError("solver=:cg is matrix-free in full LM; use _cg_solve_full! instead."))
    else
        throw(ArgumentError("Unknown solver: $solver. Use :direct, :cholesky, :cg, or :qr."))
    end
end

function _ldiv_normal_full!(out, factor, rhs)
    ldiv!(out, factor, rhs)
    return out
end

"""
    _solve_normal_full!(delta_x, JJ, rhs, solver) -> Bool

Solve the damped normal equations `JJ * delta_x = rhs` for the full LM solver.
Returns `true` when the solve succeeds. A failed `:cholesky` factorisation
returns `false` so the LM loop can increase `lambda` and retry.

# Solvers
- `:direct`    — LU decomposition via `\\`
- `:cholesky`  — strict Cholesky factorisation; failure rejects the LM step
- `:cg`        — matrix-free preconditioned CG (handled outside this dense helper)
- `:qr`        — QR decomposition (robust for ill-conditioned systems)
"""
function _solve_normal_full!(delta_x, JJ, rhs, solver::Symbol)
    if solver === :direct
        delta_x .= JJ \ rhs
        return true
    end
    factor, solve_ok = _factor_normal_full!(JJ, solver)
    solve_ok || return false
    _ldiv_normal_full!(delta_x, factor, rhs)
    return true
end
"""
    `levenberg_marquardt(f, g, initial_x; <keyword arguments>`
Returns the argmin over x of `sum(f(x).^2)` using the Levenberg-Marquardt
algorithm, and an estimate of the Jacobian of `f` at x.
The function `f` should take an input vector of length n and return an output
vector of length m. The function `g` is the Jacobian of f, and should return an m x
n matrix. `initial_x` is an initial guess for the solution.
Implements box constraints as described in Kanzow, Yamashita, Fukushima (2004; J
Comp & Applied Math).
# Keyword arguments
* `x_tol::Real=1e-8`: search tolerance in x
* `g_tol::Real=1e-12`: search tolerance in gradient
* `p`: The power of the norm for iterative weight update
* `update_weights`: wether to update the weights in the inner iteration loops
* `maxIter::Integer=1000`: maximum number of iterations
* `min_step_quality=1e-3`: for steps below this quality, the trust region is shrinked
* `good_step_quality=0.75`: for steps above this quality, the trust region is expanded
* `lambda::Real=10`: (inverse of) initial trust region radius
* `tau=Inf`: set initial trust region radius using the heuristic : tau*maximum(jacobian(df)'*jacobian(df))
* `lambda_increase=10.0`: `lambda` is multiplied by this factor after step below min quality
* `lambda_decrease=0.1`: `lambda` is multiplied by this factor after good quality steps
* `show_trace::Bool=false`: print a status summary on each iteration if true
* `keep_best`: If true, return the best solution instead of that of the last iteration. If callback is supplied, use the best value from it.
* `lower,upper=[]`: bound solution to these limits
* `damping_mode::Symbol=:fletcher`: damping strategy (`:fletcher`, `:fletcher_floor`)
* `solver::Symbol=:cholesky`: linear solver (`:direct`, `:cholesky`, `:cg`, `:qr`)
* `cg_tol::Real=1e-6`: relative residual tolerance for `solver=:cg`; when adaptive tolerance is enabled, this is the tight lower tolerance
* `cg_maxiter::Integer=500`: maximum CG iterations for `solver=:cg`
* `cg_preconditioner_floor::Real=eps(T)`: lower floor for required Jacobi preconditioner entries
* `cg_adaptive_tol::Bool=true`: loosen CG solves early and tighten as the LM gradient decreases
* `cg_tol_max::Real=1e-2`: loose upper tolerance for adaptive CG
"""

# I think a smarter way to do this *might* be to create a type similar to `OnceDifferentiable`
# and the like. This way we could not only merge the two functions, but also have a convenient
# way to provide an autodiff-made acceleration when someone doesn't provide an `avv`.
# it would probably be very inefficient performace-wise for most cases, but it wouldn't hurt to have it somewhere
function levenberg_marquardt_gpu() end

function levenberg_marquardt(
    df::OnceDifferentiable,
    initial_x::AbstractVector{T};
    p=2.0,
    update_weights=true,
    x_tol::Real=1e-8,
    g_tol::Real=1e-12,
    maxIter::Integer=1000,
    lambda=T(10),
    tau=T(Inf),
    lambda_increase::Real=2.0,
    lambda_decrease::Real=0.2,
    min_step_quality::Real=1e-3,
    good_step_quality::Real=0.75,
    show_trace::Bool=false,
    lower::AbstractVector{T}=Array{T}(undef, 0),
    upper::AbstractVector{T}=Array{T}(undef, 0),
    avv!::Union{Function,Nothing,LsqFit.Avv}=nothing,
    callback=nothing,
    keep_best=true,
    earlystop=0,
    damping_mode::Symbol=:fletcher,
    solver::Symbol=:cholesky,
    cg_tol::Real=1e-6,
    cg_maxiter::Int=500,
    cg_preconditioner_floor::Real=eps(T),
    cg_adaptive_tol::Bool=true,
    cg_tol_max::Real=1e-2,
) where {T}

    damping_mode in (:fletcher, :fletcher_floor) ||
        throw(ArgumentError("Unknown damping_mode: $damping_mode. Use :fletcher or :fletcher_floor."))
    solver in (:direct, :cholesky, :cg, :qr) ||
        throw(ArgumentError("Unknown solver: $solver. Use :direct, :cholesky, :cg, or :qr."))
    cg_maxiter > 0 || throw(ArgumentError("cg_maxiter must be positive, got $cg_maxiter"))
    cg_tol > 0 || throw(ArgumentError("cg_tol must be positive, got $cg_tol"))
    cg_tol_max >= cg_tol ||
        throw(ArgumentError("cg_tol_max must be greater than or equal to cg_tol."))
    cg_preconditioner_floor > 0 ||
        throw(ArgumentError("cg_preconditioner_floor must be positive, got $cg_preconditioner_floor"))

    # First evaluation
    value_jacobian!!(df, initial_x)

    if isfinite(tau)
        lambda = tau * maximum(jacobian(df)' * jacobian(df))
    end


    # check parameters
    (
        (isempty(lower) || length(lower) == length(initial_x)) &&
        (isempty(upper) || length(upper) == length(initial_x))
    ) || throw(
        ArgumentError(
            "Bounds must either be empty or of the same length as the number of parameters.",
        ),
    )
    (
        (isempty(lower) || all(initial_x .>= lower)) &&
        (isempty(upper) || all(initial_x .<= upper))
    ) || throw(ArgumentError("Initial guess must be within bounds."))
    (0 <= min_step_quality < 1) ||
        throw(ArgumentError(" 0 <= min_step_quality < 1 must hold."))
    (0 < good_step_quality <= 1) ||
        throw(ArgumentError(" 0 < good_step_quality <= 1 must hold."))
    (min_step_quality < good_step_quality) ||
        throw(ArgumentError("min_step_quality < good_step_quality must hold."))


    # other constants
    MAX_LAMBDA = 1e16 # minimum trust region radius
    MIN_LAMBDA = 1e-16 # maximum trust region radius
    MIN_DIAGONAL = 1e-6 # lower bound on values of diagonal matrix used to regularize the trust region step
    WEIGHT_SHIFT = 1e-4 # Shift for the wight to avoid numerical instability when p - 2 < 0


    converged = false
    x_converged = false
    g_converged = false
    iterCt = 0
    initial_grad_norm = T(NaN)
    x = copy(initial_x)
    best_x = copy(initial_x)
    delta_x = copy(initial_x)

    trial_f = similar(value(df))

    # Create buffers
    n = length(x)
    m = length(trial_f)
    use_cg_solver = solver === :cg
    JJ = use_cg_solver ? Matrix{T}(undef, 0, 0) : Matrix{T}(undef, n, n)
    n_buffer = Vector{T}(undef, n)
    Jdelta_buffer = similar(value(df))
    val_rmse = T[]
    DtD = zeros(T, n)
    cg_tmp = Vector{T}(undef, m)
    cg_Ap = Vector{T}(undef, n)
    cg_r = Vector{T}(undef, n)
    cg_z = Vector{T}(undef, n)
    cg_p = Vector{T}(undef, n)
    cg_M = Vector{T}(undef, n)
    cg_damping_diag = Vector{T}(undef, n)
    cg_stats = CGSolveStats()
    weighted_tmp = Vector{T}(undef, m)

    # Initialised weights
    wt = ones(T, m)
    if p != 2.0
        wt .= abs.(value(df, initial_x) .+ WEIGHT_SHIFT) .^ (p - 2.0)
    end
    #wtm = diagm(wt)
    residual = sum(abs2.(value(df)) .* wt)
    best_residual = residual


    # and an alias for the jacobian
    J = @timeit to "jacobian" jacobian(df)
    J2 = use_cg_solver ? Matrix{T}(undef, 0, 0) : copy(J)

    v = Array{T}(undef, n)

    # Maintain a trace of the system.
    tr = OptimBase.OptimizationTrace{LevenbergMarquardt}()
    if show_trace
        d = Dict("lambda" => lambda)
        os = OptimBase.OptimizationState{LevenbergMarquardt}(
            iterCt,
            sum(abs2, value(df)),
            NaN,
            d,
        )
        push!(tr, os)
        println(os)
    end

    while (~converged && iterCt < maxIter)
        # jacobian! will check if x is new or not, so it is only actually
        # evaluated if x was updated last iteration.
        @timeit to "jacobian" jacobian!(df, x) # has alias J


        # Page 170 least square data fitting and applications
        # we want to solve:
        #    argmin 0.5*||J(x)*delta_x + f(x)||^2 + lambda*||diagm(J'*J)*delta_x||^2
        # Note the form below that DtD is diag(J'WJ) where W is assumed to be the identity matrix
        # This DtD is for the normalisation of the λ term - seed Gavin 2020 page 3
        # Solving for the minimum gives:
        #    (J'*J + lambda*diagm(DtD)) * delta_x == -J' * f(x), where DtD = sum(abs2, J,1)
        # Where we have used the equivalence: diagm(J'*J) = diagm(sum(abs2, J,1))
        # It is additionally useful to bound the elements of DtD below to help
        # prevent "parameter evaporation".

        # Vector of the diagonal elements
        #DtD = vec(sum(abs2, J, dims=1))
        @timeit to "ATWA_DIAG!" ATWA_DIAG!(DtD, J, wt)

        # Scaled the lower bound by the mean value of the weight
        wt_mean = mean(wt)
        for i = 1:length(DtD)
            if DtD[i] <= MIN_DIAGONAL * wt_mean
                DtD[i] = MIN_DIAGONAL * wt_mean
            end
        end

        # delta_x = ( J'*W*J + lambda * Diagonal(DtD) ) \ ( -J'*value(df) )
        # Faster version J' * wt * J since wt is a diagonal matrix
        # This is the most tiem consuming part....
        #@timeit to "mul!(JJ)" mul!(JJ, transpose(J .* wt), J)
        #@timeit to "ATWA! (mul!(JJ))" ATWA!(JJ, J, wt)

        #n_buffer is delta C, JJ is g compared to Mark's code
        # This computes the right hand side term of the equation above
        _weighted_jtr_neg!(n_buffer, J, wt, value(df), weighted_tmp)
        grad_norm = norm(n_buffer, Inf)
        if !isfinite(grad_norm)
            lambda = min(max(lambda_increase * lambda, MIN_LAMBDA), MAX_LAMBDA)
            iterCt += 1
            continue
        end
        if !isfinite(initial_grad_norm)
            initial_grad_norm = max(grad_norm, eps(T))
        end
        current_cg_tol = if use_cg_solver
            @timeit to "adaptive cg tol" _adaptive_cg_tol(
                cg_tol,
                cg_tol_max,
                grad_norm,
                initial_grad_norm,
                cg_adaptive_tol,
                T,
            )
        else
            T(cg_tol)
        end

        # Solve the matrix equation (A*x == B). Strict Cholesky failure rejects
        # the step and retries later with a stronger damping parameter.
        normal_factor = nothing
        solve_ok = if use_cg_solver
            @timeit to "cg solve" _cg_solve_full!(
                v,
                J,
                wt,
                lambda,
                DtD,
                n_buffer,
                cg_tmp,
                cg_Ap,
                cg_r,
                cg_z,
                cg_p,
                cg_M,
                cg_damping_diag;
                damping_mode,
                maxiter=cg_maxiter,
                tol=current_cg_tol,
                preconditioner_floor=cg_preconditioner_floor,
                stats=cg_stats,
            )
        else
            # Embed the weights into the jacobian matrix - equivalent to JTAJ
            @timeit to "JJ2" _build_normal_full!(JJ, J2, J, wt, lambda, DtD, damping_mode)
            normal_factor, factor_ok = @timeit to "matinv" _factor_normal_full!(JJ, solver)
            factor_ok && _ldiv_normal_full!(v, normal_factor, n_buffer)
            factor_ok
        end
        if !solve_ok
            lambda = min(max(lambda_increase * lambda, MIN_LAMBDA), MAX_LAMBDA)
            iterCt += 1
            continue
        end
        v_norm = norm(v)
        if !isfinite(v_norm) || !all(isfinite, v)
            lambda = min(max(lambda_increase * lambda, MIN_LAMBDA), MAX_LAMBDA)
            iterCt += 1
            continue
        end


        delta_x .= v
        if !all(isfinite, delta_x)
            lambda = min(max(lambda_increase * lambda, MIN_LAMBDA), MAX_LAMBDA)
            iterCt += 1
            continue
        end



        # Box contraint from the original implementation - leave out for now
        # # apply box constraints
        # if !isempty(lower)
        #     @simd for i in 1:n
        #        @inbounds delta_x[i] = max(x[i] + delta_x[i], lower[i]) - x[i]
        #     end
        # end
        # if !isempty(upper)
        #     @simd for i in 1:n
        #        @inbounds delta_x[i] = min(x[i] + delta_x[i], upper[i]) - x[i]
        #     end
        # end

        # if the linear assumption is valid, our new residual should be:
        # e.g. J * Δx
        mul!(Jdelta_buffer, J, delta_x)
        # Predicted new residual vector
        Jdelta_buffer .= Jdelta_buffer .+ value(df)
        # Total squared residual - loss function
        # NOTE: this needs to be changed with iterative weight update
        predicted_residual = transpose(Jdelta_buffer) * (wt .* Jdelta_buffer)
        if !isfinite(predicted_residual)
            lambda = min(max(lambda_increase * lambda, MIN_LAMBDA), MAX_LAMBDA)
            iterCt += 1
            continue
        end

        # try the step and compute its quality
        # compute it inplace according to NLSolversBase value(obj, cache, state)
        # interface. No bang (!) because it doesn't update df besides mutating
        # the number of f_calls

        # re-use n_buffer - for x^{n+1}
        @timeit to "Buffer update" n_buffer .= x .+ delta_x
        @timeit to "NN Eval" value(df, trial_f, n_buffer)

        # update the sum of squares
        trial_residual = sum(abs2.(trial_f) .* wt)
        if !isfinite(trial_residual)
            lambda = min(max(lambda_increase * lambda, MIN_LAMBDA), MAX_LAMBDA)
            iterCt += 1
            continue
        end

        # step quality = residual change / predicted residual change
        # the higher the better?
        rho = (trial_residual - residual) / (predicted_residual - residual)
        if isfinite(rho) && trial_residual < residual && rho > min_step_quality
            # apply the step to x - n_buffer is ready to be used by the delta_x
            # calculations after this step.
            x .= n_buffer
            # There should be an update_x_value to do this safely
            # Copy the x and f(x) - avoid calling the function the next time
            copyto!(df.x_f, x)
            copyto!(value(df), trial_f)
            # Update the residual
            residual = trial_residual
            # Keep track the best solution
            if (residual < best_residual) && isnothing(callback)
                best_residual = residual
                best_x .= x
            end
            if rho > good_step_quality
                # increase trust region radius
                lambda = max(lambda_decrease * lambda, MIN_LAMBDA)
            end
        else
            # decrease trust region radius
            lambda = min(lambda_increase * lambda, MAX_LAMBDA)
        end

        iterCt += 1

        # Update weights
        if p != 2.0 && update_weights
            vdf = value(df)
            for (i, v) in enumerate(wt)
                t = abs(vdf[i] + WEIGHT_SHIFT)^(p - 2.0)
                wt[i] = t
            end
            residual = sum(abs2.(value(df)) .* wt)
            if !isfinite(residual)
                lambda = min(max(lambda_increase * lambda, MIN_LAMBDA), MAX_LAMBDA)
                continue
            end
        end

        _weighted_jtr_neg!(n_buffer, J, wt, value(df), weighted_tmp)
        g_norm = norm(n_buffer, Inf)
        if !isfinite(g_norm)
            lambda = min(max(lambda_increase * lambda, MIN_LAMBDA), MAX_LAMBDA)
            continue
        end

        # show state
        if show_trace
            d = Dict(
                "g(x)" => g_norm,
                "dx" => copy(delta_x),
                "lambda" => lambda,
                "cg_tol_eff" => current_cg_tol,
                "cg_iterations" => use_cg_solver ? cg_stats.iterations : 0,
                "cg_relres" => use_cg_solver ? cg_stats.relres : NaN,
            )
            os = OptimBase.OptimizationState{LevenbergMarquardt}(
                iterCt,
                sum(abs2, value(df)),
                g_norm,
                d,
            )
            push!(tr, os)
            println(os)
        end

        # check convergence criteria:
        # 1. Small weighted gradient: norm(J^T * W * value(df), Inf) < g_tol
        # 2. Small step size: norm(delta_x) < x_tol
        if g_norm < g_tol
            g_converged = true
        end
        if norm(delta_x) < x_tol * (x_tol + norm(x))
            x_converged = true
        end
        converged = g_converged | x_converged

        if !isnothing(callback)
            @timeit to "callback eval" rmse_val, xtmp = callback()
            if length(val_rmse) > 0 && rmse_val < minimum(val_rmse)
                best_x .= xtmp
            end
            push!(val_rmse, rmse_val)
        end
        if earlystop > 0
            earlystopcheck(val_rmse, earlystop) && (converged = true)
        end
    end

    # Rewind to the best x
    if keep_best
        value_jacobian!!(df, best_x)
        x .= best_x
    end

    MultivariateOptimizationResults(
        LevenbergMarquardt(),    # method
        initial_x,             # initial_x
        x,                     # minimizer
        sum(abs2, value(df)),       # minimum
        iterCt,                # iterations
        !converged,            # iteration_converged
        x_converged,           # x_converged
        0.0,                   # x_tol
        0.0,
        false,                 # f_converged
        0.0,                   # f_tol
        0.0,
        g_converged,           # g_converged
        g_tol,                  # g_tol
        0.0,
        false,                 # f_increased
        tr,                    # trace
        first(df.f_calls),               # f_calls
        first(df.df_calls),               # g_calls
        0,                      # h_calls
    )
end

"Signal stop if the min is more than n cycles away"
earlystopcheck(array, n) = (length(array) - argmin(array)) > n

"""
Compute the output of 

\$ diag(A^T W A) \$
"""
function ATWA_DIAG!(out, A, W)
    m, n = size(A)
    fill!(out, 0)
    @assert size(out, 1) == n
    Threads.@threads for j = 1:n
        for i = 1:m
            @inbounds out[j] += A[i, j] * A[i, j] * W[i]
        end
    end
    out
end

"""
Implement the output of

\$ A^T W A \$
"""
function ATWA!(out, A, W)
    m, n = size(A)
    fill(out, 0)
    @assert size(out) == (n, n)
    @assert size(W, 1) == m
    for b = 1:n
        @info b
        for a = 1:n
            tmp = zero(eltype(A))
            for i = 1:m
                tmp += A[i, a] * W[i] * A[i, b]
            end
            out[a, b] = tmp
        end
    end
    out
end
