# Native Two-Point Steepest Descent (TPSD) optimizer

using LinearAlgebra
using Parameters
using Printf

"""
    NativeTPSD{T<:AbstractFloat} <: AbstractOptimizer

Native implementation of Two-Point Steepest Descent (TPSD) optimizer.

This implementation uses the Barzilai-Borwein step size selection method
and bypasses Optim.jl overhead for improved performance.

# Parameters
- `f_tol::T = 1e-4`: Force convergence tolerance (max force magnitude)
- `s_tol::T = 1e-6`: Stress convergence tolerance (max stress component)
- `e_tol::T = 1e-5`: Energy convergence tolerance (energy change per atom)
- `convergence::String = "force"`: "force" for force/stress criteria or "step_energy" for repose-style predicted decrease
- `itermax::Int = 1000`: Maximum iterations
- `α_tol::T = 1e-10`: Minimum step size (alpha) tolerance
- `maxstep::T = 0.2`: Maximum displacement per iteration (Euclidean norm)
- `max_cell_strain::T = 0.1`: Maximum variable-cell strain increment
- `log_level::Symbol = :standard`: Logging verbosity (`:minimal`, `:standard`, `:detailed`)
- `keep_trajectory::Bool = false`: Store optimization trajectory

# Algorithm
Uses the Barzilai-Borwein (1988) two-point step size formula:
```
α_k = (s_{k-1}' s_{k-1}) / (s_{k-1}' y_{k-1})
```
where s = x_k - x_{k-1} and y = g_k - g_{k-1}.

# Example
```julia
calc = EDDPCalculator(cell, cf, nn)
opt = NativeTPSD(f_tol=1e-3, itermax=500)

while !converged(opt, calc) && opt.iteration < opt.itermax
    step!(opt, calc)
end

println("Converged in ", opt.iteration, " iterations")
```

# Performance
Native TPSD typically achieves 20-40% speedup compared to Optim.jl version.

# References
- Barzilai & Borwein (1988), IMA J. Numer. Anal., 8, 141-148

# See Also
- [`NativeFIRE`](@ref): Alternative dynamics-based optimizer
- [`relax!`](@ref): High-level relaxation interface
- [`opt_tpsd`](@ref): Standalone TPSD function
"""
@with_kw mutable struct NativeTPSD <: AbstractOptimizer
    # Convergence criteria
    f_tol::Float64 = 1e-4
    s_tol::Float64 = 1e-6
    e_tol::Float64 = 1e-5
    convergence::String = "force"
    require_residual_convergence::Bool = false
    itermax::Int = 1000
    α_tol::Float64 = 1e-10

    # Step size limiting
    maxstep::Float64 = 0.2
    max_cell_strain::Float64 = 0.1

    # Cell relaxation
    relax_cell::Bool = false
    ext_pressure::Matrix{Float64} = zeros(3, 3)
    cell_damping::Float64 = 1.0

    # Exact optimizer-step constraints
    step_constraints::Vector{Any} = Any[]

    # Logging
    log_level::Symbol = :standard  # :minimal, :standard, :detailed
    keep_trajectory::Bool = false
    trajectory_every::Int = 1

    # Internal state
    iteration::Int = 0
    x_prev::Vector{Float64} = Float64[]
    xp_prev::Vector{Float64} = Float64[]
    pxp_prev::Vector{Float64} = Float64[]
    cell_ref::Matrix{Float64} = zeros(0, 0)
    α::Float64 = 1.0
    step_trust_run::Float64 = 0.0
    e_prev::Float64 = Inf
    step_energy_derun::Float64 = 0.0
    step_energy_de::Float64 = Inf
    last_step_norm::Float64 = 0.0
    last_max_cell_strain::Float64 = 0.0
    converged_flag::Bool = false
    trajectory::Vector = []
end

"""
    initialize!(opt::NativeTPSD, calc::AbstractCalc)

Initialize TPSD optimizer before first step.
"""
function initialize!(opt::NativeTPSD, calc::AbstractCalc)
    pos = get_positions(calc)
    n_dof = length(pos)
    if opt.relax_cell
        n_dof += 9
    end

    opt.x_prev = zeros(Float64, n_dof)
    opt.xp_prev = zeros(Float64, n_dof)
    opt.pxp_prev = zeros(Float64, n_dof)
    opt.cell_ref = opt.relax_cell ? copy(get_cellmat(calc)[1:3, 1:3]) : zeros(0, 0)
    opt.α = 1.0
    opt.step_trust_run = 0.0
    opt.iteration = 0
    opt.e_prev = Inf
    opt.step_energy_derun = 0.0
    opt.step_energy_de = Inf
    opt.last_step_norm = 0.0
    opt.last_max_cell_strain = 0.0
    opt.converged_flag = false
    opt.trajectory = []

    nothing
end

"""
    step!(opt::NativeTPSD, calc::AbstractCalc)

Perform one TPSD optimization step.
"""
function step!(opt::NativeTPSD, calc::AbstractCalc)
    if opt.iteration == 0
        initialize!(opt, calc)
    end

    pos = get_positions(calc)
    D, nat = size(pos, 1), size(pos, 2)

    x = _augmented_x(opt, calc)

    E = eddp_energy(calc; forces=true, rebuild_nl=true)
    F = _projected_forces(opt, calc; rebuild_nl=opt.relax_cell, make_copy=false)
    F_opt = _strain_projected_forces(opt, calc, F)

    if !opt.relax_cell
        g = -vec(F_opt)
        if opt.iteration > 0
            s = x - opt.x_prev
            y = g - opt.xp_prev
            sy = dot(s, y)

            if abs(sy) > opt.α_tol && sy > 0
                ss = dot(s, s)
                opt.α = clamp(ss / sy, 1e-10, 1e10)
            elseif sy > 0
                opt.α = max(opt.α, 1e-10)
            else
                opt.α = max(opt.α / 2.0, 1e-8)
            end
        else
            opt.α = min(opt.maxstep / (norm(g) + 1e-10), 0.1)
        end

        dr = opt.α .* g
        apply_step_constraints!(opt.step_constraints, opt, calc, dr)
        norm_dr = sqrt(dot(dr, dr))
        if norm_dr > opt.maxstep
            dr .*= opt.maxstep / norm_dr
        end
        apply_step_constraints!(opt.step_constraints, opt, calc, dr)
        opt.last_step_norm = sqrt(dot(dr, dr))
        opt.last_max_cell_strain = 0.0
        x_new = x - dr

        _writeback_positions!(opt, calc, x_new)

        opt.x_prev .= x
        opt.xp_prev .= g
        opt.pxp_prev .= g
        opt.e_prev = E
        opt.iteration += 1
        record_step_energy_convergence!(opt, calc, dr, g)

        log_step!(opt, calc)
        return nothing
    end

    ionic_len = length(F_opt)
    xp = vec(F_opt)
    cell_gradient = _cell_gradient(opt, calc)
    xp = vcat(xp, vec(cell_gradient))
    apply_step_constraints!(opt.step_constraints, opt, calc, xp)
    pxp = copy(xp)
    cell_range = (ionic_len + 1):length(pxp)
    pxp[cell_range] ./= _cell_gradient_scale(opt, calc)

    # Repose-style TPSD scalar step and running trust region.
    if opt.iteration == 0
        opt.α = 1e-4
    else
        dx = x - opt.x_prev
        dxp = xp - opt.xp_prev
        dpxp = pxp - opt.pxp_prev
        denom = dot(dxp, dpxp)
        if isfinite(denom) && abs(denom) > opt.α_tol
            opt.α = abs(dot(dx, dxp) / denom)
        end
        opt.α = clamp(opt.α, 1e-10, 1e10)
        proposed_max = maximum(abs.(opt.α .* pxp))
        natoms = max(length(get_cell(calc)), 1)
        opt.step_trust_run += (proposed_max - opt.step_trust_run) / (10.0 * natoms^(1 / 3))
        trust = min(0.1, 2.0 * opt.step_trust_run)
        opt.α = min(opt.α, trust / (maximum(abs.(pxp)) + floatmin(Float64)))
    end

    dr = opt.α .* pxp
    apply_step_constraints!(opt.step_constraints, opt, calc, dr)
    _limit_cell_step!(opt, calc, dr)
    max_abs_dr = maximum(abs.(dr))
    if max_abs_dr > opt.maxstep
        dr .*= opt.maxstep / max_abs_dr
    end
    apply_step_constraints!(opt.step_constraints, opt, calc, dr)
    _limit_cell_step!(opt, calc, dr)
    opt.last_step_norm = sqrt(dot(dr, dr))
    opt.last_max_cell_strain = _cell_step_strain_norm(opt, calc, dr)
    x_new = x + dr

    _writeback_positions!(opt, calc, x_new)

    opt.x_prev .= x
    opt.xp_prev .= xp
    opt.pxp_prev .= pxp
    opt.e_prev = E
    opt.iteration += 1
    record_step_energy_convergence!(opt, calc, dr, xp)

    log_step!(opt, calc)
    nothing
end

"""
    converged(opt::NativeTPSD, calc::AbstractCalc) -> Bool

Check if TPSD optimization has converged.
"""
function converged(opt::NativeTPSD, calc::AbstractCalc)
    if opt.iteration == 0
        return false
    end

    # Check using utility function
    return check_convergence(opt, calc)
end

"""
    reset!(opt::NativeTPSD)

Reset TPSD optimizer to initial state.
"""
function reset!(opt::NativeTPSD)
    opt.iteration = 0
    opt.x_prev .= 0.0
    opt.xp_prev .= 0.0
    opt.pxp_prev .= 0.0
    opt.cell_ref = zeros(0, 0)
    opt.α = 1.0
    opt.step_trust_run = 0.0
    opt.e_prev = Inf
    opt.step_energy_derun = 0.0
    opt.step_energy_de = Inf
    opt.last_step_norm = 0.0
    opt.last_max_cell_strain = 0.0
    opt.converged_flag = false
    empty!(opt.trajectory)
    nothing
end

"""
    log_detailed!(opt::NativeTPSD)

Log detailed TPSD-specific information.
"""
function log_detailed!(opt::NativeTPSD)
    println("  Step size (α): $(opt.α)")
    println("  Energy change: $(opt.e_prev)")
end
