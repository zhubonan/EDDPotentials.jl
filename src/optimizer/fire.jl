# Native Fast Inertial Relaxation Engine (FIRE) optimizer

using LinearAlgebra
using Parameters
using Printf

"""
    NativeFIRE{T<:AbstractFloat} <: AbstractOptimizer

Native implementation of Fast Inertial Relaxation Engine (FIRE) optimizer.

FIRE is a molecular dynamics-based minimization method that uses velocity
modifications and adaptive time stepping for efficient structural relaxation.

# Algorithm Parameters
- `dt::T = 0.1`: Initial/current timestep
- `maxstep::T = 0.2`: Maximum displacement per iteration
- `dtmax::T = 1.0`: Maximum timestep
- `Nmin::Int = 5`: Minimum steps before acceleration
- `finc::T = 1.1`: Timestep increase factor
- `fdec::T = 0.5`: Timestep decrease factor
- `astart::T = 0.1`: Initial mixing parameter (alpha)
- `fa::T = 0.99`: Alpha decrease factor
- `downhill_check::Bool = false`: Enable energy-based uphill detection

# Convergence Parameters
- `f_tol::T = 1e-2`: Force convergence tolerance
- `s_tol::T = 1e-1`: Stress convergence tolerance
- `e_tol::T = 1e-5`: Energy convergence tolerance
- `itermax::Int = 1000`: Maximum iterations

# Logging
- `log_level::Symbol = :standard`: Logging verbosity (`:minimal`, `:standard`, `:detailed`)
- `keep_trajectory::Bool = false`: Store optimization trajectory

# Algorithm
FIRE combines MD-like dynamics with adaptive damping:
1. Compute power P = v · F (velocity dot force)
2. If P > 0 (moving downhill):
   - Mix velocity with force direction: v = (1-α)v + α F̂|v|
   - Increase timestep: dt = min(dt * finc, dtmax)
   - Decrease mixing: α = α * fa
3. If P ≤ 0 (moving uphill):
   - Reset velocity: v = 0
   - Decrease timestep: dt = dt * fdec
   - Reset mixing: α = astart
4. Update positions: x = x + dt * v

# Example
```julia
calc = EDDPCalculator(cell, cf, nn)
opt = NativeFIRE(maxstep=0.2, f_tol=1e-3)

while !converged(opt, calc) && opt.iteration < opt.itermax
    step!(opt, calc)
end

println("Converged in ", opt.iteration, " iterations")
```

# Performance
FIRE is often 2-3x faster than TPSD for smooth energy landscapes.
Best for rough optimization far from minimum.

# References
- Bitzek et al. (2006), Phys. Rev. Lett., 97, 170201
- Guénolé et al. (2020), Comp. Mater. Sci., 175, 109584

# See Also
- [`NativeTPSD`](@ref): Alternative gradient-based optimizer
- [`relax!`](@ref): High-level relaxation interface
- [`opt_fire`](@ref): Standalone FIRE function
"""
@with_kw mutable struct NativeFIRE <: AbstractOptimizer
    # Algorithm parameters
    dt::Float64 = 0.1
    maxstep::Float64 = 0.2
    dtmax::Float64 = 1.0
    Nmin::Int = 5
    finc::Float64 = 1.1
    fdec::Float64 = 0.5
    astart::Float64 = 0.1
    fa::Float64 = 0.99
    downhill_check::Bool = false

    # Cell relaxation
    relax_cell::Bool = false
    ext_pressure::Matrix{Float64} = zeros(3, 3)
    cell_damping::Float64 = 0.1

    # Convergence criteria
    f_tol::Float64 = 1e-2
    s_tol::Float64 = 1e-1
    e_tol::Float64 = 1e-5
    itermax::Int = 1000

    # Logging
    log_level::Symbol = :standard
    keep_trajectory::Bool = false

    # Internal state
    iteration::Int = 0
    vel::Vector{Float64} = Float64[]
    a::Float64 = 0.1  # Current alpha (mixing parameter)
    Nsteps::Int = 0  # Steps since last reset
    e_last::Float64 = Inf
    x_last::Vector{Float64} = Float64[]
    converged_flag::Bool = false
    trajectory::Vector = []

    # Internal tracking
    dt_init::Float64 = 0.1  # Store initial dt for reset
end

"""
    initialize!(opt::NativeFIRE, calc::AbstractCalc)

Initialize FIRE optimizer before first step.
"""
function initialize!(opt::NativeFIRE, calc::AbstractCalc)
    pos = get_positions(calc)
    n_dof = length(pos)
    if opt.relax_cell
        n_dof += 9
    end

    opt.vel = zeros(Float64, n_dof)
    opt.a = opt.astart
    opt.Nsteps = 0
    opt.iteration = 0
    opt.e_last = Inf
    opt.x_last = zeros(Float64, n_dof)
    opt.converged_flag = false
    opt.trajectory = []

    opt.dt_init = opt.dt

    nothing
end

"""
    step!(opt::NativeFIRE, calc::AbstractCalc)

Perform one FIRE optimization step.
"""
function step!(opt::NativeFIRE, calc::AbstractCalc)
    if opt.iteration == 0
        initialize!(opt, calc)
        if opt.downhill_check
            opt.e_last = eddp_energy(calc; forces=false, rebuild_nl=true)
        end
    end

    x = _augmented_x(opt, calc)

    E = eddp_energy(calc; forces=true, rebuild_nl=true)
    F = eddp_forces(calc; rebuild_nl=opt.relax_cell, make_copy=false)

    gradient = vec(F)  # FIRE: force = gradient direction
    if opt.relax_cell
        S_eff = eddp_stress(calc) .- opt.ext_pressure
        g_cell = opt.cell_damping * 10 .* vec(S_eff)  # move in stress direction
        gradient = vcat(gradient, g_cell)
    end
    grad_norm = sqrt(dot(gradient, gradient))

    if grad_norm <= eps(Float64)
        opt.x_last = x
        opt.e_last = E
        opt.iteration += 1
        log_step!(opt, calc)
        return nothing
    end

    is_uphill = false
    if opt.downhill_check && opt.iteration > 0
        if E > opt.e_last
            _writeback_positions!(opt, calc, opt.x_last)
            x = opt.x_last
            is_uphill = true
            E = eddp_energy(calc; forces=true, rebuild_nl=true)
            F = eddp_forces(calc; rebuild_nl=opt.relax_cell, make_copy=false)
            gradient = vec(F)
            if opt.relax_cell
                S_eff = eddp_stress(calc) .- opt.ext_pressure
                gradient = vcat(gradient, opt.cell_damping * 10 .* vec(S_eff))
            end
            grad_norm = sqrt(dot(gradient, gradient))
        end
    end

    vf = dot(gradient, opt.vel)

    if vf > 0.0 && !is_uphill
        vel_norm = sqrt(dot(opt.vel, opt.vel))
        if grad_norm > eps(Float64)
            force_direction = gradient / grad_norm
            opt.vel = (1.0 - opt.a) * opt.vel + opt.a * force_direction * vel_norm
        end
        if opt.Nsteps >= opt.Nmin
            opt.dt = min(opt.dt * opt.finc, opt.dtmax)
            opt.a *= opt.fa
        end
        opt.Nsteps += 1
    else
        opt.vel .= 0.0
        opt.a = opt.astart
        opt.dt *= opt.fdec
        opt.Nsteps = 0
    end

    opt.vel += opt.dt * gradient

    dr = opt.dt * opt.vel
    normdr = sqrt(dot(dr, dr))
    if normdr > opt.maxstep
        dr *= opt.maxstep / normdr
    end

    x_new = x + dr
    _writeback_positions!(opt, calc, x_new)

    opt.x_last = x
    opt.e_last = E
    opt.iteration += 1

    log_step!(opt, calc)
    nothing
end

"""
    converged(opt::NativeFIRE, calc::AbstractCalc) -> Bool

Check if FIRE optimization has converged.
"""
function converged(opt::NativeFIRE, calc::AbstractCalc)
    if opt.iteration == 0
        return false
    end

    # Check using utility function
    return check_convergence(opt, calc)
end

"""
    reset!(opt::NativeFIRE)

Reset FIRE optimizer to initial state.
"""
function reset!(opt::NativeFIRE)
    opt.iteration = 0
    opt.vel .= 0.0
    opt.a = opt.astart
    opt.dt = opt.dt_init
    opt.Nsteps = 0
    opt.e_last = Inf
    opt.x_last .= 0.0
    opt.converged_flag = false
    empty!(opt.trajectory)
    nothing
end

"""
    log_detailed!(opt::NativeFIRE)

Log detailed FIRE-specific information.
"""
function log_detailed!(opt::NativeFIRE)
    println("  Timestep (dt): $(opt.dt)")
    println("  Mixing (α): $(opt.a)")
    println("  Steps since reset: $(opt.Nsteps)")
    println("  Velocity norm: $(sqrt(dot(opt.vel, opt.vel)))")
end
