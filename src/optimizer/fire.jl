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
calc = NNCalc(cell, cf, nn)
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
    # Get initial positions
    x = get_augmented_positions_internal(calc)

    # Initialize velocities to zero
    opt.vel = zeros(Float64, length(x))
    opt.a = opt.astart
    opt.Nsteps = 0
    opt.iteration = 0
    opt.e_last = Inf
    opt.x_last = zeros(Float64, length(x))
    opt.converged_flag = false
    opt.trajectory = []

    # Store initial dt for resets
    opt.dt_init = opt.dt

    nothing
end

"""
    step!(opt::NativeFIRE, calc::AbstractCalc)

Perform one FIRE optimization step.
"""
function step!(opt::NativeFIRE, calc::AbstractCalc)
    # Initialize on first step
    if opt.iteration == 0
        initialize!(opt, calc)
        # Get initial energy for downhill check
        if opt.downhill_check
            opt.e_last = eddp_energy(calc; forces=false, rebuild_nl=true)
        end
    end

    # Get current positions
    x = get_augmented_positions_internal(calc)

    # Compute energy and forces
    E = eddp_energy(calc; forces=true, rebuild_nl=true)
    F = eddp_forces(calc; rebuild_nl=false, make_copy=false)

    # Gradient (force, not negative)
    gradient = F[:]

    # Check if moving uphill (optional energy-based check)
    is_uphill = false
    if opt.downhill_check && opt.iteration > 0
        if E > opt.e_last
            # Energy increased - revert to last position
            set_augmented_positions_internal!(calc, opt.x_last)
            x = opt.x_last
            is_uphill = true
        end
    end

    # Compute power P = v · F
    vf = dot(gradient, opt.vel)

    # Determine if moving downhill
    if vf > 0.0 && !is_uphill
        # Moving downhill - modify velocity
        vel_norm = sqrt(dot(opt.vel, opt.vel))
        grad_norm = sqrt(dot(gradient, gradient))

        if grad_norm > eps(Float64)
            # Mix velocity with force direction: v = (1-α)v + α F̂|v|
            force_direction = gradient / grad_norm
            opt.vel = (1.0 - opt.a) * opt.vel + opt.a * force_direction * vel_norm
        end

        # Increase dt and decrease alpha if enough steps
        if opt.Nsteps >= opt.Nmin
            opt.dt = min(opt.dt * opt.finc, opt.dtmax)
            opt.a *= opt.fa
        end

        opt.Nsteps += 1
    else
        # Moving uphill or P ≤ 0 - reset
        opt.vel .= 0.0
        opt.a = opt.astart
        opt.dt *= opt.fdec
        opt.Nsteps = 0
    end

    # Velocity Verlet: update velocity
    opt.vel += opt.dt * gradient

    # Compute displacement
    dr = opt.dt * opt.vel

    # Limit step size
    normdr = sqrt(dot(dr, dr))
    if normdr > opt.maxstep
        dr *= opt.maxstep / normdr
    end

    # Update positions
    x_new = x + dr
    set_augmented_positions_internal!(calc, x_new)

    # Store state for next iteration
    opt.x_last = x
    opt.e_last = E
    opt.iteration += 1

    # Log step
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
