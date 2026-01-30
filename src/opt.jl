#=
Optimisation algorithms
=#
using LinearAlgebra

include("tpsd.jl")

"""

    opt_tpsd(vc::AbstractCalc; f_tol=1e-3, s_tol=1e-3, e_tol=1e-5, 
                  itermax=1000, trace=false, α_tol=1e-10, callback=nothing, 
                  trajectory=nothing)

Optimise a structure (calculation) using TPSD.
Reference:
    - Barzilai, J. and Borwein, J.M., 1988. Two-point step size gradient methods. IMA journal of numerical analysis, 8(1), pp.141-148

"""
function opt_tpsd(
    vc::AbstractCalc;
    f_tol=1e-4,
    s_tol=1e-6,
    e_tol=1e-5,
    itermax=1000,
    trace=0,
    α_tol=1e-10,
    callback=nothing,
    trajectory=nothing,
)
    f = eddp_forces(vc)
    f0 = similar(f)
    e0 = -9999.0
    x = get_positions(vc)
    step = similar(x)
    x0 = x

    # Initial step
    α = 1e-8
    dx = similar(x)
    df = similar(f)

    converged = false
    i = 1

    if !isa(vc, VariableCellCalc)
        s_tol = floatmax()
    end

    while i < itermax

        e = eddp_energy(vc)
        f = eddp_forces(vc)

        if isa(vc, VariableCellCalc)
            fa = eddp_forces(vc.calc)
            sa = eddp_stress(vc)
        else
            fa = f
            sa = eddp_stress(vc)
        end

        fmax = maximum(norm.(eachcol(fa)))
        smax = maximum(abs.(sa))
        de = abs(e - e0)

        # Check for convergence criteria
        if fmax < f_tol && smax < s_tol && de < e_tol
            converged = true
        end

        #  Compute the step size
        dx .= x .- x0
        df .= f .- f0
        if i > 1
            α = abs(dot(dx, df) / (dot(df, df) + floatmin(eltype(x))))
        end

        # Show trace information
        if (trace > 0) && (i % trace == 0)
            @info @sprintf(
                "Iteration %d  |F|= %-10.5g |Smax|= %-10.5g dE= %-10.5g α= %-10.5g |step|= %-10.5g",
                i,
                fmax,
                smax,
                de,
                α,
                norm(step)
            )
        end

        # Check if Alpha is too small
        if i > 10 && α < α_tol
            converged = false
            break
        end

        # Check if we are converged
        converged && break

        # Update variables 
        copy!(x0, x)
        copy!(f0, f)
        e0 = e

        # Compute step size 
        step .= f .* α

        # Apply the step
        x = x .+ step
        set_positions!(vc, x)

        # callback function
        isnothing(callback) || callback()

        isnothing(trajectory) || push!(trajectory, deepcopy(get_cell(vc)))

        i += 1
    end
    converged
end

"""
    opt_fire(vc::AbstractCalc; f_tol=1e-2, s_tol=1e-1, e_tol=1e-5,
             itermax=1000, maxstep=0.2, dt=0.1, dtmax=1.0, Nmin=5,
             finc=1.1, fdec=0.5, astart=0.1, fa=0.99, downhill_check=false,
             trace=false, callback=nothing, trajectory=nothing)

Optimize a structure using the Fast Inertial Relaxation Engine (FIRE) algorithm.

FIRE is a molecular dynamics-based minimization method that uses velocity modifications
and adaptive time stepping. Often faster than TPSD for smooth energy landscapes.

# Algorithm Parameters
- `dt::Float64 = 0.1`: Initial timestep
- `maxstep::Float64 = 0.2`: Maximum displacement per iteration
- `dtmax::Float64 = 1.0`: Maximum timestep
- `Nmin::Int = 5`: Steps before acceleration
- `finc::Float64 = 1.1`: Timestep increase factor
- `fdec::Float64 = 0.5`: Timestep decrease factor
- `astart::Float64 = 0.1`: Initial mixing parameter (alpha)
- `fa::Float64 = 0.99`: Alpha decrease factor
- `downhill_check::Bool = false`: Enable energy-based uphill detection

# Convergence Parameters
- `f_tol::Float64 = 1e-2`: Force convergence tolerance
- `s_tol::Float64 = 1e-1`: Stress convergence tolerance
- `e_tol::Float64 = 1e-5`: Energy convergence tolerance
- `itermax::Int = 1000`: Maximum iterations

# Other
- `trace::Bool = false`: Print progress
- `callback::Function`: Called after each iteration
- `trajectory::Vector`: Store cells if provided

# References
- Bitzek et al. (2006), Phys. Rev. Lett., 97, 170201
- Guénolé et al. (2020), Comp. Mater. Sci., 175, 109584

# See Also
- [`opt_tpsd`](@ref): Alternative TPSD optimizer
"""
function opt_fire(
    vc::AbstractCalc;
    f_tol=1e-2,
    s_tol=1e-1,
    e_tol=1e-5,
    itermax=1000,
    maxstep=0.2,
    dt=0.1,
    dtmax=1.0,
    Nmin=5,
    finc=1.1,
    fdec=0.5,
    astart=0.1,
    fa=0.99,
    downhill_check=false,
    trace=false,
    callback=nothing,
    trajectory=nothing,
)
    # Get initial state
    x = get_positions(vc)
    vel = zeros(size(x))

    # FIRE state variables
    α = astart
    dt_current = dt
    Nsteps = 0

    # For downhill checking
    e_last = Inf
    x_last = copy(x)

    converged = false
    i = 1

    # Check if variable cell
    if !isa(vc, VariableCellCalc)
        s_tol = floatmax()
    end

    while i <= itermax
        # Compute energy and forces
        e = eddp_energy(vc)
        f = eddp_forces(vc)

        if isa(vc, VariableCellCalc)
            fa = eddp_forces(vc.calc)
            sa = eddp_stress(vc)
        else
            fa = f
            sa = eddp_stress(vc)
        end

        fmax = maximum(norm.(eachcol(fa)))
        smax = maximum(abs.(sa))
        de = abs(e - e_last)

        # Check convergence
        if fmax < f_tol && smax < s_tol
            converged = true
        end

        # Print trace
        if trace
            @info @sprintf(
                "FIRE Iter %d  |F|= %-10.5g |Smax|= %-10.5g E= %-10.5g dt= %-10.5g α= %-10.5g Nsteps= %d",
                i,
                fmax,
                smax,
                e,
                dt_current,
                α,
                Nsteps
            )
        end

        # Check if converged
        converged && break

        # Optional downhill check
        is_uphill = false
        if downhill_check && i > 1
            if e > e_last
                # Energy increased - revert
                set_positions!(vc, x_last)
                x = x_last
                is_uphill = true
            end
        end

        # Compute power P = v · F
        vf = sum(vel .* f)

        # FIRE velocity modification or reset
        if vf > 0.0 && !is_uphill
            # Moving downhill - mix velocity with force direction
            vel_norm = sqrt(sum(vel .^ 2))
            f_norm = sqrt(sum(f .^ 2))

            if f_norm > eps(Float64)
                vel = (1.0 - α) * vel + α * (f / f_norm) * vel_norm
            end

            # Increase dt and decrease α if enough steps
            if Nsteps >= Nmin
                dt_current = min(dt_current * finc, dtmax)
                α *= fa
            end
            Nsteps += 1
        else
            # Moving uphill or P ≤ 0 - reset
            vel .= 0.0
            α = astart
            dt_current *= fdec
            Nsteps = 0
        end

        # Update velocity (Velocity Verlet)
        vel += dt_current * f

        # Compute displacement
        dr = dt_current * vel
        normdr = sqrt(sum(dr .^ 2))
        if normdr > maxstep
            dr = dr * (maxstep / normdr)
        end

        # Store state for downhill check
        x_last = copy(x)
        e_last = e

        # Update positions
        x = x + dr
        set_positions!(vc, x)

        # Callback
        isnothing(callback) || callback()

        isnothing(trajectory) || push!(trajectory, deepcopy(get_cell(vc)))

        i += 1
    end

    converged
end
