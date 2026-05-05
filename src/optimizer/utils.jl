# Utility functions for optimizers

using LinearAlgebra

"""
    check_convergence(opt::AbstractOptimizer, calc::AbstractCalc) -> Bool

Check convergence based on forces, stress, and energy criteria.

Returns `true` if all applicable criteria are satisfied.
"""
function check_convergence(opt::AbstractOptimizer, calc::AbstractCalc)
    rebuild_nl = is_variable_cell(calc)
    # Get forces
    F = eddp_forces(calc; rebuild_nl=rebuild_nl)
    fmax = maximum(norm.(eachcol(F)))

    # Check force convergence
    force_converged = fmax < opt.f_tol

    # For variable cell, also check stress
    if is_variable_cell(calc)
        S = optimizer_effective_stress(calc; rebuild_nl=true)
        smax = maximum(abs.(S))
        stress_converged = smax < opt.s_tol
        return force_converged && stress_converged
    end

    return force_converged
end

"""
    TrajectoryFrame{T}

A single frame in an optimization trajectory.
"""
struct TrajectoryFrame{T}
    iteration::Int
    energy::T
    fmax::T
    smax::T
    positions::Matrix{T}
    cell_matrix::Union{Nothing, Matrix{T}}
    forces::Union{Nothing, Matrix{T}}
    stress::Union{Nothing, Matrix{T}}
    extra_dim_norm::T
end

"""
    log_step!(opt::AbstractOptimizer, calc::AbstractCalc)

Log optimization step based on log_level.

# Log Levels
- `:minimal`: No per-step logging
- `:standard`: Energy, max force, max stress per iteration
- `:detailed`: Full state including velocities, step sizes, etc.
"""
function log_step!(opt::AbstractOptimizer, calc::AbstractCalc)
    keep_traj = hasproperty(opt, :keep_trajectory) && opt.keep_trajectory

    # Skip the full metric computation only when absolutely nothing is requested.
    if opt.log_level == :minimal && !keep_traj
        return
    end

    rebuild_nl = is_variable_cell(calc)
    # Compute metrics
    E = eddp_energy(calc; forces=false, rebuild_nl=rebuild_nl)
    F = eddp_forces(calc; rebuild_nl=rebuild_nl)
    fmax = maximum(norm.(eachcol(F)))

    smax = 0.0
    stress_mat = nothing
    if is_variable_cell(calc)
        S = optimizer_effective_stress(calc; rebuild_nl=true)
        smax = maximum(abs.(S))
        stress_mat = S
    end

    # Standard logging
    if opt.log_level == :standard
        if opt.iteration == 1
            println("Iter    Energy          Fmax          Smax")
            println("----  --------------  ------------  ------------")
        end
        @printf("%4d  %14.8f  %12.6e  %12.6e\n", opt.iteration, E, fmax, smax)
    end

    # Detailed logging (includes algorithm-specific info)
    if opt.log_level == :detailed
        if opt.iteration == 1
            println("="^70)
            println("Starting optimization with $(typeof(opt))")
            println("="^70)
        end
        println("\nIteration $(opt.iteration):")
        println("  Energy:    $E")
        println("  Max force: $fmax")
        println("  Max stress: $smax")

        # Algorithm-specific logging
        log_detailed!(opt)
    end

    # Store in trajectory if requested
    if keep_traj
        cell = get_cell(calc)
        pos = get_positions(cell)
        D = size(pos, 1)
        extra_norm = D > 3 ? maximum(norm.(eachcol(pos[4:end, :]))) : 0.0
        frame = TrajectoryFrame(
            opt.iteration,
            E,
            fmax,
            smax,
            copy(pos),
            copy(cellmat(cell)),
            copy(F),
            stress_mat !== nothing ? copy(stress_mat) : nothing,
            extra_norm,
        )
        push!(opt.trajectory, frame)
    end
end

"""
    log_detailed!(opt::AbstractOptimizer)

Log algorithm-specific detailed information.

Default implementation does nothing. Concrete types can override.
"""
function log_detailed!(opt::AbstractOptimizer)
    # Default: no additional logging
    nothing
end
