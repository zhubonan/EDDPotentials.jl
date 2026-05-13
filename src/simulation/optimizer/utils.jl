# Utility functions for optimizers

using LinearAlgebra

function _augmented_x(opt::AbstractOptimizer, calc::AbstractCalc)
    pos = get_positions(calc)
    x = vec(pos)
    if opt.relax_cell
        x = vcat(x, vec(get_cellmat(calc)[1:3, 1:3]))
    end
    x
end

function _atom_pos_size(calc::AbstractCalc)
    pos = get_positions(calc)
    (size(pos, 1), size(pos, 2))
end

function _writeback_positions!(opt::AbstractOptimizer, calc::AbstractCalc, x_new::Vector)
    pos = get_positions(calc)
    D, nat = size(pos, 1), size(pos, 2)
    pos_new = reshape(x_new[1:D*nat], D, nat)
    delta_pos = pos_new - pos    # TPSD position step in Cartesian

    if opt.relax_cell
        cellmat_old = get_cellmat(calc)
        cellmat_step = reshape(x_new[D*nat+1:end], 3, 3)
        cell_3d = cellmat_old[1:3, 1:3]
        delta = cellmat_step - cell_3d
        strain = delta / cell_3d
        cellmat_new = (I + strain) * cell_3d
        cellmat_old[1:3, 1:3] .= cellmat_new
        # scale_positions=true keeps fractional coords constant during cell change
        set_cellmat!(calc, cellmat_old; scale_positions=true)
        # Re-read positions (now scaled to new cell), then apply TPSD delta
        scaled_pos = get_positions(calc)
        set_positions!(calc, scaled_pos + delta_pos)
    else
        set_positions!(calc, pos_new)
    end
end

function check_convergence(opt::AbstractOptimizer, calc::AbstractCalc)
    F = eddp_forces(calc)
    fmax = maximum(norm.(eachcol(F)))
    force_converged = fmax < opt.f_tol

    if opt.relax_cell
        S = eddp_stress(calc) .- opt.ext_pressure
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
    if opt.log_level == :minimal && !keep_traj
        return
    end

    E = eddp_energy(calc; forces=false, rebuild_nl=true)
    F = eddp_forces(calc)
    fmax = maximum(norm.(eachcol(F)))

    smax = 0.0
    stress_mat = nothing
    if opt.relax_cell
        S = eddp_stress(calc) .- opt.ext_pressure
        smax = maximum(abs.(S))
        stress_mat = S
    end

    if opt.log_level == :standard
        cell = get_cell(calc)
        pos = get_positions(cell)
        D = size(pos, 1)
        _hyper = D > 3
        if opt.iteration == 1
            if _hyper
                println("Iter    Energy          Fmax          Smax        |extra|")
                println("----  --------------  ------------  ------------  ----------")
            else
                println("Iter    Energy          Fmax          Smax")
                println("----  --------------  ------------  ------------")
            end
        end
        if _hyper
            extra_norm = maximum(norm.(eachcol(pos[4:end, :])))
            @printf("%4d  %14.8f  %12.6e  %12.6e  %10.4f\n", opt.iteration, E, fmax, smax, extra_norm)
        else
            @printf("%4d  %14.8f  %12.6e  %12.6e\n", opt.iteration, E, fmax, smax)
        end
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
