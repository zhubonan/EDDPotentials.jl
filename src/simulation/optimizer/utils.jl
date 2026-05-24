# Utility functions for optimizers

using LinearAlgebra

function _cell_volume(calc::AbstractCalc)
    abs(det(get_cellmat(calc)[1:3, 1:3]))
end

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
        proposed_pos = scaled_pos + delta_pos
        if hasproperty(opt, :step_constraints)
            apply_cartesian_position_constraints!(opt.step_constraints, opt, calc, pos, proposed_pos)
            apply_direct_position_constraints!(opt.step_constraints, opt, calc, scaled_pos, proposed_pos)
        end
        set_positions!(calc, proposed_pos)
    else
        set_positions!(calc, pos_new)
    end
end

function _restore_cell_state!(calc::AbstractCalc, cellmat_old::AbstractMatrix, positions_old::AbstractMatrix)
    cellmat_new = copy(get_cellmat(calc))
    cellmat_new[1:3, 1:3] .= cellmat_old[1:3, 1:3]
    set_cellmat!(calc, cellmat_new; scale_positions=false)
    set_positions!(calc, positions_old)
    nothing
end

function _projected_forces(opt::AbstractOptimizer, calc::AbstractCalc; kwargs...)
    F = copy(eddp_forces(calc; kwargs...))
    if hasproperty(opt, :step_constraints)
        apply_force_constraints!(opt.step_constraints, opt, calc, F)
    end
    F
end

function _projected_stress(opt::AbstractOptimizer, calc::AbstractCalc)
    S = eddp_stress(calc) .- opt.ext_pressure
    if hasproperty(opt, :step_constraints)
        apply_stress_constraints!(opt.step_constraints, opt, calc, S)
    end
    S
end

function _cell_gradient_from_stress(opt::AbstractOptimizer, calc::AbstractCalc, S_eff::AbstractMatrix)
    H = get_cellmat(calc)[1:3, 1:3]
    -opt.cell_damping * _cell_volume(calc) .* (S_eff * inv(H)')
end

function _cell_force_from_stress(opt::AbstractOptimizer, calc::AbstractCalc, S_eff::AbstractMatrix)
    -_cell_gradient_from_stress(opt, calc, S_eff)
end

function _limit_augmented_step!(opt::AbstractOptimizer, calc::AbstractCalc, dr::AbstractVector)
    pos = get_positions(calc)
    D, nat = size(pos)
    ionic = reshape(view(dr, 1:(D * nat)), D, nat)
    for i in 1:nat
        n = norm(view(ionic, :, i))
        if n > opt.maxstep
            @views ionic[:, i] .*= opt.maxstep / n
        end
    end

    if opt.relax_cell && hasproperty(opt, :maxstrain)
        cell_step = reshape(view(dr, (D * nat + 1):length(dr)), 3, 3)
        H = get_cellmat(calc)[1:3, 1:3]
        strain = cell_step / H
        strain = 0.5 .* (strain .+ transpose(strain))
        max_strain = maximum(abs.(_strain_to_voigt(strain)))
        if max_strain > opt.maxstrain
            cell_step .*= opt.maxstrain / max_strain
        end
    end
    dr
end

function _is_hydrostatic_pressure(P::AbstractMatrix; atol=1e-12)
    offdiag = copy(P)
    offdiag[diagind(offdiag)] .= 0.0
    maximum(abs.(offdiag)) <= atol || return false
    diagvals = diag(P)
    maximum(abs.(diagvals .- diagvals[1])) <= atol
end

function _optimizer_objective(opt::AbstractOptimizer, calc::AbstractCalc; forces=false, rebuild_nl=true)
    E = eddp_energy(calc; forces, rebuild_nl)
    if opt.relax_cell && any(!iszero, opt.ext_pressure)
        _is_hydrostatic_pressure(opt.ext_pressure) ||
            throw(ArgumentError("downhill_check with relax_cell=true only supports hydrostatic external pressure"))
        P = tr(opt.ext_pressure) / 3
        return E + P * _cell_volume(calc)
    end
    E
end

function _downhill_tolerance(opt::AbstractOptimizer, objective)
    max(opt.e_tol, sqrt(eps(Float64)) * max(1.0, abs(float(objective))))
end

function check_convergence(opt::AbstractOptimizer, calc::AbstractCalc)
    F = _projected_forces(opt, calc)
    fmax = maximum(norm.(eachcol(F)))
    force_converged = fmax < opt.f_tol

    if opt.relax_cell
        S = _projected_stress(opt, calc)
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
    F = _projected_forces(opt, calc)
    fmax = maximum(norm.(eachcol(F)))

    smax = 0.0
    stress_mat = nothing
    if opt.relax_cell
        S = _projected_stress(opt, calc)
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
