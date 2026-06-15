# Utility functions for optimizers

using LinearAlgebra

const _CONVERGENCE_MODES = ("step_energy", "force")

function _normalize_convergence_mode(mode::AbstractString)
    normalized = lowercase(strip(mode))
    normalized in _CONVERGENCE_MODES ||
        throw(ArgumentError("optimizer convergence must be one of step_energy or force; got `$mode`"))
    normalized
end

function _augmented_x(opt::AbstractOptimizer, calc::AbstractCalc)
    pos = opt.relax_cell ? _reference_positions(opt, calc) : get_positions(calc)
    x = vec(pos)
    if opt.relax_cell
        x = vcat(x, vec(_current_cell_strain(opt, calc)))
    end
    x
end

_symmetrize_matrix(A::AbstractMatrix) = 0.5 .* (A .+ transpose(A))

function _cell_ref(opt::AbstractOptimizer, calc::AbstractCalc)
    if hasproperty(opt, :cell_ref) && !isempty(opt.cell_ref)
        return opt.cell_ref
    end
    get_cellmat(calc)[1:3, 1:3]
end

function _current_cell_strain(opt::AbstractOptimizer, calc::AbstractCalc)
    H = get_cellmat(calc)[1:3, 1:3]
    H0 = _cell_ref(opt, calc)
    _symmetrize_matrix(H / H0)
end

function _reference_positions(opt::AbstractOptimizer, calc::AbstractCalc)
    pos = copy(get_positions(calc))
    opt.relax_cell || return pos
    size(pos, 1) >= 3 || return pos
    strain = _current_cell_strain(opt, calc)
    pos[1:3, :] .= inv(strain) * pos[1:3, :]
    pos
end

function _strain_projected_forces(opt::AbstractOptimizer, calc::AbstractCalc, F::AbstractMatrix)
    opt.relax_cell || return F
    size(F, 1) >= 3 || return F
    strain = _current_cell_strain(opt, calc)
    out = copy(F)
    out[1:3, :] .= _reference_strain_forces(strain, F[1:3, :])
    out
end

_reference_strain_forces(strain::AbstractMatrix, F::AbstractMatrix) = transpose(strain) * F

function _cell_block_range(calc::AbstractCalc)
    pos = get_positions(calc)
    D, nat = size(pos)
    (D * nat + 1):(D * nat + 9)
end

_cell_step_view(calc::AbstractCalc, dx::AbstractVector) = reshape(view(dx, _cell_block_range(calc)), 3, 3)

function _limit_cell_step!(opt::AbstractOptimizer, calc::AbstractCalc, dx::AbstractVector)
    opt.relax_cell || return dx
    hasproperty(opt, :max_cell_strain) || return dx
    opt.max_cell_strain > 0 || return dx

    strain_norm = _cell_step_strain_norm(opt, calc, dx)
    strain_norm > 0 || return dx
    if strain_norm > opt.max_cell_strain
        cell_step = _cell_step_view(calc, dx)
        cell_step .*= opt.max_cell_strain / strain_norm
    end
    dx
end

function _cell_step_strain_norm(opt::AbstractOptimizer, calc::AbstractCalc, dx::AbstractVector)
    opt.relax_cell || return 0.0
    last(_cell_block_range(calc)) <= length(dx) || return 0.0
    cell_step = _cell_step_view(calc, dx)
    norm(_symmetrize_matrix(cell_step))
end

function _cell_gradient(opt::AbstractOptimizer, calc::AbstractCalc)
    S_eff = _symmetrize_matrix(_projected_stress(opt, calc))
    strain = _current_cell_strain(opt, calc)
    volume = _float_value(CellBase.volume(get_cell(calc)))
    cell_gradient = volume .* (S_eff * transpose(inv(strain)))
    _symmetrize_matrix(cell_gradient)
end

function _cell_gradient_scale(opt::AbstractOptimizer, calc::AbstractCalc)
    nat = max(length(get_cell(calc)), 1)
    gamma = opt.cell_damping > 0 ? opt.cell_damping : 1.0
    gamma * nat * 10.0
end

function _cell_direction(opt::AbstractOptimizer, calc::AbstractCalc)
    _symmetrize_matrix(_cell_gradient(opt, calc) ./ _cell_gradient_scale(opt, calc))
end

_float_value(x) = try
    Float64(x)
catch
    Float64(ustrip(x))
end

function _external_pressure_scalar(opt::AbstractOptimizer)
    hasproperty(opt, :ext_pressure) || return 0.0
    size(opt.ext_pressure) == (3, 3) || return 0.0
    tr(opt.ext_pressure) / 3
end

function _optimizer_step_parameter(opt::AbstractOptimizer)
    if hasproperty(opt, :α)
        return opt.α
    elseif hasproperty(opt, :dt)
        return opt.dt
    end
    0.0
end

function _is_final_frame(opt::AbstractOptimizer, calc::AbstractCalc)
    opt.iteration >= opt.itermax || check_convergence(opt, calc)
end

function _trajectory_should_record(opt::AbstractOptimizer, calc::AbstractCalc)
    every = hasproperty(opt, :trajectory_every) ? max(opt.trajectory_every, 1) : 1
    opt.iteration % every == 0 || _is_final_frame(opt, calc)
end

function _atom_pos_size(calc::AbstractCalc)
    pos = get_positions(calc)
    (size(pos, 1), size(pos, 2))
end

function _writeback_positions!(opt::AbstractOptimizer, calc::AbstractCalc, x_new::Vector)
    pos_cart = get_positions(calc)
    D, nat = size(pos_cart, 1), size(pos_cart, 2)
    pos_new = reshape(x_new[1:D*nat], D, nat)

    if opt.relax_cell
        cellmat_old = get_cellmat(calc)
        strain_new = _symmetrize_matrix(reshape(x_new[D*nat+1:end], 3, 3))
        cellmat_new = strain_new * _cell_ref(opt, calc)
        cellmat_old[1:3, 1:3] .= cellmat_new
        set_cellmat!(calc, cellmat_old; scale_positions=false)

        proposed_pos = copy(pos_new)
        if D >= 3
            proposed_pos[1:3, :] .= strain_new * pos_new[1:3, :]
        end
        if hasproperty(opt, :step_constraints)
            apply_cartesian_position_constraints!(opt.step_constraints, opt, calc, pos_cart, proposed_pos)
        end
        set_positions!(calc, proposed_pos)
    else
        set_positions!(calc, pos_new)
    end
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

function record_step_energy_convergence!(
    opt::AbstractOptimizer,
    calc::AbstractCalc,
    step_vector::AbstractVector,
    gradient_vector::AbstractVector,
)
    hasproperty(opt, :step_energy_derun) || return nothing
    de = abs(dot(step_vector, gradient_vector))
    nat = max(length(get_cell(calc)), 1)
    denom = 10.0 * nat^(1 / 3)
    opt.step_energy_de = de
    opt.step_energy_derun += (log10(de + floatmin(Float64)) - opt.step_energy_derun) / denom
    nothing
end

function step_energy_converged(opt::AbstractOptimizer)
    hasproperty(opt, :step_energy_derun) || return false
    opt.iteration >= 10 || return false
    isfinite(opt.e_tol) && opt.e_tol > 0 || return false
    opt.step_energy_derun <= log10(opt.e_tol)
end

function residual_converged(opt::AbstractOptimizer, calc::AbstractCalc)
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

function check_convergence(opt::AbstractOptimizer, calc::AbstractCalc)
    if hasproperty(opt, :convergence)
        mode = _normalize_convergence_mode(opt.convergence)
        if mode == "step_energy"
            step_energy_converged(opt) || return false
            if hasproperty(opt, :require_residual_convergence) && opt.require_residual_convergence
                return residual_converged(opt, calc)
            end
            return true
        end
    end

    return residual_converged(opt, calc)
end

"""
    TrajectoryFrame{T}

A single frame in an optimization trajectory.
"""
struct TrajectoryFrame{T}
    iteration::Int
    energy::T
    enthalpy::T
    fmax::T
    smax::T
    smax_gpa::T
    step_energy_de::T
    step_energy_derun::T
    step_norm::T
    max_cell_strain::T
    optimizer_step::T
    convergence::String
    is_final::Bool
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
    if keep_traj && _trajectory_should_record(opt, calc)
        cell = get_cell(calc)
        pos = get_positions(cell)
        D = size(pos, 1)
        extra_norm = D > 3 ? maximum(norm.(eachcol(pos[4:end, :]))) : 0.0
        volume_term = _external_pressure_scalar(opt) * _float_value(CellBase.volume(cell))
        de = hasproperty(opt, :step_energy_de) ? opt.step_energy_de : Inf
        derun = hasproperty(opt, :step_energy_derun) ? opt.step_energy_derun : 0.0
        step_norm = hasproperty(opt, :last_step_norm) ? opt.last_step_norm : 0.0
        max_cell_strain = hasproperty(opt, :last_max_cell_strain) ? opt.last_max_cell_strain : 0.0
        mode = hasproperty(opt, :convergence) ? _normalize_convergence_mode(opt.convergence) : "force"
        frame = TrajectoryFrame(
            opt.iteration,
            E,
            E + volume_term,
            fmax,
            smax,
            eVAngToGPa(smax),
            de,
            derun,
            step_norm,
            max_cell_strain,
            _optimizer_step_parameter(opt),
            mode,
            _is_final_frame(opt, calc),
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
