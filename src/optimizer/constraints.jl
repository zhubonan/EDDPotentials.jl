using LinearAlgebra

"""
    AbstractStepConstraint

Constraint applied directly to an optimizer step vector. These constraints
project the proposed displacement, unlike calculator modifiers that change the
energy/force model.
"""
abstract type AbstractStepConstraint end

"""
    ZeroTranslationConstraint()

Remove the mean Cartesian displacement from all atoms in an optimizer step.
"""
struct ZeroTranslationConstraint <: AbstractStepConstraint end

"""
    IonicConstraint(atoms)

Freeze all Cartesian degrees of freedom for the atoms listed in `atoms`.
"""
struct IonicConstraint <: AbstractStepConstraint
    atoms::Vector{Int}
    IonicConstraint(atoms) = new(collect(Int, atoms))
end

"""
    FixCellConstraint()

Zero all cell degrees of freedom in an optimizer step.
"""
struct FixCellConstraint <: AbstractStepConstraint end

"""
    FixVolumeConstraint()

Project the cell part of a proposed optimizer step so the updated cell preserves
the current volume exactly. If the proposed cell is singular or inverted, the
cell step is rejected.
"""
struct FixVolumeConstraint <: AbstractStepConstraint end

function apply_step_constraints!(constraints, opt::AbstractOptimizer, calc::AbstractCalc, dx::AbstractVector)
    isempty(constraints) && return dx
    for constraint in constraints
        constrain_step!(constraint, opt, calc, dx)
    end
    return dx
end

function constrain_step!(::ZeroTranslationConstraint, opt::AbstractOptimizer, calc::AbstractCalc, dx::AbstractVector)
    pos = get_positions(calc)
    D, nat = size(pos)
    nat == 0 && return dx
    ionic = reshape(view(dx, 1:(D * nat)), D, nat)
    shift = vec(sum(ionic; dims=2)) ./ nat
    for i in 1:nat
        @views ionic[:, i] .-= shift
    end
    return dx
end

function constrain_step!(constraint::IonicConstraint, opt::AbstractOptimizer, calc::AbstractCalc, dx::AbstractVector)
    pos = get_positions(calc)
    D, nat = size(pos)
    for atom in constraint.atoms
        1 <= atom <= nat || throw(BoundsError(1:nat, atom))
        start = (atom - 1) * D + 1
        dx[start:(start + D - 1)] .= 0
    end
    return dx
end

function constrain_step!(::FixCellConstraint, opt::AbstractOptimizer, calc::AbstractCalc, dx::AbstractVector)
    opt.relax_cell || return dx
    pos = get_positions(calc)
    D, nat = size(pos)
    dx[(D * nat + 1):end] .= 0
    return dx
end

function constrain_step!(::FixVolumeConstraint, opt::AbstractOptimizer, calc::AbstractCalc, dx::AbstractVector)
    opt.relax_cell || return dx

    pos = get_positions(calc)
    D, nat = size(pos)
    cell_start = D * nat + 1
    cell_step = reshape(view(dx, cell_start:length(dx)), 3, 3)
    cell_old = get_cellmat(calc)[1:3, 1:3]
    cell_trial = cell_old .+ cell_step

    vol_old = det(cell_old)
    vol_trial = det(cell_trial)
    if !isfinite(vol_trial) || abs(vol_trial) <= eps(real(vol_trial)) || sign(vol_trial) != sign(vol_old)
        cell_step .= 0
        return dx
    end

    scale = cbrt(abs(vol_old / vol_trial))
    cell_corrected = cell_trial .* scale
    cell_step .= cell_corrected .- cell_old
    return dx
end

function constrain_step!(constraint::AbstractStepConstraint, opt::AbstractOptimizer, calc::AbstractCalc, dx::AbstractVector)
    throw(MethodError(constrain_step!, (constraint, opt, calc, dx)))
end
