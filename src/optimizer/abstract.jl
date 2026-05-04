# Abstract optimizer interface for structural relaxation

"""
    AbstractOptimizer

Abstract base type for structural optimization algorithms.

All concrete optimizer types must implement:
- `step!(opt::AbstractOptimizer, calc::AbstractCalc)`: Perform one optimization step
- `converged(opt::AbstractOptimizer, calc::AbstractCalc)`: Check convergence
- `reset!(opt::AbstractOptimizer)`: Reset optimizer state

Optional methods:
- `initialize!(opt::AbstractOptimizer, calc::AbstractCalc)`: Initialize before first step
"""
abstract type AbstractOptimizer end

"""
    step!(opt::AbstractOptimizer, calc::AbstractCalc)

Perform one optimization step.

This method should:
1. Get current positions from calculator
2. Compute energy, forces, and stress (if needed)
3. Update positions according to algorithm
4. Set new positions in calculator
5. Update internal optimizer state
"""
function step! end

"""
    converged(opt::AbstractOptimizer, calc::AbstractCalc) -> Bool

Check if optimization has converged based on convergence criteria.

Returns `true` if converged, `false` otherwise.
"""
function converged end

"""
    reset!(opt::AbstractOptimizer)

Reset optimizer to initial state.

Useful for restarting optimization from current positions.
"""
function reset! end

"""
    initialize!(opt::AbstractOptimizer, calc::AbstractCalc)

Initialize optimizer before first optimization step.

Default implementation does nothing.
"""
function initialize!(opt::AbstractOptimizer, calc::AbstractCalc)
    # Default: no initialization needed
    nothing
end

"""
    optimize!(opt::AbstractOptimizer, calc::AbstractCalc; callback=nothing) -> AbstractOptimizer

Run optimization until convergence or `opt.itermax` is reached.

This helper provides a common execution loop for native optimizers. It returns
the optimizer instance so callers can inspect trajectory and convergence state.
"""
function optimize!(opt::AbstractOptimizer, calc::AbstractCalc; callback=nothing)
    while !converged(opt, calc) && (opt.iteration < opt.itermax)
        step!(opt, calc)
        isnothing(callback) || callback()
    end
    opt
end

"""
    get_iteration(opt::AbstractOptimizer) -> Int

Get current iteration number.
"""
function get_iteration(opt::AbstractOptimizer)
    return opt.iteration
end

"""
    is_variable_cell(calc::AbstractCalc) -> Bool

Check if calculator supports variable cell optimization.

Returns `true` if calc has a CellRelaxationModifier (i.e. is a MetaCalculator
with that modifier).
"""
function is_variable_cell(calc::AbstractCalc)
    !isnothing(get_cell_relaxation_modifier(calc)) && return true
    return false
end

"""
    get_cell_relaxation_modifier(calc::AbstractCalc)

Return the active `CellRelaxationModifier` for calculators that expose a
modifier pipeline, or `nothing` when absent.
"""
function get_cell_relaxation_modifier(calc::AbstractCalc)
    if hasproperty(calc, :modifiers)
        for mod in calc.modifiers
            if mod isa CellRelaxationModifier || nameof(typeof(mod)) == :CellRelaxationModifier
                return mod
            end
        end
    end
    return nothing
end

"""
    optimizer_effective_stress(calc::AbstractCalc; rebuild_nl=true)

Stress used by optimizer convergence/logging. For MetaCalculators with
CellRelaxationModifier, returns the external-pressure-adjusted stress.
Otherwise returns the reported stress tensor.
"""
function optimizer_effective_stress(calc::AbstractCalc; rebuild_nl=true)
    eddp_stress(calc; rebuild_nl=rebuild_nl)
end

function optimizer_effective_stress(calc::MetaCalculator; rebuild_nl=true)
    stress = eddp_stress(calc; rebuild_nl=rebuild_nl)
    mod = get_cell_relaxation_modifier(calc)
    isnothing(mod) && return stress
    stress .- mod.external_pressure
end

"""
    get_augmented_positions_internal(calc::AbstractCalc) -> Vector

Get positions, including augmented lattice DOF if variable cell.
"""
function get_augmented_positions_internal(calc::AbstractCalc)
    if is_variable_cell(calc)
        if hasproperty(calc, :modifiers)
            mod = get_cell_relaxation_modifier(calc)
            if !isnothing(mod)
                return get_augmented_positions(mod, get_cell(calc))[:]
            end
        end
    end

    return get_positions(get_cell(calc))[:]
end

"""
    set_augmented_positions_internal!(calc::AbstractCalc, x::AbstractVector)

Set positions, including augmented lattice DOF if variable cell.

Routes through the calculator's own `set_positions!` to ensure dirty flags
are managed correctly.
"""
function set_augmented_positions_internal!(calc::AbstractCalc, x::AbstractVector)
    if is_variable_cell(calc)
        if hasproperty(calc, :modifiers)
            ndim = size(get_positions(get_cell(calc)), 1)
            set_positions!(calc, reshape(x, ndim, :))
            return
        end
    else
        ndim = size(get_positions(get_cell(calc)), 1)
        set_positions!(calc, reshape(x, ndim, :))
    end
end
