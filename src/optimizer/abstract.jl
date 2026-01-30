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
    get_iteration(opt::AbstractOptimizer) -> Int

Get current iteration number.
"""
function get_iteration(opt::AbstractOptimizer)
    return opt.iteration
end

"""
    is_variable_cell(calc::AbstractCalc) -> Bool

Check if calculator supports variable cell optimization.

Returns `true` if `calc` is a VariableCellCalc or has a CellRelaxationModifier.
"""
function is_variable_cell(calc::AbstractCalc)
    # Check for legacy VariableCellCalc
    if calc isa VariableCellCalc
        return true
    end

    # Check for MetaCalculator with CellRelaxationModifier
    if hasproperty(calc, :modifiers)
        for mod in calc.modifiers
            if nameof(typeof(mod)) == :CellRelaxationModifier
                return true
            end
        end
    end

    return false
end

"""
    get_augmented_positions_internal(calc::AbstractCalc) -> Vector

Get positions, including augmented lattice DOF if variable cell.
"""
function get_augmented_positions_internal(calc::AbstractCalc)
    if is_variable_cell(calc)
        # Try to get augmented positions
        if calc isa VariableCellCalc
            return get_positions(calc)[:]
        elseif hasproperty(calc, :modifiers)
            # MetaCalculator with CellRelaxationModifier
            for mod in calc.modifiers
                if nameof(typeof(mod)) == :CellRelaxationModifier
                    return get_augmented_positions(mod, get_cell(calc))[:]
                end
            end
        end
    end

    # Fixed cell: just return atomic positions
    return get_positions(get_cell(calc))[:]
end

"""
    set_augmented_positions_internal!(calc::AbstractCalc, x::AbstractVector)

Set positions, including augmented lattice DOF if variable cell.
"""
function set_augmented_positions_internal!(calc::AbstractCalc, x::AbstractVector)
    if is_variable_cell(calc)
        if calc isa VariableCellCalc
            set_positions!(calc, x)
        elseif hasproperty(calc, :modifiers)
            # MetaCalculator with CellRelaxationModifier
            for mod in calc.modifiers
                if nameof(typeof(mod)) == :CellRelaxationModifier
                    cell = get_cell(calc)
                    ndim = size(get_positions(cell), 1)
                    set_augmented_positions!(mod, cell, reshape(x, ndim, :))
                    return
                end
            end
        end
    else
        # Fixed cell: set atomic positions directly
        cell = get_cell(calc)
        ndim = size(get_positions(cell), 1)
        set_positions!(cell, reshape(x, ndim, :))
    end
end
