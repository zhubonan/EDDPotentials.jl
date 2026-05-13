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
