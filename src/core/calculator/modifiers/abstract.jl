"""
Abstract base type for calculator modifiers.

Modifiers can:
1. Pre-process: Transform the cell before source calculation
2. Post-process: Transform energy/forces/stress after source calculation
3. Augment: Add additional contributions to results
"""
abstract type AbstractModifier end

"""
    pre_calculate!(modifier::AbstractModifier, cell::Cell)

Called before source calculator runs. Use this for coordinate transformations
or other preparations that need to happen before the source calculation.

Default implementation: does nothing.
"""
function pre_calculate!(modifier::AbstractModifier, cell::Cell)
    nothing
end

"""
    post_calculate!(modifier::AbstractModifier, cell::Cell, energy, forces, stress)

Called after source calculator runs. Receives the current results and returns
potentially modified (energy, forces, stress) tuple.

This is where most modifiers do their work - adding penalties, transforming
coordinates, applying pressure, etc.

Default implementation: returns inputs unchanged.
"""
function post_calculate!(
    modifier::AbstractModifier,
    cell::Cell,
    energy::T,
    forces::Matrix{T},
    stress::Matrix{T}
) where {T}
    return (energy, forces, stress)
end

"""
    requires_forces(modifier::AbstractModifier) -> Bool

Returns true if this modifier needs forces to be computed.
If false, the modifier only affects energy.

Default: false
"""
function requires_forces(modifier::AbstractModifier)
    return false
end

"""
    get_parameter(modifier::AbstractModifier, name::Symbol)

Get a parameter value from the modifier by name.
"""
function get_parameter(modifier::AbstractModifier, name::Symbol)
    return getfield(modifier, name)
end

"""
    set_parameter!(modifier::AbstractModifier, name::Symbol, value)

Set a parameter value on the modifier by name.
"""
function set_parameter!(modifier::AbstractModifier, name::Symbol, value)
    setfield!(modifier, name, value)
    return value
end
