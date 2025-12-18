"""
    ExtraDimensionPenaltyModifier{T}

Adds harmonic penalty for extra dimensions (d > 3).

This modifier implements a harmonic restoring force that penalizes
motion in dimensions beyond 3. Useful for hyperdimensional relaxation
where extra dimensions are used temporarily.

# Mathematical Form
```math
E' = E + α ∑_{i>3} ∑_atoms x_i^2 / 2
F'_i = F_i - 2α x_i  for i > start_dim
F'_i = F_i          for i ≤ start_dim
```

# Fields
- `alpha::T`: Penalty strength parameter
- `start_dim::Int`: First dimension to apply penalty (typically 4)

# Example
```julia
# Create modifier with penalty strength 0.001
modifier = ExtraDimensionPenaltyModifier(0.001)

# Or specify custom start dimension
modifier = ExtraDimensionPenaltyModifier(0.001; start_dim=5)

# Use in MetaCalculator
calc = MetaCalculator(source, [modifier])

# Modify penalty during relaxation
modifier.alpha = 0.1
# Or: set_parameter!(modifier, :alpha, 0.1)
```
"""
mutable struct ExtraDimensionPenaltyModifier{T} <: AbstractModifier
    alpha::T
    start_dim::Int
end

"""
    ExtraDimensionPenaltyModifier(alpha::T; start_dim=4) where T

Create an extra dimension penalty modifier.

# Arguments
- `alpha::T`: Penalty strength (typically 1e-4 to 1e-2)
- `start_dim=4`: First dimension to apply penalty (default: 4, so penalties apply to dims 4,5,6,...)
"""
function ExtraDimensionPenaltyModifier(alpha::T; start_dim=4) where {T}
    ExtraDimensionPenaltyModifier{T}(alpha, start_dim)
end

# This modifier requires forces to be computed
requires_forces(::ExtraDimensionPenaltyModifier) = true

"""
    post_calculate!(mod::ExtraDimensionPenaltyModifier, cell, energy, forces, stress)

Add harmonic penalty to energy and forces for extra dimensions.
"""
function post_calculate!(
    mod::ExtraDimensionPenaltyModifier,
    cell::Cell,
    energy::T,
    forces::Matrix{T},
    stress::Matrix{T}
) where {T}
    pos = positions(cell)
    D = size(pos, 1)

    # Add penalty energy
    penalty_energy = zero(T)
    if D >= mod.start_dim
        # Sum of squared positions in extra dimensions
        for d in mod.start_dim:D
            for atom in 1:size(pos, 2)
                penalty_energy += pos[d, atom]^2
            end
        end
        penalty_energy *= mod.alpha / 2
    end

    # Modify forces for extra dimensions
    penalty_forces = copy(forces)
    if D >= mod.start_dim
        for d in mod.start_dim:D
            for atom in 1:size(pos, 2)
                penalty_forces[d, atom] = forces[d, atom] - 2 * mod.alpha * pos[d, atom]
            end
        end
    end

    return (energy + penalty_energy, penalty_forces, stress)
end

# Display
function Base.show(io::IO, mod::ExtraDimensionPenaltyModifier)
    print(io, "ExtraDimensionPenaltyModifier(alpha=$(mod.alpha), start_dim=$(mod.start_dim))")
end
