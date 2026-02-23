"""
    AdditiveCalculatorModifier{C<:AbstractCalc}

Adds contributions from another calculator to the results.

This modifier is useful for combining different potential contributions,
such as adding a repulsive core potential to an EDDP model, or combining
pair potentials with machine learning potentials.

# Fields
- `calc::C`: The additional calculator whose contributions will be added

# Example
```julia
# Combine EDDP with pair potential
nn_calc = NNCalc(cell, cf, model)
pair_calc = PairpotCalc(cell, repulsion_workspace)

# Create modifier that adds pair potential
modifier = AdditiveCalculatorModifier(pair_calc)

# Use in MetaCalculator
calc = MetaCalculator(nn_calc, [modifier])

# Energy will be: E_total = E_nn + E_pair
# Forces will be: F_total = F_nn + F_pair
```
"""
mutable struct AdditiveCalculatorModifier{C<:AbstractCalc} <: AbstractModifier
    calc::C
end

# This modifier requires forces to be computed
requires_forces(::AdditiveCalculatorModifier) = true

"""
    post_calculate!(mod::AdditiveCalculatorModifier, cell, energy, forces, stress)

Add contributions from the wrapped calculator.

Note: This assumes the wrapped calculator shares the same cell reference,
or that its cell has been updated appropriately before this is called.
"""
function post_calculate!(
    mod::AdditiveCalculatorModifier,
    cell::Cell,
    energy::T,
    forces::Matrix{T},
    stress::Matrix{T}
) where {T}
    # Compute additional contributions
    # Note: rebuild_nl=false since neighbor list should already be updated
    add_energy = eddp_energy(mod.calc; rebuild_nl=false)
    add_forces = eddp_forces(mod.calc; rebuild_nl=false)
    add_stress = eddp_stress(mod.calc; rebuild_nl=false)

    return (
        energy + add_energy,
        forces .+ add_forces,
        stress .+ add_stress
    )
end

# Display
function Base.show(io::IO, mod::AdditiveCalculatorModifier)
    calc_type = typeof(mod.calc).name.name
    print(io, "AdditiveCalculatorModifier(", calc_type, ")")
end
