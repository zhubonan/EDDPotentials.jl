"""
    MetaCalculator{T,D,S<:AbstractCalc}

Calculator with modifier pipeline architecture.

The MetaCalculator has:
- A single source calculator (typically NNCalc)
- An ordered list of modifiers that process results
- Unified caching and dirty-flag system

# Fields
- `source::S`: The source calculator (e.g., NNCalc)
- `cell::Cell{T,D}`: The cell (single source of truth)
- `last_cell::Cell{T,D}`: Last computed cell state for caching
- `modifiers::Vector{AbstractModifier}`: Pipeline of modifiers (applied in order)
- `energy::T`: Cached energy result
- `forces::Matrix{T}`: Cached forces result
- `stress::Matrix{T}`: Cached stress result
- `energy_dirty::Bool`: Whether energy needs recomputation
- `forces_dirty::Bool`: Whether forces need recomputation

# Example
```julia
source = NNCalc(cell, cf, ensemble)
modifiers = [
    ExtraDimensionPenaltyModifier(1e-3),
    CellRelaxationModifier(cell; external_pressure=pressure_mat)
]
calc = MetaCalculator(source, modifiers)

# Use like any AbstractCalc
energy = eddp_energy(calc)
forces = eddp_forces(calc)
```
"""
mutable struct MetaCalculator{T,D,S<:AbstractCalc} <: AbstractCalc
    # The source calculator (typically NNCalc)
    source::S

    # The cell (single source of truth)
    cell::Cell{T,D}

    # Last computed cell state for caching
    last_cell::Cell{T,D}

    # Pipeline of modifiers (applied in order)
    modifiers::Vector

    # Cached results
    energy::T
    forces::Matrix{T}
    stress::Matrix{T}

    # Dirty flags
    energy_dirty::Bool
    forces_dirty::Bool
end

"""
    MetaCalculator(source::AbstractCalc, modifiers::Vector{<:AbstractModifier}=AbstractModifier[])

Create a MetaCalculator with a source calculator and optional modifiers.
"""
function MetaCalculator(
    source::AbstractCalc,
    modifiers::Vector=Any[]
)
    cell = get_cell(source)
    T = eltype(cellmat(cell))
    D = CellBase.n_dimensions(cell)

    MetaCalculator(
        source,
        cell,
        deepcopy(cell),
        modifiers,
        zero(T),
        zeros(T, D, length(cell)),
        zeros(T, 3, 3),
        true,
        true
    )
end

"""
    is_cell_equal(cell1::Cell, cell2::Cell) -> Bool

Check if two cells are equal (same positions and lattice).
"""
function is_cell_equal(cell1::Cell, cell2::Cell)
    # Check if lattices are equal
    if get_cellmat(cell1) != get_cellmat(cell2)
        return false
    end

    # Check if positions are equal
    pos1 = get_positions(cell1)
    pos2 = get_positions(cell2)

    if size(pos1) != size(pos2)
        return false
    end

    return pos1 ≈ pos2
end

"""
    calculate!(calc::MetaCalculator; forces=true, rebuild_nl=true, kwargs...)

Main calculation entry point. Executes the modifier pipeline:
1. Check cache - return early if clean and cell unchanged
2. Pre-calculation phase - apply pre_calculate!() for all modifiers
3. Source calculation - run base calculator
4. Post-calculation phase - apply post_calculate!() in pipeline order
5. Cache results
"""
function calculate!(calc::MetaCalculator; forces=true, rebuild_nl=true, kwargs...)
    # Check if calculation is needed
    if !calc.energy_dirty && !(forces && calc.forces_dirty)
        if is_cell_equal(calc.cell, calc.last_cell)
            return calc
        end
    end

    # Update source calculator's cell reference to match metacalculator's cell
    # (in case it has been modified externally)
    if get_cell(calc.source) !== calc.cell
        # Synchronize cell changes to source
        set_cellmat!(calc.source, get_cellmat(calc.cell); scale_positions=false)
        set_positions!(calc.source, get_positions(calc.cell))
    end

    # Apply pre-calculation modifiers
    for mod in calc.modifiers
        pre_calculate!(mod, calc.cell)
    end

    # Run source calculator
    calculate!(calc.source; forces, rebuild_nl, kwargs...)

    # Get source results
    energy = eddp_energy(calc.source)
    if forces
        forces_result = eddp_forces(calc.source)
        stress_result = eddp_stress(calc.source)
    else
        forces_result = calc.forces
        stress_result = calc.stress
    end

    # Apply post-calculation modifiers in pipeline order
    for mod in calc.modifiers
        if forces || !requires_forces(mod)
            energy, forces_result, stress_result = post_calculate!(
                mod, calc.cell, energy, forces_result, stress_result
            )
        end
    end

    # Cache results
    calc.energy = energy
    if forces
        calc.forces = forces_result
        calc.stress = stress_result
        calc.forces_dirty = false
    end
    calc.energy_dirty = false
    calc.last_cell = deepcopy(calc.cell)

    calc
end

# Implement AbstractCalc interface

"""
    get_cell(calc::MetaCalculator)

Get the cell from the metacalculator.
"""
get_cell(calc::MetaCalculator) = calc.cell

"""
    eddp_energy(calc::MetaCalculator; kwargs...)

Compute energy through the modifier pipeline.
"""
function eddp_energy(calc::MetaCalculator; kwargs...)
    calculate!(calc; forces=false, kwargs...)
    calc.energy
end

"""
    eddp_forces(calc::MetaCalculator; kwargs...)

Compute forces through the modifier pipeline.
"""
function eddp_forces(calc::MetaCalculator; kwargs...)
    calculate!(calc; forces=true, kwargs...)
    copy(calc.forces)
end

"""
    eddp_stress(calc::MetaCalculator; kwargs...)

Compute stress through the modifier pipeline.
"""
function eddp_stress(calc::MetaCalculator; kwargs...)
    calculate!(calc; forces=true, kwargs...)
    copy(calc.stress)
end

# Position accessors with modifier dispatch

"""
    get_positions(calc::MetaCalculator)

Get positions, potentially modified by modifiers that change position semantics.
"""
function get_positions(calc::MetaCalculator)
    # Check if any modifier changes position semantics
    for mod in calc.modifiers
        if modifies_positions(mod)
            # Delegate to the modifier's custom accessor
            return get_augmented_positions(mod, calc.cell)
        end
    end
    # Default: return cell positions
    get_positions(calc.cell)
end

"""
    set_positions!(calc::MetaCalculator, new)

Set positions, potentially delegating to modifiers that change position semantics.
"""
function set_positions!(calc::MetaCalculator, new)
    # Find modifiers that change position semantics
    for mod in calc.modifiers
        if modifies_positions(mod)
            set_augmented_positions!(mod, calc.cell, new)
            calc.energy_dirty = true
            calc.forces_dirty = true
            return
        end
    end
    # Default: set cell positions
    set_positions!(calc.cell, new)
    calc.energy_dirty = true
    calc.forces_dirty = true
end

"""
    set_cellmat!(calc::MetaCalculator, cellmat; kwargs...)

Set the cell matrix.
"""
function set_cellmat!(calc::MetaCalculator, cellmat; kwargs...)
    set_cellmat!(calc.cell, cellmat; kwargs...)
    calc.energy_dirty = true
    calc.forces_dirty = true
end

# Modifier query/parameter access

"""
    get_modifier(calc::MetaCalculator, type::Type{<:AbstractModifier})

Get a modifier of the specified type from the pipeline.
Returns `nothing` if no modifier of that type is found.
"""
function get_modifier(calc::MetaCalculator, type::Type{<:AbstractModifier})
    idx = findfirst(m -> isa(m, type), calc.modifiers)
    isnothing(idx) ? nothing : calc.modifiers[idx]
end

"""
    get_modifier_parameter(calc::MetaCalculator, type::Type{<:AbstractModifier}, name::Symbol)

Get a parameter value from a modifier by type and parameter name.
Returns `nothing` if the modifier is not found.
"""
function get_modifier_parameter(calc::MetaCalculator, type::Type{<:AbstractModifier}, name::Symbol)
    mod = get_modifier(calc, type)
    isnothing(mod) ? nothing : get_parameter(mod, name)
end

"""
    set_modifier_parameter!(calc::MetaCalculator, type::Type{<:AbstractModifier}, name::Symbol, value)

Set a parameter value on a modifier by type and parameter name.
Throws an error if the modifier is not found.
"""
function set_modifier_parameter!(calc::MetaCalculator, type::Type{<:AbstractModifier}, name::Symbol, value)
    mod = get_modifier(calc, type)
    isnothing(mod) && throw(ArgumentError("No modifier of type $type found in calculator pipeline"))
    set_parameter!(mod, name, value)
end

# Display

"""
    Base.show(io::IO, calc::MetaCalculator)

Display the MetaCalculator with its source and modifier pipeline.
"""
function Base.show(io::IO, calc::MetaCalculator)
    println(io, "MetaCalculator")
    println(io, "  Source: ", typeof(calc.source).name.name)
    if isempty(calc.modifiers)
        println(io, "  Pipeline: (empty)")
    else
        println(io, "  Pipeline:")
        for (i, mod) in enumerate(calc.modifiers)
            print(io, "    $i. ")
            show(io, mod)
            println(io)
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", calc::MetaCalculator)
    show(io, calc)
end
