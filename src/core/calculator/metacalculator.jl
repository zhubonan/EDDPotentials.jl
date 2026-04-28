"""
    MetaCalculator{T,S<:AbstractCalc}

Calculator with modifier pipeline architecture.

The MetaCalculator delegates cell ownership to its source calculator and
composes an ordered list of modifiers on top.

# Fields
- `source::S`: The source calculator (e.g., NNCalc, EDDPCalculator)
- `modifiers::Vector`: Pipeline of modifiers (applied in order)
- `energy::T`: Cached energy result after modifier pipeline
- `forces::Matrix{T}`: Cached forces result after modifier pipeline
- `stress::Matrix{T}`: Cached stress result after modifier pipeline
- `energy_dirty::Bool`: Whether post-modifier energy needs recomputation
- `forces_dirty::Bool`: Whether post-modifier forces need recomputation

# Example
```julia
source = NNCalc(cell, cf, ensemble)
calc = MetaCalculator(source, [
    ExtraDimensionPenaltyModifier(1e-3),
    CellRelaxationModifier(cell; external_pressure=pressure_mat)
])
energy = eddp_energy(calc)
forces = eddp_forces(calc)
```
"""
mutable struct MetaCalculator{T,S<:AbstractCalc} <: AbstractCalc
    source::S
    modifiers::Vector
    energy::T
    forces::Matrix{T}
    stress::Matrix{T}
    energy_dirty::Bool
    forces_dirty::Bool
end

"""
    MetaCalculator(source::AbstractCalc, modifiers::Vector = Any[])

Create a MetaCalculator with a source calculator and optional modifiers.

The cell is obtained from `get_cell(source)`. Workspace buffers (forces,
stress) are sized to match the source cell.
"""
function MetaCalculator(
    source::AbstractCalc,
    modifiers::Vector=Any[]
)
    cell = get_cell(source)
    T = eltype(cellmat(cell))
    D = CellBase.n_dimensions(cell)

    MetaCalculator{T,typeof(source)}(
        source,
        modifiers,
        zero(T),
        zeros(T, D, length(cell)),
        zeros(T, 3, 3),
        true,
        true
    )
end

"""
    calculate!(calc::MetaCalculator; forces=true, rebuild_nl=true, kwargs...)

Main calculation entry point. Executes the modifier pipeline:
1. Check cache — return early if clean
2. Pre-calculation phase — apply `pre_calculate!` for all modifiers
3. Source calculation — run base calculator
4. Post-calculation phase — apply `post_calculate!` in pipeline order
5. Cache results

Caching relies on the dirty flags set by `set_positions!`/`set_cellmat!`.
The source calculator's own caching is transparent to MetaCalculator.
"""
function calculate!(calc::MetaCalculator; forces=true, rebuild_nl=true, kwargs...)
    if !calc.energy_dirty && !(forces && calc.forces_dirty)
        return calc
    end

    src_cell = get_cell(calc.source)

    for mod in calc.modifiers
        pre_calculate!(mod, src_cell)
    end

    needs_force_for_modifiers = any(requires_forces(mod) for mod in calc.modifiers)
    compute_forces = forces || needs_force_for_modifiers

    calculate!(calc.source; forces=compute_forces, rebuild_nl, kwargs...)

    energy = eddp_energy(calc.source)
    if compute_forces
        forces_result = eddp_forces(calc.source)
        stress_result = eddp_stress(calc.source)
    else
        forces_result = calc.forces
        stress_result = calc.stress
    end

    for mod in calc.modifiers
        if compute_forces || !requires_forces(mod)
            energy, forces_result, stress_result = post_calculate!(
                mod, src_cell, energy, forces_result, stress_result
            )
        end
    end

    calc.energy = energy
    if compute_forces
        calc.forces = forces_result
        calc.stress = stress_result
        calc.forces_dirty = false
    end
    calc.energy_dirty = false

    calc
end

# AbstractCalc interface — delegate to source

"""
    get_cell(calc::MetaCalculator)

Return the cell from the source calculator.
"""
get_cell(calc::MetaCalculator) = get_cell(calc.source)

"""
    eddp_energy(calc::MetaCalculator; kwargs...)

Compute energy through the modifier pipeline.
"""
function eddp_energy(calc::MetaCalculator; forces=false, kwargs...)
    calculate!(calc; forces=forces, kwargs...)
    calc.energy
end

"""
    eddp_forces(calc::MetaCalculator; kwargs...)

Compute forces through the modifier pipeline.
"""
function eddp_forces(calc::MetaCalculator; make_copy=true, kwargs...)
    calculate!(calc; forces=true, kwargs...)
    make_copy ? copy(calc.forces) : calc.forces
end

"""
    eddp_stress(calc::MetaCalculator; kwargs...)

Compute stress through the modifier pipeline.
"""
function eddp_stress(calc::MetaCalculator; make_copy=true, kwargs...)
    calculate!(calc; forces=true, kwargs...)
    make_copy ? copy(calc.stress) : calc.stress
end

"""
    eddp_enthalpy(calc::MetaCalculator; kwargs...)

Compute enthalpy through the modifier pipeline. When a `CellRelaxationModifier`
is present the cached energy already includes the PV term, so this is
equivalent to `eddp_energy`. Raw internal energy is available via
`eddp_energy(calc.source)`.
"""
function eddp_enthalpy(calc::MetaCalculator; kwargs...)
    eddp_energy(calc; kwargs...)
end

# Position accessors with modifier dispatch

"""
    get_positions(calc::MetaCalculator)

Get positions, potentially augmented by modifiers that change position semantics.
Delegates to the first modifier with `modifies_positions == true`, or falls back
to the source cell's positions.
"""
function get_positions(calc::MetaCalculator)
    src_cell = get_cell(calc.source)
    for mod in calc.modifiers
        if modifies_positions(mod)
            return get_augmented_positions(mod, src_cell)
        end
    end
    get_positions(src_cell)
end

"""
    set_positions!(calc::MetaCalculator, new)

Set positions, potentially delegating to modifiers that change position semantics.
Delegates to the first modifier with `modifies_positions == true`, or falls back
to setting the source cell's positions directly.
"""
function set_positions!(calc::MetaCalculator, new)
    src_cell = get_cell(calc.source)
    for mod in calc.modifiers
        if modifies_positions(mod)
            set_augmented_positions!(mod, src_cell, new)
            calc.energy_dirty = true
            calc.forces_dirty = true
            return
        end
    end
    set_positions!(src_cell, new)
    calc.energy_dirty = true
    calc.forces_dirty = true
end

"""
    set_cellmat!(calc::MetaCalculator, cellmat; kwargs...)

Set the cell matrix on the source calculator.
"""
function set_cellmat!(calc::MetaCalculator, cellmat; kwargs...)
    set_cellmat!(get_cell(calc.source), cellmat; kwargs...)
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
