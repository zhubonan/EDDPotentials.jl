"""
    MetaCalculator{T,S<:AbstractCalc}

Calculator with modifier pipeline architecture.

The MetaCalculator delegates cell ownership to its source calculator and
composes an ordered list of modifiers on top.

# Fields
- `source::S`: The source calculator (e.g., EDDPCalculator)
- `modifiers::Vector`: Pipeline of modifiers (applied in order)
- `energy::T`: Cached energy result after modifier pipeline
- `forces::Matrix{T}`: Cached forces result after modifier pipeline
- `stress::Matrix{T}`: Cached stress result after modifier pipeline
- `energy_dirty::Bool`: Whether post-modifier energy needs recomputation
- `forces_dirty::Bool`: Whether post-modifier forces need recomputation

# Example
```julia
source = EDDPCalculator(cell, cf, ensemble)
calc = MetaCalculator(source, [ExtraDimensionPenaltyModifier(1e-3)])
opt_tpsd(calc; relax_cell=true)
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
    last_cellmat::Matrix{T}
    last_positions::Matrix{T}
    last_species::Vector
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
        true,
        copy(cellmat(cell)),
        copy(positions(cell)),
        copy(species(cell))
    )
end

function _source_cell_changed(calc::MetaCalculator)
    cell = get_cell(calc.source)
    size(cellmat(cell)) == size(calc.last_cellmat) || return true
    size(positions(cell)) == size(calc.last_positions) || return true
    length(species(cell)) == length(calc.last_species) || return true
    return !_cell_unchanged(cell, calc.last_cellmat, calc.last_positions, calc.last_species)
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
    if _source_cell_changed(calc)
        calc.energy_dirty = true
        calc.forces_dirty = true
    end

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
    calc.last_cellmat = copy(cellmat(src_cell))
    calc.last_positions = copy(positions(src_cell))
    calc.last_species = copy(species(src_cell))

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

Return the enthalpy. Equivalent to `eddp_energy` for MetaCalculator.
Raw internal energy is available via `eddp_energy(calc.source)`.
"""
function eddp_enthalpy(calc::MetaCalculator; kwargs...)
    eddp_energy(calc; kwargs...)
end

"""
    eddp_energy_std(calc::MetaCalculator; kwargs...)

Compute ensemble energy standard deviation by delegating to the source calculator.
"""
function eddp_energy_std(calc::MetaCalculator; kwargs...)
    eddp_energy_std(calc.source; kwargs...)
end

"""
    eddp_energy_std_per_atom(calc::MetaCalculator; kwargs...)

Compute per-atom ensemble energy standard deviation by delegating to the source calculator.
"""
function eddp_energy_std_per_atom(calc::MetaCalculator; kwargs...)
    eddp_energy_std_per_atom(calc.source; kwargs...)
end

# Position accessors with modifier dispatch

"""
    get_positions(calc::MetaCalculator)

Get positions, potentially augmented by modifiers that change position semantics.
Delegates to the first modifier with `modifies_positions == true`, or falls back
to the source cell's positions.
"""
function get_positions(calc::MetaCalculator)
    get_positions(get_cell(calc.source))
end

function set_positions!(calc::MetaCalculator, new)
    set_positions!(get_cell(calc.source), new)
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
    calc.energy_dirty = true
    calc.forces_dirty = true
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
