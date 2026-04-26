"""
    EDDPCalculator{M<:AbstractCellFeatureSpec, X<:AbstractNNInterface} <: AbstractCalc

A model-level calculator that bundles a trained EDDP model with its feature
specification and reference energies. Unlike [`NNCalc`](@ref), this type can
be used without being bound to a specific atomic structure.

# Loading

```julia
calc = load_calculator("model.json")
```

# Prediction (two-argument form)

```julia
energy = eddp_energy(calc, cell)
forces = eddp_forces(calc, cell)
stress = eddp_stress(calc, cell)
```

# Calculator mode (attach a cell, use with optimizers)

```julia
set_cell!(calc, cell)
eddp_energy(calc)
opt_tpsd(calc)
```
"""
mutable struct EDDPCalculator{M<:AbstractCellFeatureSpec,X<:AbstractNNInterface} <: AbstractCalc
    cf::M
    nninterface::X
    elemental_energies::Dict{Symbol,Float64}
    shell_size::Float64
    nmax::Int
    core_repulsion::Any  # Union{Nothing, CoreRepulsion}
    metadata::Dict{String,Any}
    _nncalc::Any
end

function EDDPCalculator(
    cf::M,
    nninterface::X,
    elemental_energies::Dict{Symbol,Float64};
    shell_size::Float64=2.0,
    nmax::Int=500,
    core_repulsion=CoreRepulsion(0.5),
    metadata::Dict{String,Any}=Dict{String,Any}(),
) where {M<:AbstractCellFeatureSpec,X<:AbstractNNInterface}
    EDDPCalculator{M,X}(cf, nninterface, elemental_energies, shell_size, nmax, core_repulsion, metadata, nothing)
end

"""
    EDDPCalculator(artifact::ModelArtifact; kwargs...) -> EDDPCalculator

Construct a calculator from a pre-loaded [`ModelArtifact`](@ref).
"""
function EDDPCalculator(artifact::ModelArtifact; kwargs...)
    EDDPCalculator(artifact.feature_spec, artifact.model, artifact.elemental_energies; kwargs...)
end

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

"""
    load_calculator(path::AbstractString; kwargs...) -> EDDPCalculator

Load a trained EDDP model from a JSON or JLD2 artifact file and return a
calculator ready for inference.

# Keyword arguments
- `shell_size=2.0`: Extra shell for neighbour list skin region.
- `nmax=500`: Maximum neighbour list entries per atom.
- `core_repulsion=CoreRepulsion(0.5)`: Repulsive core potential (pass `nothing` to disable).

# Example
```julia
calc = load_calculator("model.json")
energy = eddp_energy(calc, cell)
```
"""
function load_calculator(
    path::AbstractString;
    shell_size::Float64=2.0,
    nmax::Int=500,
    core_repulsion=CoreRepulsion(0.5),
)
    artifact = load_model_artifact(path)
    EDDPCalculator(artifact; shell_size, nmax, core_repulsion)
end

# ---------------------------------------------------------------------------
# Cell binding
# ---------------------------------------------------------------------------

"""
    set_cell!(calc::EDDPCalculator, cell::Cell) -> calc

Attach a cell to the calculator. On first call an internal [`NNCalc`](@ref)
is created. On subsequent calls the existing workspace is reused when the
cell has the same number of atoms and spatial dimensions — only positions,
lattice, and species are updated in-place. Otherwise a new [`NNCalc`](@ref)
is allocated.

Once a cell is attached the single-argument `AbstractCalc` interface becomes
available: `eddp_energy(calc)`, `eddp_forces(calc)`, `opt_tpsd(calc)`, etc.
"""
function set_cell!(calc::EDDPCalculator, cell::Cell)
    nnc = calc._nncalc
    if nnc === nothing
        calc._nncalc = _new_nncalc(calc, cell)
        return calc
    end

    # Reuse the existing workspace when dimensions and atom count match.
    # This is the common case during geometry optimisation (same atoms,
    # positions change) and avoids costly reallocation.
    ndims_new = CellBase._n_dimensions(cell)
    ndims_old = CellBase._n_dimensions(get_cell(nnc))
    nat_old = length(get_cell(nnc))
    nat_new = length(cell)

    if ndims_new == ndims_old && nat_old == nat_new
        # In-place update — copy cell data and reset dirty flags
        set_cellmat!(nnc.cell, cellmat(cell); scale_positions=false)
        set_positions!(nnc.cell, positions(cell))
        CellBase.species(nnc.cell) .= CellBase.species(cell)
        nnc.param.energy_calculated = false
        nnc.param.forces_calculated = false
    else
        calc._nncalc = _new_nncalc(calc, cell)
    end
    calc
end

"""Internal helper to construct an NNCalc from the EDDPCalculator's model."""
function _new_nncalc(calc::EDDPCalculator, cell::Cell)
    NNCalc(
        cell, calc.cf, calc.nninterface;
        shell_size=calc.shell_size,
        nmax=calc.nmax,
        core=calc.core_repulsion,
        elemental_energies=calc.elemental_energies,
    )
end

"""
    extract_nncalc(calc::EDDPCalculator) -> NNCalc

Return the underlying [`NNCalc`](@ref) from a bound calculator. Throws an
error if no cell has been attached via [`set_cell!`](@ref).
"""
function extract_nncalc(calc::EDDPCalculator)
    calc._nncalc === nothing && error(
        "No cell attached — call set_cell!(calc, cell) first."
    )
    calc._nncalc
end

# ---------------------------------------------------------------------------
# AbstractCalc interface (single-arg, requires bound cell)
# ---------------------------------------------------------------------------

function _assert_bound(calc::EDDPCalculator)
    calc._nncalc === nothing && error(
        "No cell attached — call set_cell!(calc, cell) first, " *
        "or use the two-argument form: eddp_energy(calc, cell)."
    )
end

get_cell(calc::EDDPCalculator) = (_assert_bound(calc); get_cell(calc._nncalc))
calculate!(calc::EDDPCalculator; kwargs...) = (_assert_bound(calc); calculate!(calc._nncalc; kwargs...))

function eddp_energy(calc::EDDPCalculator; kwargs...)
    _assert_bound(calc)
    eddp_energy(calc._nncalc; kwargs...)
end

function eddp_forces(calc::EDDPCalculator; kwargs...)
    _assert_bound(calc)
    eddp_forces(calc._nncalc; kwargs...)
end

function eddp_stress(calc::EDDPCalculator; kwargs...)
    _assert_bound(calc)
    eddp_stress(calc._nncalc; kwargs...)
end

function eddp_energy_std(calc::EDDPCalculator)
    _assert_bound(calc)
    eddp_energy_std(calc._nncalc)
end

function set_positions!(calc::EDDPCalculator, new)
    _assert_bound(calc)
    set_positions!(calc._nncalc, new)
end

function set_cellmat!(calc::EDDPCalculator, new; kwargs...)
    _assert_bound(calc)
    set_cellmat!(calc._nncalc, new; kwargs...)
end

function eddp_enthalpy(calc::EDDPCalculator; kwargs...)
    _assert_bound(calc)
    eddp_enthalpy(calc._nncalc; kwargs...)
end

function eddp_energy_per_atom(calc::EDDPCalculator; kwargs...)
    _assert_bound(calc)
    eddp_energy_per_atom(calc._nncalc; kwargs...)
end

function eddp_energy_std_per_atom(calc::EDDPCalculator; kwargs...)
    _assert_bound(calc)
    eddp_energy_std_per_atom(calc._nncalc; kwargs...)
end

# ---------------------------------------------------------------------------
# Two-argument prediction (no persistent binding)
# ---------------------------------------------------------------------------

"""
    eddp_energy(calc::EDDPCalculator, cell::Cell; kwargs...) -> Number

Compute the total energy using the model in `calc` for the given `cell`.
"""
function eddp_energy(calc::EDDPCalculator, cell::Cell; kwargs...)
    set_cell!(calc, cell)
    eddp_energy(calc._nncalc; kwargs...)
end

"""
    eddp_forces(calc::EDDPCalculator, cell::Cell; kwargs...) -> Matrix

Compute atomic forces using the model in `calc` for the given `cell`.
"""
function eddp_forces(calc::EDDPCalculator, cell::Cell; kwargs...)
    set_cell!(calc, cell)
    eddp_forces(calc._nncalc; kwargs...)
end

"""
    eddp_stress(calc::EDDPCalculator, cell::Cell; kwargs...) -> Matrix

Compute the 3×3 stress tensor using the model in `calc` for the given `cell`.
"""
function eddp_stress(calc::EDDPCalculator, cell::Cell; kwargs...)
    set_cell!(calc, cell)
    eddp_stress(calc._nncalc; kwargs...)
end

"""
    eddp_energy_per_atom(calc::EDDPCalculator, cell::Cell; kwargs...) -> Number
"""
eddp_energy_per_atom(calc::EDDPCalculator, cell::Cell; kwargs...) =
    eddp_energy(calc, cell; kwargs...) / length(cell)

"""
    eddp_energy_std(calc::EDDPCalculator, cell::Cell) -> Number
"""
function eddp_energy_std(calc::EDDPCalculator, cell::Cell)
    set_cell!(calc, cell)
    eddp_energy_std(calc._nncalc)
end

"""
    eddp_energy_std_per_atom(calc::EDDPCalculator, cell::Cell; kwargs...)
"""
eddp_energy_std_per_atom(calc::EDDPCalculator, cell::Cell; kwargs...) =
    eddp_energy_std(calc, cell; kwargs...) / length(cell)

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

function Base.show(io::IO, calc::EDDPCalculator)
    print(io, "EDDPCalculator(")
    print(io, typeof(calc.nninterface).name.name)
    if calc._nncalc !== nothing
        print(io, ", bound to ", length(get_cell(calc._nncalc)), " atoms)")
    else
        print(io, ", no cell attached)")
    end
end

function Base.show(io::IO, ::MIME"text/plain", calc::EDDPCalculator)
    print(io, "EDDPCalculator with ")
    print(io, typeof(calc.nninterface).name.name)
    println(io)
    if calc._nncalc !== nothing
        print(io, "  Bound to cell with ")
        print(io, length(get_cell(calc._nncalc)))
        println(io, " atoms")
    else
        println(io, "  No cell attached — call set_cell!(calc, cell) to bind")
    end
end
