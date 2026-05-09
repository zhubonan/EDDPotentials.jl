struct __NotSpecified__ end
const __not_specified__ = __NotSpecified__()

"""
    EDDPCalculator{M<:AbstractCellFeatureSpec, X<:AbstractNNInterface} <: AbstractCalc

A model-level calculator that bundles a trained EDDP model with its feature
specification and reference energies. Can be used without being bound to a
specific atomic structure, or bound to a cell for optimization workflows.

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
    core_repulsion::Any
    metadata::Dict{String,Any}

    cell::Any
    last_cellmat::Any
    last_positions::Any
    last_species::Any
    last_nn_build_pos::Any
    nl::Any
    v::Any
    gv::Any
    workspace::Any
    forces::Any
    stress::Any
    eng::Any
    _forces_calculated::Bool
    _energy_calculated::Bool
    _max_nat::Int
    _max_ndims::Int
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
    EDDPCalculator{M,X}(
        cf, nninterface, elemental_energies, shell_size, nmax, core_repulsion, metadata,
        nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing, nothing, nothing,
        false, false, 0, 0,
    )
end

function EDDPCalculator(
    cell::Cell{T},
    cf::M,
    nninterface::X;
    shell_size::Float64=2.0,
    nmax::Int=500,
    core_repulsion=CoreRepulsion(0.5),
    core=__not_specified__,
    elemental_energies::Dict=Dict{Symbol,T}(),
    metadata::Dict{String,Any}=Dict{String,Any}(),
) where {T,M<:AbstractCellFeatureSpec,X<:AbstractNNInterface}
    if core !== __not_specified__
        core_repulsion = core
    end
    ee = Dict{Symbol,T}(Symbol(k) => T(v) for (k, v) in elemental_energies)
    calc = EDDPCalculator(cf, nninterface, ee;
        shell_size, nmax, core_repulsion, metadata)
    _bind_cell!(calc, cell)
    calc
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

"""Internal: allocate workspace buffers for a given cell."""
function _bind_cell!(calc::EDDPCalculator, cell::Cell{T}) where {T}
    rcut = suggest_rcut(calc.cf; shell=calc.shell_size)
    nl = NeighbourList(cell, rcut, calc.nmax; savevec=true)
    v = zeros(T, nfeatures(calc.cf), length(cell))

    workspace = GradientWorkspace(
        v;
        ndims=CellBase._n_dimensions(cell),
        core=calc.core_repulsion,
        do_grad=true,
        one_body_offset=one_body_offset(calc.cf),
    )
    if calc.cf isa CellFeature
        _populate_feat_bufs!(workspace, calc.cf.two_body, calc.cf.three_body, nl)
    end

    calc.cell = cell
    calc.last_cellmat = copy(cellmat(cell))
    calc.last_positions = copy(positions(cell))
    calc.last_species = copy(species(cell))
    calc.last_nn_build_pos = sposarray(cell)
    calc.nl = nl
    calc.v = v
    calc.gv = similar(v)
    calc.workspace = workspace
    calc.forces = workspace.tot_forces
    calc.stress = workspace.tot_stress
    calc.eng = zeros(T, nions(cell))
    calc._forces_calculated = false
    calc._energy_calculated = false
    calc._max_nat = length(cell)
    calc._max_ndims = CellBase._n_dimensions(cell)
    calc
end

"""
    set_cell!(calc::EDDPCalculator, cell::Cell) -> calc

Attach a cell to the calculator. On first call the internal workspace is
allocated. On subsequent calls the existing workspace is reused when the
cell has the same number of atoms and spatial dimensions — only positions,
lattice, and species are updated in-place. Otherwise a new workspace is
allocated.

Once a cell is attached the single-argument `AbstractCalc` interface becomes
available: `eddp_energy(calc)`, `eddp_forces(calc)`, `opt_tpsd(calc)`, etc.
"""
function set_cell!(calc::EDDPCalculator, cell::Cell)
    if calc.cell === nothing
        _bind_cell!(calc, cell)
        return calc
    end

    ndims_new = CellBase._n_dimensions(cell)
    nat_new = length(cell)
    ndims_old = CellBase._n_dimensions(calc.cell)
    nat_old = length(calc.cell)

    if ndims_new == ndims_old && nat_new == nat_old
        set_cellmat!(calc.cell, cellmat(cell); scale_positions=false)
        set_positions!(calc.cell, positions(cell))
        CellBase.species(calc.cell) .= CellBase.species(cell)
        calc._energy_calculated = false
        calc._forces_calculated = false
    else
        _bind_cell!(calc, cell)
    end
    calc
end

# ---------------------------------------------------------------------------
# AbstractCalc interface (single-arg, requires bound cell)
# ---------------------------------------------------------------------------

function _assert_bound(calc::EDDPCalculator)
    calc.cell === nothing && error(
        "No cell attached — call set_cell!(calc, cell) first, " *
        "or use the two-argument form: eddp_energy(calc, cell)."
    )
end

get_cell(calc::EDDPCalculator) = (_assert_bound(calc); calc.cell)

function set_positions!(calc::EDDPCalculator, new)
    _assert_bound(calc)
    set_positions!(calc.cell, new)
end

function set_cellmat!(calc::EDDPCalculator, new; kwargs...)
    _assert_bound(calc)
    set_cellmat!(calc.cell, new; kwargs...)
end

# ---------------------------------------------------------------------------
# Core computation methods
# ---------------------------------------------------------------------------

function _need_calc(calc::EDDPCalculator, forces)
    if _cell_unchanged(calc.cell, calc.last_cellmat, calc.last_positions, calc.last_species) && calc._energy_calculated
        if forces
            calc._forces_calculated && return false
        else
            return false
        end
    end
    return true
end

function eddp_energy(calc::EDDPCalculator; forces=false, rebuild_nl=true)
    _assert_bound(calc)
    calculate!(calc; forces, rebuild_nl)
    sum(calc.eng) + get_elemental_energy(calc.cell, calc.elemental_energies)
end

function eddp_enthalpy(calc::EDDPCalculator; forces=false, rebuild_nl=true)
    eddp_energy(calc; forces, rebuild_nl)
end

function eddp_forces(calc::EDDPCalculator; rebuild_nl=true, make_copy=true)
    _assert_bound(calc)
    calculate!(calc; forces=true, rebuild_nl)
    if make_copy
        return copy(calc.forces)
    end
    calc.forces
end

function eddp_stress(calc::EDDPCalculator; rebuild_nl=true, make_copy=true)
    _assert_bound(calc)
    calculate!(calc; forces=true, rebuild_nl)
    if make_copy
        return copy(calc.stress)
    end
    calc.stress
end

function calculate!(calc::EDDPCalculator; forces=true, rebuild_nl=true)
    _assert_bound(calc)
    _need_calc(calc, forces) || return calc

    @timeit to "_rebuild" _rebuild_on_demand(calc; rebuild_nl)
    @timeit to "_calculate" _calculate!(calc, false, forces)

    calc.last_cellmat .= cellmat(calc.cell)
    copy!(calc.last_positions, positions(calc.cell))
    calc.last_species .= species(calc.cell)
    calc
end

function eddp_energy_std(calc::EDDPCalculator)
    _assert_bound(calc)
    calc.nninterface isa EnsembleNNInterface || error(
        "eddp_energy_std requires an EnsembleNNInterface, got $(typeof(calc.nninterface))"
    )
    eddp_energy(calc)
    per_atom = reduce(vcat, forward!.(calc.nninterface.models, Ref(calc.v)))
    std(sum(per_atom, dims=2))
end

function update_feature_vector!(calc::EDDPCalculator; rebuild_nl=true, compute_gradients=true)
    _assert_bound(calc)
    cell = calc.cell
    nl = calc.nl
    fill!(calc.v, 0)

    rebuild_nl ? rebuild!(nl, cell) : CellBase.update!(nl, cell)

    one_body_vectors!(calc.v, cell, calc.cf)
    n1bd = one_body_offset(calc.cf)
    if compute_gradients
        @timeit to "compute_fv_gv!" compute_fv_gv!(calc.workspace, calc.cf, cell; nl, offset=n1bd)
    else
        @timeit to "compute_fv!" compute_fv!(calc.workspace, calc.cf, cell; nl, offset=n1bd)
    end
    calc.v
end

function _rebuild_on_demand(calc::EDDPCalculator; rebuild_nl)
    cell = calc.cell
    pos = sposarray(cell)
    if rebuild_nl
        rebuild = true
    else
        tol = (calc.nl.rcut - suggest_rcut(calc.cf; shell=0.0))^2 / 2
        rebuild = false
        for i = 1:natoms(cell)
            if distance_squared_between(pos[i], calc.last_nn_build_pos[i]) > tol
                rebuild = true
                break
            end
        end
    end
    if rebuild
        calc.last_nn_build_pos .= pos
        @timeit to "rebuild!" rebuild!(calc.nl, calc.cell)
    else
        @timeit to "update!" CellBase.update!(calc.nl, calc.cell)
    end
end

function _calculate!(calc::EDDPCalculator, rebuild=true, compute_forces=true)
    @timeit to "update_feature_vector!" update_feature_vector!(
        calc;
        rebuild_nl=rebuild,
        compute_gradients=compute_forces,
    )
    calc.eng .= @timeit to "forward!" forward!(calc.nninterface, calc.v)[1, :]
    if calc.workspace.hardcore.core_repulsion !== nothing
        calc.eng .+= calc.workspace.hardcore.ecore
    end
    calc._energy_calculated = true
    calc._forces_calculated = false
    fill!(calc.stress, 0.0)
    fill!(calc.forces, 0.0)
    if compute_forces
        @timeit to "backward!" backward!(
            calc.nninterface;
            gu=one(eltype(calc.v)),
            weight_and_bias=false,
        )
        @timeit to "gradinp!" gradinp!(calc.gv, calc.nninterface)
        n1bd = one_body_offset(calc.cf)
        @timeit to "apply_chainrule!" force_via_chainrule!(calc, calc.gv, offset=n1bd)
        calc.stress ./= volume(get_cell(calc))
        calc._forces_calculated = true
    end
end

function force_via_chainrule!(calc::EDDPCalculator, gv; offset=calc.workspace.one_body_offset)
    _force_update!(calc.workspace, gv; offset)
    _stress_update!(calc.workspace, gv; offset)
    calc
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
    eddp_energy(calc; kwargs...)
end

"""
    eddp_forces(calc::EDDPCalculator, cell::Cell; kwargs...) -> Matrix

Compute atomic forces using the model in `calc` for the given `cell`.
"""
function eddp_forces(calc::EDDPCalculator, cell::Cell; kwargs...)
    set_cell!(calc, cell)
    eddp_forces(calc; kwargs...)
end

"""
    eddp_stress(calc::EDDPCalculator, cell::Cell; kwargs...) -> Matrix

Compute the 3×3 stress tensor using the model in `calc` for the given `cell`.
"""
function eddp_stress(calc::EDDPCalculator, cell::Cell; kwargs...)
    set_cell!(calc, cell)
    eddp_stress(calc; kwargs...)
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
    eddp_energy_std(calc)
end

"""
    eddp_energy_std_per_atom(calc::EDDPCalculator, cell::Cell; kwargs...)
"""
eddp_energy_std_per_atom(calc::EDDPCalculator, cell::Cell; kwargs...) =
    eddp_energy_std(calc, cell; kwargs...) / length(cell)

function eddp_energy_per_atom(calc::EDDPCalculator; kwargs...)
    _assert_bound(calc)
    eddp_energy(calc; kwargs...) / length(calc.cell)
end

function eddp_energy_std_per_atom(calc::EDDPCalculator; kwargs...)
    _assert_bound(calc)
    eddp_energy_std(calc; kwargs...) / length(calc.cell)
end

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

function Base.show(io::IO, calc::EDDPCalculator)
    print(io, "EDDPCalculator(")
    print(io, typeof(calc.nninterface).name.name)
    if calc.cell !== nothing
        print(io, ", bound to ", length(calc.cell), " atoms)")
    else
        print(io, ", no cell attached)")
    end
end

function Base.show(io::IO, ::MIME"text/plain", calc::EDDPCalculator)
    print(io, "EDDPCalculator with ")
    print(io, typeof(calc.nninterface).name.name)
    println(io)
    if calc.cell !== nothing
        print(io, "  Bound to cell with ")
        print(io, length(calc.cell))
        println(io, " atoms")
    else
        println(io, "  No cell attached — call set_cell!(calc, cell) to bind")
    end
end

"""
    predict(calc::EDDPCalculator, cell::Cell; forces=true, stress=true) -> NamedTuple

One-shot energy, forces, and stress prediction for a given `cell`.
Returns a `NamedTuple` with keys `:energy`, `:forces` (if `forces=true`),
and `:stress` (if `stress=true`).

More efficient than calling `eddp_energy`, `eddp_forces`, and `eddp_stress`
separately since the cell is bound only once.

# Example
```julia
calc = load_calculator("model.json")
result = predict(calc, cell)
result.energy     # total energy
result.forces     # atomic forces matrix
result.stress     # stress tensor (3×3)
```
"""
function predict(calc::EDDPCalculator, cell::Cell; forces::Bool=true, stress::Bool=true)
    set_cell!(calc, cell)
    result = (energy=eddp_energy(calc),)
    if forces
        result = merge(result, (forces=eddp_forces(calc),))
    end
    if stress
        result = merge(result, (stress=eddp_stress(calc),))
    end
    result
end

# ---------------------------------------------------------------------------
# Calculator factory — bridge to MetaCalculator
# ---------------------------------------------------------------------------

"""
    create_calculator(calc::EDDPCalculator, cell::Cell; modifiers=[]) -> AbstractCalc

Construct a calculator bound to `cell`. Returns a bound `EDDPCalculator` when
no modifiers are given, or a [`MetaCalculator`](@ref) with the modifier pipeline
when modifiers are present.

# Examples

```julia
calc = load_calculator("model.json")

# Simple fixed-cell relaxation
nnc = create_calculator(calc, cell)
opt_tpsd(nnc)

# Variable-cell relaxation
nnc = create_calculator(calc, cell)
opt_tpsd(nnc; relax_cell=true)

# Hyperdimensional relaxation
hcell = add_dimensions(cell, 3)
hc = create_calculator(calc, hcell; modifiers=[
    ExtraDimensionPenaltyModifier(1e-4),
])
scheduled_penalty_relax!(hc; relax_kwargs=(relax_cell=true,))
```
"""
function create_calculator(calc::EDDPCalculator, cell::Cell; modifiers=[])
    bound = EDDPCalculator(
        calc.cf, calc.nninterface, calc.elemental_energies;
        shell_size=calc.shell_size,
        nmax=calc.nmax,
        core_repulsion=calc.core_repulsion,
        metadata=calc.metadata,
    )
    _bind_cell!(bound, cell)
    isempty(modifiers) && return bound
    MetaCalculator(bound, modifiers)
end

function get_committee_forces(calc::EDDPCalculator)
    force_list = map(calc.nninterface.models) do model
        nc = EDDPCalculator(calc.cell, calc.cf, model; elemental_energies=calc.elemental_energies)
        eddp_forces(nc)
    end
end

function get_committee_ferror(calc::EDDPCalculator)
    flist = get_committee_forces(calc)
    fmean = reduce(+, flist) ./ length(flist)
    fmax = 0.0
    for i = 1:length(flist)
        maxdiff = maximum(map(norm, eachcol(flist[i] .- fmean)))
        if maxdiff > fmax
            fmax = maxdiff
        end
    end
    fmax
end
