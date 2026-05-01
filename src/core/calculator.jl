import Optim
import Base
using Optim: LBFGS, optimize
using LineSearches: HagerZhang
using LinearAlgebra
import CellBase:
    set_cellmat!,
    set_positions!,
    get_cellmat,
    get_positions,
    add_dimensions,
    remove_dimensions,
    distance_squared_between
using StatsBase: ZScoreTransform, transform!
"""
    AbstractCalc

Abstract supertype for all EDDP calculators. Concrete subtypes must implement:
- `get_cell(calc)` — return the `Cell` object
- `calculate!(calc; forces=true, rebuild_nl=true)` — trigger energy/force/stress computation
- `eddp_energy(calc)` — total energy
- `eddp_forces(calc)` — atomic forces matrix
- `eddp_stress(calc)` — 3×3 stress tensor

See [`NNCalc`](@ref), [`MetaCalculator`](@ref).
"""
abstract type AbstractCalc end
const AC = AbstractCalc

## Methods to be implemented for the Abstract type

"""
    get_cell(calc::AbstractCalc) -> Cell

Return the `Cell` object associated with the calculator.
"""
function get_cell end

"""
    eddp_energy(calc::AbstractCalc) -> Number

Return the total energy (eV) of the system.
"""
function eddp_energy end

"""
    calculate!(calc::AbstractCalc; forces=true, rebuild_nl=true)

Compute energy, forces, and stress. Results are cached internally — subsequent
calls with an unchanged cell are no-ops unless `forces=true` is requested after
an energy-only call. Set `rebuild_nl=false` to skip neighbor list rebuild and
instead update incrementally.
"""
function calculate! end

"""
    eddp_forces(calc::AbstractCalc) -> Matrix

Return the atomic forces as a `D×N` matrix where `D` is the number of spatial
dimensions and `N` is the number of atoms. Forces are in eV/Å.
"""
function eddp_forces end

"""
    eddp_stress(calc::AbstractCalc) -> Matrix

Return the 3×3 stress tensor (eV/Å³). The stress is the virial stress divided
by the cell volume.
"""
function eddp_stress end


## Base default implementation 

function get_cellmat(ac::AC)
    get_cellmat(get_cell(ac))
end

get_positions(ac::AC) = get_positions(get_cell(ac))

function set_cellmat!(calc::AC, cellmat; kwargs...)
    set_cellmat!(get_cell(calc), cellmat; kwargs...)
end


function set_positions!(calc::AC, positions)
    set_positions!(get_cell(calc), positions)
end

function Base.show(io::IO, c::AbstractCalc)
    println(io, "$(typeof(c).name.name) for: ")
    Base.show(io, get_cell(c))
end

function Base.show(io::IO, m::MIME"text/plain", cw::AbstractCalc)
    println(io, "$(typeof(cw).name.name) for: ")
    Base.show(io, m, get_cell(cw))
end

### Concrete implementation

@with_kw mutable struct NNCalcParam
    forces_calculated::Bool = false
    energy_calculated::Bool = false
    mode::String = "one-pass"
end

"""
    NNCalc(cell::Cell, cf::CellFeature, nn::AbstractNNInterface; kwargs...)

Primary calculator for EDDP energy/force/stress evaluation.

`NNCalc` binds a structure (`cell`), feature definition (`cf`), and trained
model interface (`nn`) into a reusable calculator object.
"""
mutable struct NNCalc{T,N<:NeighbourList,M<:AbstractCellFeatureSpec,X<:AbstractNNInterface,D} <:
               AbstractCalc
    cell::Cell{T,D}
    "Cached cell matrix for change detection"
    last_cellmat::Matrix{T}
    "Cached positions for change detection"
    last_positions::Matrix{T}
    "Cached species for change detection"
    last_species::Vector{Symbol}
    "Positions when the NeighbourList is built"
    last_nn_build_pos::Vector{SVector{D,T}}
    "NeighbourList"
    nl::N
    cf::M
    "Combined Feature Vector"
    v::Matrix{T}
    "Gradient of the feature vector"
    gv::Matrix{T}
    "Workspace"
    workspace::GradientWorkspace{T}
    "Forces"
    forces::Matrix{T}
    "Stress"
    stress::Matrix{T}
    "atomic_energy"
    eng::Vector{T}
    param::NNCalcParam
    "NNInterface"
    nninterface::X
    "elemental_energies"
    elemental_energies::Dict{Symbol,T}
end

get_cell(ac::NNCalc) = ac.cell

# """
# Copy the lattice and positions from one cell to the other
# """
# function copycell!(cell_from::Cell, cell_to::Cell)
#     set_cellmat!(cell_to, cellmat(cell_from))
#     set_positions!(cell_to, positions(cell_from))
#     species(cell_to) .= species(cell_from)
# end

"""
    _cell_unchanged(cell, last_cellmat, last_positions, last_species)

Check if the cell matches the cached state (cell matrix, positions, species).
"""
function _cell_unchanged(cell, last_cellmat, last_positions, last_species)
    all(cellmat(cell) .== last_cellmat) || return false
    all(positions(cell) .== last_positions) || return false
    all(species(cell) .== last_species) || return false
    return true
end

_make_symbol_keys(x) = x
function _make_symbol_keys(dict::Dict{T,K}) where {T,K}
    Dict{Symbol,K}(Symbol(key) => _make_symbol_keys(value) for (key, value) in pairs(dict))
end

function NNCalc(
    cell::Cell{T},
    cf::AbstractCellFeatureSpec,
    nn::AbstractNNInterface;
    shell_size=2.0,
    rcut=suggest_rcut(cf; shell=shell_size),
    mode="one-pass",
    nmax=500,
    savevec=true,
    core=CoreRepulsion(0.5),
    elemental_energies=Dict{Symbol, T}(),
) where {T}
    nl = NeighbourList(cell, rcut, nmax; savevec)
    v = zeros(T, nfeatures(cf), length(cell))

    workspace = GradientWorkspace(
        v;
        ndims=CellBase._n_dimensions(cell),
        core,
        do_grad=true,
        one_body_offset=one_body_offset(cf),
    ) # Buffer for force calculation
    _populate_feat_bufs!(workspace, cf.two_body, cf.three_body, nl)
    NNCalc(
        cell,
        copy(cellmat(cell)),          # last_cellmat
        copy(positions(cell)),        # last_positions
        copy(species(cell)),          # last_species
        sposarray(cell),              # last_nn_build_pos
        nl,
        cf,
        v,
        similar(v),  # Gradient of the input to the NN 
        workspace,
        workspace.tot_forces,  # Forces
        workspace.tot_stress,  # Stress
        zeros(T, nions(cell)), # Energy
        NNCalcParam(; mode=mode),
        nn,
        Dict{Symbol, T}(_make_symbol_keys(elemental_energies)),
    )
end

"""
    eddp_energy(calc::NNCalc; forces=false, rebuild_nl=true)

Return the total energy (eV) including elemental reference energies.
Triggers `calculate!` if the cell has changed since the last call.
"""
function eddp_energy(calc::NNCalc; forces=false, rebuild_nl=true)
    calculate!(calc; forces, rebuild_nl)
    sum(calc.eng) + get_elemental_energy(calc.cell, calc.elemental_energies)
end

get_elemental_energy(cell::Cell, ref_energies::Dict) =
    sum(x -> get(ref_energies, x, 0.0), species(cell))

"""
    eddp_enthalpy(calc::AbstractCalc) -> Number

Return the enthalpy `H = E + PV`. For calculators with a `CellRelaxationModifier`
this includes the external pressure contribution. For plain calculators, returns the energy.
"""
function eddp_enthalpy(calc::NNCalc; forces=false, rebuild_nl=true)
    eddp_energy(calc; forces, rebuild_nl)
end

function eddp_enthalpy(calc::AbstractCalc; kwargs...)
    eddp_energy(calc; kwargs...)
end

"""
    eddp_forces(calc::NNCalc; rebuild_nl=true, make_copy=true) -> Matrix

Return the `D×N` forces matrix (eV/Å). Set `make_copy=false` to get a
reference to the internally cached matrix instead of a copy.
"""
function eddp_forces(calc::NNCalc; rebuild_nl=true, make_copy=true)
    calculate!(calc; forces=true, rebuild_nl)
    if make_copy
        return copy(calc.forces)
    end
    calc.forces
end

"""
    eddp_stress(calc::NNCalc; rebuild_nl=true, make_copy=true) -> Matrix

Return the 3×3 virial stress tensor (eV/Å³), divided by cell volume.
Set `make_copy=false` to get a reference to the internally cached matrix.
"""
function eddp_stress(calc::NNCalc; rebuild_nl=true, make_copy=true)
    calculate!(calc; forces=true, rebuild_nl)
    if make_copy
        return copy(calc.stress)
    end
    calc.stress
end

function _need_calc(calc::NNCalc, forces)
    if _cell_unchanged(get_cell(calc), calc.last_cellmat, calc.last_positions, calc.last_species) && calc.param.energy_calculated
        if forces
            calc.param.forces_calculated && return false
        else
            return false
        end
    end
    return true
end

"""
    calculate!(calc::NNCalc; forces=true, rebuild_nl=true)

Core computation function for energy, forces, and stress. Results are cached —
subsequent calls with an unchanged cell are no-ops. Set `rebuild_nl=false` to
use incremental neighbor list updates instead of full rebuilds.
"""
@timeit to function calculate!(calc::NNCalc; forces=true, rebuild_nl=true)
    # Nothing to do if the cell has not changed since last time

    _need_calc(calc, forces) || return

    @timeit to "_rebuild" _rebuild_on_demand(calc; rebuild_nl)
    # Disable rebuilding NL - we have done this already
    @timeit to "_calculate" _calculate!(calc, false, forces)
    # Update as the last calculated cell
    calc.last_cellmat .= cellmat(calc.cell)
    copy!(calc.last_positions, positions(calc.cell))
    calc.last_species .= species(calc.cell)
    calc
end

"""
    _rebuild_on_demand(calc; rebuild_nl)

Trigger rebuild of the NeighbourList
"""
function _rebuild_on_demand(calc; rebuild_nl)
    cell = get_cell(calc)
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

"""
    eddp_energy_std(calc::NNCalc{<:Any,<:Any,<:Any,<:EnsembleNNInterface}) -> Number

Return the standard deviation of the predicted total energy across all models
in the ensemble. Requires the calculator to use an [`EnsembleNNInterface`](@ref).
Call [`eddp_energy`](@ref) first if the cell has changed.
"""
function eddp_energy_std(calc::NNCalc{T,N,M,X}) where {T,N,M,X<:EnsembleNNInterface}
    eddp_energy(calc)
    per_atom = reduce(vcat, forward!.(calc.nninterface.models, Ref(calc.v)))
    std(sum(per_atom, dims=2))
end

function get_committee_forces(calc)
    force_list = map(calc.nninterface.models) do model
        # TODO - allow others to be customised
        calc = NNCalc(calc.cell, calc.cf, model; elemental_energies=calc.elemental_energies)
        eddp_forces(calc)
    end
end


function get_committee_ferror(calc)
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


"""
    _calculate!(calc, rebuild, forces=true)

Internal function to compute energy, forces and stress.
"""
function _calculate!(calc::NNCalc, rebuild=true, compute_forces=true)
    # Compute feature vector and the gradients
    @timeit to "update_feature_vector!" update_feature_vector!(
        calc;
        rebuild_nl=rebuild,
        compute_gradients=compute_forces,
    )
    # Energy evaluation
    calc.eng .= @timeit to "forward!" forward!(calc.nninterface, calc.v)[1, :]
    # Add the core energy if any
    if calc.workspace.hardcore.core_repulsion !== nothing
        calc.eng .+= calc.workspace.hardcore.ecore
    end
    calc.param.energy_calculated = true
    calc.param.forces_calculated = false
    fill!(calc.stress, 0.0)
    fill!(calc.forces, 0.0)
    if compute_forces
        @timeit to "backward!" backward!(
            calc.nninterface;
            gu=one(eltype(calc.v)),
            weight_and_bias=false,
        )
        # Calculate the gradient of the feature vectors on the outputs (energies)
        @timeit to "gradinp!" gradinp!(calc.gv, calc.nninterface)
        # Apply chain rule to get the forces
        n1bd = one_body_offset(calc.cf)
        # Force is only applicable on n-body features where N>1
        @timeit to "apply_chainrule!" force_via_chainrule!(calc, calc.gv, offset=n1bd)
        # Scale stress by the volume
        calc.stress ./= volume(get_cell(calc))
        calc.param.forces_calculated = true
    end
end


"""
    update_feature_vector!(calc::NNCalc; rebuild_nl=true, gradients=true)

Returns the updated the feature vectors after atomic displacements
"""
function update_feature_vector!(calc::NNCalc; rebuild_nl=true, compute_gradients=true)

    cell = get_cell(calc)
    nl = calc.nl
    fill!(calc.v, 0)

    # Update or rebuild the neighbour list
    rebuild_nl ? rebuild!(nl, cell) : CellBase.update!(nl, cell)

    # Update the vectors
    one_body_vectors!(calc.v, cell, calc.cf)
    n1bd = one_body_offset(calc.cf)
    if compute_gradients
        @timeit to "compute_fv_gv!" compute_fv_gv!(calc.workspace, calc.cf, cell; nl, offset=n1bd)
    else
        @timeit to "compute_fv!" compute_fv!(calc.workspace, calc.cf, cell; nl, offset=n1bd)
    end
    # Construct the combined feature vector that includes onebody interactions
    calc.v
end

"""
    eddp_pressure_gpa(vc::AbstractCalc) 

Return pressure in unit of GPa.
"""
function eddp_pressure_gpa(vc::AbstractCalc)
    eVAngToGPa(tr(eddp_stress(vc)) / 3.0)
end

"""
Apply Chain rule to compute forces
"""
function force_via_chainrule!(calc, gv; offset=calc.workspace.one_body_offset)
    _force_update!(calc.workspace, gv; offset)
    _stress_update!(calc.workspace, gv; offset)
    calc
end

"""
     check_global_minsep(nl::NeighbourList, threshold)

Check if there are atoms that are too close to each other.
"""
function check_global_minsep(nl::NeighbourList, minsep)
    for i = 1:length(nl.nneigh)
        for (_, _, d) in eachneighbour(nl, i)
            if d < minsep
                return false
            end
        end
    end
    return true
end

"""
    eddp_pressure(calc::AbstractCalc)

Return the total pressure obtained from the trace of the stress tensor.
"""
function eddp_pressure(calc::AbstractCalc)
    stress = eddp_stress(calc)
    return (stress[1, 1] + stress[2, 2] + stress[3, 3]) / 3
end


## Add Convenience methods
for func in [:eddp_energy, :eddp_forces, :eddp_pressure, :eddp_energy_std, :eddp_enthalpy]
    @eval begin
        function $func(cell::Cell, cf::CellFeature, itf::AbstractNNInterface; kwargs...)
            $func(NNCalc(cell, cf, itf); kwargs...)
        end
        @doc """
            $($func)(cell::Cell, cf::CellFeature, itf::AbstractNNInterface;kwargs...)

        Convenient method for calling $($func) with a Calculator constructed ad-hoc.
        """ $func(cell::Cell, cf::CellFeature, itf::AbstractNNInterface; kwargs...)
    end
end

"""
    eddp_energy_per_atom(calc_or_cell, args...; kwargs...) -> Number

Total energy normalized by the number of atoms (eV/atom).
"""
eddp_energy_per_atom(cell::Cell, args...; kwargs...) =
    eddp_energy(cell, args...; kwargs...) / length(cell)
eddp_energy_per_atom(calc::AbstractCalc, args...; kwargs...) =
    eddp_energy(calc, args...; kwargs...) / length(get_cell(calc))
"""
    eddp_energy_std_per_atom(calc_or_cell, args...; kwargs...) -> Number

Ensemble energy standard deviation normalized per atom (eV/atom).
Useful for assessing model confidence independently of system size.
"""
eddp_energy_std_per_atom(cell::Cell, args...; kwargs...) =
    eddp_energy_std(cell, args...; kwargs...) / length(cell)
eddp_energy_std_per_atom(calc::AbstractCalc, args...; kwargs...) =
    eddp_energy_std(calc, args...; kwargs...) / length(get_cell(calc))




