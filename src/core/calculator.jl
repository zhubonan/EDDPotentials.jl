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
abstract type AbstractCalc end
const AC = AbstractCalc

## Methods to be implemented for the Abstract type

function get_cell end

function eddp_energy end

function calculate! end

function eddp_forces end

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

mutable struct NNCalc{T,N<:NeighbourList,M<:CellFeature,X<:AbstractNNInterface,D} <:
               AbstractCalc
    cell::Cell{T,D}
    last_cell::Cell{T,D}
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
    is_equal(cell_a, cell_b)

Check if two cells are equal to each other
"""
function is_equal(cell_a, cell_b)
    all(cellmat(cell_a) .== cellmat(cell_b)) &&
        all(positions(cell_a) .== positions(cell_b)) &&
        all(species(cell_a) .== species(cell_b))
end

_make_symbol_keys(x) = x
function _make_symbol_keys(dict::Dict{T,K}) where {T,K}
    Dict{Symbol,K}(Symbol(key) => _make_symbol_keys(value) for (key, value) in pairs(dict))
end

function NNCalc(
    cell::Cell{T},
    cf::CellFeature,
    nn::AbstractNNInterface;
    shell_size=2.0,
    rcut=suggest_rcut(cf; shell=shell_size),
    mode="one-pass",
    nmax=500,
    savevec=true,
    core=CoreRepulsion(1.0),
    elemental_energies=Dict{Symbol, T}(),
) where {T}
    nl = NeighbourList(cell, rcut, nmax; savevec)
    v = zeros(T, nfeatures(cf), length(cell))

    workspace = GradientWorkspace(
        v;
        ndims=CellBase._n_dimensions(cell),
        core,
        do_grad=true,
        one_body_offset=feature_size(cf)[1],
    ) # Buffer for force calculation 
    NNCalc(
        cell,
        deepcopy(cell),
        sposarray(cell),
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

Return the total energy of the calculator.
"""
function eddp_energy(calc::NNCalc; forces=false, rebuild_nl=true)
    calculate!(calc; forces, rebuild_nl)
    sum(calc.eng) + get_elemental_energy(calc.cell, calc.elemental_energies)
end

get_elemental_energy(cell::Cell, ref_energies::Dict) =
    sum(x -> get(ref_energies, x, 0.0), species(cell))

function eddp_enthalpy(calc::NNCalc; forces=false, rebuild_nl=true)
    eddp_energy(calc; forces, rebuild_nl)
end

function eddp_forces(calc::NNCalc; rebuild_nl=true, make_copy=true)
    calculate!(calc; forces=true, rebuild_nl)
    if make_copy
        return copy(calc.forces)
    end
    calc.forces
end

function eddp_stress(calc::NNCalc; rebuild_nl=true, make_copy=true)
    calculate!(calc; forces=true, rebuild_nl)
    if make_copy
        return copy(calc.stress)
    end
    calc.stress
end

function _need_calc(calc::NNCalc, forces)
    if is_equal(get_cell(calc), calc.last_cell) && calc.param.energy_calculated
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

Core function used to compute energy, forces and stress.
"""
@timeit to function calculate!(calc::NNCalc; forces=true, rebuild_nl=true)
    # Nothing to do if the cell has not changed since last time

    _need_calc(calc, forces) || return

    @timeit to "_rebuild" _rebuild_on_demand(calc; rebuild_nl)
    # Disable rebuilding NL - we have done this already
    @timeit to "_calculate" _calculate!(calc, false, forces)
    # Update as the last calculated cell
    calc.last_cell = deepcopy(calc.cell)
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
Return standard deviation of the predicted total energy
Note: must be run after a energy call!
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
    if calc.workspace.hardcore.core !== nothing
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
        n1bd = feature_size(calc.cf)[1]
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
    n1bd, n2bd, _ = feature_size(calc.cf)
    if compute_gradients
        #if isnothing(gv)
        @timeit to "compute_fv_gv!" compute_fv_gv!(
            calc.workspace,
            calc.cf.two_body,
            calc.cf.three_body,
            cell;
            nl,
            offset=n1bd,
        )
        #else
        #    @timeit to "compute_fv_gv!" compute_fv_gv!(calc.force_buffer, calc.cf.two_body, calc.cf.three_body, cell, gv;nl, offset=n1bd)
        #end
    else
        @timeit to "compute_fv!" compute_fv!(
            calc.workspace,
            calc.cf.two_body,
            calc.cf.three_body,
            cell;
            nl,
            offset=n1bd,
        )
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

### Filter for allowing variable cell optimisation


"""

Filter for including lattice vectors in the optimisation


Reference: 
 E. B. Tadmor, G. S. Smith, N. Bernstein, and E. Kaxiras,
            Phys. Rev. B 59, 235 (1999)

Base on ase.constraints.UnitCellFilter
"""
struct VariableCellCalc{T,C} <: AbstractCalc
    calc::C
    orig_lattice::Lattice{T}
    lattice_factor::T
    external_pressure::Matrix{T}
end


_need_calc(calc::VariableCellCalc, forces) = _need_calc(calc.calc, forces)
get_cell(c::VariableCellCalc) = get_cell(c.calc)
eddp_energy(c::VariableCellCalc; rebuild_nl=true, kwargs...) =
    eddp_energy(c.calc; rebuild_nl, kwargs...)

eddp_energy_std(vc::VariableCellCalc) = eddp_energy_std(vc.calc)

function VariableCellCalc(calc::AbstractCalc; external_pressure=zeros(3, 3))
    # Reconstruct if a VariableCellCalc is passed
    if isa(calc, VariableCellCalc) 
        calc = calc.calc
    end
    latt = copy(cellmat(get_cell(calc)))
    lattice_factor = convert(eltype(latt), nions(get_cell(calc)))
    VariableCellCalc(calc, Lattice(latt), lattice_factor, external_pressure)
end

raw"""
Obtain the deformation gradient matrix such that 

$$F C = C'$$
"""
function deformgradient(vc::VariableCellCalc)
    copy(transpose(transpose(cellmat(vc.orig_lattice)) \ transpose(cellmat(get_cell(vc)))))
end

"""
    get_positions(cf::VariableCellCalc)

Composed positions vectors including cell defromations
"""
function CellBase.get_positions(vc::VariableCellCalc)
    cell = get_cell(vc)
    dgrad = deformgradient(vc)
    apos = copy(positions(cell))
    apos[1:3, :] .= dgrad \ apos[1:3, :]
    lpos = vc.lattice_factor .* dgrad'
    # Pad the "fake" positions if needed (for high dimensional cell)
    npad = size(apos, 1) - size(lpos, 1)
    if npad > 0
        lpos = vcat(lpos, repeat([zero(eltype(lpos))], npad, 3))
    end
    hcat(apos, lpos)
end


"""
    _eddp_forces_and_stress(cf::VariableCellCalc;rebuild_nl)

Construct force and stress for the filter object
"""
function _eddp_forces_and_stress(vc::VariableCellCalc; rebuild_nl, kwargs...)
    # Do not forward passthrough kwargs (e.g. `make_copy`) to `calculate!`.
    calculate!(vc.calc; forces=true, rebuild_nl)
    dgrad = deformgradient(vc)
    vol = volume(get_cell(vc))
    # Compute the stress
    stress = eddp_stress(vc.calc) .- vc.external_pressure
    virial = (dgrad \ (vol .* stress))

    # Deformed forces
    atomic_forces = copy(eddp_forces(vc.calc))
    atomic_forces[1:3, :] .= dgrad * atomic_forces[1:3, :]
    # Pad zeros
    virial_forces = repeat([zero(eltype(atomic_forces))], size(atomic_forces, 1), 3)
    virial_forces[1:3, :] = virial ./ vc.lattice_factor

    out_forces = hcat(atomic_forces, virial_forces)
    out_stress = -virial / vol
    out_forces, out_stress
end

"""
Forces including the stress contributions that is consistent with the augmented positions
"""
eddp_forces(vc::VariableCellCalc; rebuild_nl=true, kwargs...) =
    _eddp_forces_and_stress(vc; rebuild_nl, kwargs...)[1]
_get_effective_stress(vc::VariableCellCalc; rebuild_nl=true, kwargs...) =
    _eddp_forces_and_stress(vc; rebuild_nl, kwargs...)[2]

"""
    eddp_stress(vc::VariableCellCalc;kwargs...)
Return the effective stress of the VaraibleCellCalc (including the external pressure)
"""
eddp_stress(vc::VariableCellCalc; rebuild_nl=true, kwargs...) =
    eddp_stress(vc.calc; rebuild_nl, kwargs...)

function eddp_enthalpy(vc::VariableCellCalc; kwargs...)
    stress = vc.external_pressure
    # H = E + PV
    eddp_energy(vc; kwargs...) +
    volume(get_cell(vc)) * (stress[1, 1] + stress[2, 2] + stress[3, 3]) / 3
end


"""
Update the positions passing through the filter
"""
function set_positions!(vc::VariableCellCalc, new)
    cell = get_cell(vc)
    nat = nions(cell)
    pos = new[:, 1:nat]
    new_dgrad = transpose(new[:, nat+1:end]) ./ vc.lattice_factor
    # Set the cell according to the deformation
    set_cellmat!(cell, new_dgrad[:, 1:3] * cellmat(vc.orig_lattice); scale_positions=false)
    # Scale the positions for the first three dimensions
    pos_new = copy(pos)
    pos_new[1:3, :] .= new_dgrad[:, 1:3] * pos_new[1:3, :]
    set_positions!(cell, pos_new)
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
    eddp_pressure(calc::VariableCellCalc)

Return the total pressure with the external pressure subtracted.
"""
function eddp_pressure(calc::AbstractCalc)
    stress = eddp_stress(calc)
    return (stress[1, 1] + stress[2, 2] + stress[3, 3]) / 3
end


## Add Convenience methods
for func in [:eddp_energy, :eddp_forces, :eddp_pressure, :eddp_energy_std, :eddp_enthalpy]
    @eval begin
        function $func(cell::Cell, cf::CellFeature, itf::AbstractNNInterface; kwargs...)
            $func(VariableCellCalc(NNCalc(cell, cf, itf)); kwargs...)
        end
        @doc """
            $($func)(cell::Cell, cf::CellFeature, itf::AbstractNNInterface;kwargs...)

        Convenient method for calling $($func) with a Calculator constructed ad-hoc.
        """ $func(cell::Cell, cf::CellFeature, itf::AbstractNNInterface; kwargs...)
    end
end

eddp_energy_per_atom(cell::Cell, args...; kwargs...) =
    eddp_energy(cell, args...; kwargs...) / length(cell)
eddp_energy_per_atom(calc::AbstractCalc, args...; kwargs...) =
    eddp_energy(calc, args...; kwargs...) / length(get_cell(calc))
eddp_energy_std_per_atom(cell::Cell, args...; kwargs...) =
    eddp_energy_std(cell, args...; kwargs...) / length(cell)
eddp_energy_std_per_atom(calc::AbstractCalc, args...; kwargs...) =
    eddp_energy_std(calc, args...; kwargs...) / length(get_cell(calc))



raw"""
Compute the energy and force penalty for going into the extra dimensions

```math
E^{\prime} = \sum_i x_i ^2 \alpha / 2
F_i^{\prime} = -x_i 
```
"""
mutable struct DimPenalty{T,D,C} <: AbstractCalc
    cell::Cell{T,D}
    alpha::T
    calculator::C
end

set_positions!(calc::DimPenalty, new) = set_positions!(get_cell(calc), new)
set_cellmat!(calc::DimPenalty, new) = set_cellmat!(get_cell(calc), new)

function DimPenalty(
    calc::AbstractCalc;
    alpha=eltype(positions(get_cell(calc)))(0.1),
    kwargs...,
)
    DimPenalty(get_cell(calc), alpha, calc)
end


"""
Do nothing as quantities are calculated by each
force/stress/energy call.
"""
function calculate!(calc::DimPenalty; kwargs...)
    nothing
end

function eddp_forces(calc::DimPenalty; kwargs...)
    if calc.calculator !== nothing
        pos = get_positions(calc.calculator)
    else
        pos = positions(calc.cell)
    end
    forces = pos .* -2 * calc.alpha
    # Set the first three dimensions to zero
    forces[1:3, :] .*= 0
    forces
end

function eddp_energy(calc::DimPenalty; kwargs...)
    sum(sum(x -> x^2, @view(positions(get_cell(calc))[4:end, :]), dims=1))
end

function eddp_stress(::DimPenalty{T}; kwargs...) where {T}
    zeros(T, 3, 3)
end


get_cell(calc::DimPenalty) = calc.cell


struct MultiCalc{T} <: AbstractCalc
    calcs::T
    function MultiCalc(calcs)
        _calcs = collect(calcs)
        # Ensure that variable cell calculation is the first one so we call its methods when setting/getting 
        # positions and cell
        ivar = findfirst(x -> isa(x, VariableCellCalc), calcs)
        if ivar !== nothing
            varcalc = popat!(_calcs, ivar)
            pushfirst!(_calcs, varcalc)
        end
        tup = tuple(_calcs...)
        new{typeof(tup)}(tup)
    end
end

MultiCalc(calc...) = MultiCalc(calc)

eddp_energy(calc::MultiCalc; kwargs...) = sum(eddp_energy(c;kwargs...) for c in calc.calcs)
eddp_forces(calc::MultiCalc; kwargs...) = sum([eddp_forces(c;kwargs...) for c in calc.calcs])
eddp_stress(calc::MultiCalc; kwargs...) = sum([eddp_stress(c;kwargs...) for c in calc.calcs])
get_cell(calc::MultiCalc) = get_cell(calc.calcs[1])
set_positions!(calc::MultiCalc, new) = set_positions!(calc.calcs[1], new)
set_cellmat!(calc::MultiCalc, new) = set_cellmat!(calc.calcs[1], new)
get_positions(calc::MultiCalc) = get_positions(calc.calcs[1])

function calculate!(calc::MultiCalc;kwargs...) 
    for obj in calc.calcs
        calculate!(obj;kwargs...) 
    end
end
