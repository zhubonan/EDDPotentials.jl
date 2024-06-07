import Optim
import Base
using Optim: LBFGS, optimize
using LineSearches: HagerZhang
import CellBase: set_cellmat!, set_positions!, get_cellmat, get_positions
using StatsBase: ZScoreTransform, transform!
abstract type AbstractCalc end
const AC = AbstractCalc

## Methods to be implemented for the Abstract type

function get_cell end

function get_energy end

function calculate! end

function get_forces end

function get_stress end


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

mutable struct NNCalc{T,N<:NeighbourList,M<:CellFeature,X<:AbstractNNInterface} <:
               AbstractCalc
    cell::Cell{T}
    last_cell::Cell{T}
    "Positions when the NeighbourList is built"
    last_nn_build_pos::Vector{SVector{3,T}}
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
end

get_cell(ac::NNCalc) = ac.cell

"""
Copy the lattice and positions from one cell to the other
"""
function copycell!(cell_from::Cell, cell_to::Cell)
    set_cellmat!(cell_to, cellmat(cell_from))
    set_positions!(cell_to, positions(cell_from))
    species(cell_to) .= species(cell_from)
end

"""
    is_equal(cell_a, cell_b)

Check if two cells are equal to each other
"""
function is_equal(cell_a, cell_b)
    all(cellmat(cell_a) .== cellmat(cell_b)) &&
        all(positions(cell_a) .== positions(cell_b)) &&
        all(species(cell_a) .== species(cell_b))
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
    core=CoreReplusion(1.0),
) where {T}
    nl = NeighbourList(cell, rcut, nmax; savevec)
    v = zeros(T, nfeatures(cf), length(cell))

    fb = GradientWorkspace(v; core, do_grad=true, one_body_offset=feature_size(cf)[1]) # Buffer for force calculation 
    NNCalc(
        cell,
        deepcopy(cell),
        sposarray(cell),
        nl,
        cf,
        v,
        similar(v),  # Gradient of the input to the NN 
        fb,
        fb.tot_forces,  # Forces
        fb.tot_stress,  # Stress
        zeros(T, nions(cell)), # Energy
        NNCalcParam(; mode=mode),
        nn,
    )
end

function _reinit_fb!(calc, mode)
    if mode != calc.param.mode
        fb = GradientWorkspace(calc.v; calc.force_buffer.core, mode) # Buffer for force calculation 
        calc.force_buffer = fb
        calc.forces = fb.forces
        calc.stress = fb.stress
        calc.param.mode = mode
        calc.param.forces_calculated = false
        calc.param.energy_calculated = false
    end
end

"""
    get_energy(calc::NNCalc; forces=false, rebuild_nl=true)

Return the total energy of the calculator.
"""
function get_energy(calc::NNCalc; forces=false, rebuild_nl=true)
    calculate!(calc; forces, rebuild_nl)
    # Include the core energy if any
    sum(calc.eng)
end

function get_enthalpy(calc::NNCalc; forces=false, rebuild_nl=true)
    get_energy(calc; forces, rebuild_nl)
end

function get_forces(calc::NNCalc; rebuild_nl=true, make_copy=true)
    calculate!(calc; forces=true, rebuild_nl)
    if make_copy
        return copy(calc.forces)
    end
    calc.forces
end

function get_stress(calc::NNCalc; rebuild_nl=true, make_copy=true)
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
    copycell!(calc.cell, calc.last_cell)
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
function get_energy_std(calc::NNCalc{T,N,M,X}) where {T,N,M,X<:EnsembleNNInterface}
    get_energy(calc)
    per_atom = reduce(vcat, forward!.(calc.nninterface.models, Ref(calc.v)))
    std(sum(per_atom, dims=2))
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
    get_pressure_gpa(vc::AbstractCalc) 

Return pressure in unit of GPa.
"""
function get_pressure_gpa(vc::AbstractCalc)
    eVAngToGPa(tr(EDDPotential.get_stress(vc)) / 3.0)
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
get_energy(c::VariableCellCalc; rebuild_nl=true, kwargs...) =
    get_energy(c.calc; rebuild_nl, kwargs...)

get_energy_std(vc::VariableCellCalc) = get_energy_std(vc.calc)

function VariableCellCalc(calc::NNCalc{T}; external_pressure=zeros(T, 3, 3)) where {T}
    latt = copy(cellmat(get_cell(calc)))
    lattice_factor = convert(T, nions(get_cell(calc)))
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
    apos = dgrad \ positions(cell)
    lpos = vc.lattice_factor .* dgrad'
    hcat(apos, lpos)
end


"""
    _get_forces_and_stress(cf::VariableCellCalc;rebuild_nl)

Construct force and stress for the filter object
"""
function _get_forces_and_stress(vc::VariableCellCalc; rebuild_nl, kwargs...)
    calculate!(vc.calc; forces=true, rebuild_nl, kwargs...)
    dgrad = deformgradient(vc)
    vol = volume(get_cell(vc))
    # Compute the stress
    stress = get_stress(vc.calc) .- vc.external_pressure
    forces = get_forces(vc.calc)

    virial = (dgrad \ (vol .* stress))
    # Deformed forces
    atomic_forces = dgrad * forces

    out_forces = hcat(atomic_forces, virial ./ vc.lattice_factor)
    out_stress = -virial / vol
    out_forces, out_stress
end

"""
Forces including the stress contributions that is consistent with the augmented positions
"""
get_forces(vc::VariableCellCalc; rebuild_nl=true, kwargs...) =
    _get_forces_and_stress(vc; rebuild_nl, kwargs...)[1]
_get_effective_stress(vc::VariableCellCalc; rebuild_nl=true, kwargs...) =
    _get_forces_and_stress(vc; rebuild_nl, kwargs...)[2]

"""
    get_stress(vc::VariableCellCalc;kwargs...)
Return the effective stress of the VaraibleCellCalc (including the external pressure)
"""
get_stress(vc::VariableCellCalc; rebuild_nl=true, kwargs...) =
    get_stress(vc.calc; rebuild_nl, kwargs...)

function get_enthalpy(vc::VariableCellCalc; kwargs...)
    stress = vc.external_pressure
    # H = E + PV
    get_energy(vc; kwargs...) +
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
    set_cellmat!(cell, new_dgrad * cellmat(vc.orig_lattice))
    set_positions!(cell, new_dgrad * pos)
end

"""
     check_global_minsep(nl::NeighbourList, threshold)

Check if there are atoms that are too close to each other.
"""
function check_global_minsep(nl::NeighbourList, threshold)
    for i = 1:length(nl.nneigh)
        for (_, _, d) in eachneighbour(nl, i)
            if d < threshold
                return false
            end
        end
    end
    return true
end

"""
    get_pressure(calc::VariableCellCalc)

Return the total pressure with the external pressure subtracted.
"""
function get_pressure(calc::AbstractCalc)
    stress = get_stress(calc)
    return (stress[1, 1] + stress[2, 2] + stress[3, 3]) / 3
end


## Add Convenience methods
for func in [:get_energy, :get_forces, :get_pressure, :get_energy_std, :get_enthalpy]
    @eval begin
        function $func(cell::Cell, cf::CellFeature, itf::AbstractNNInterface; kwargs...)
            $func(VariableCellCalc(NNCalc(cell, cf, itf)); kwargs...)
        end
        @doc """
            $($func)(cell::Cell, cf::CellFeature, itf::AbstractNNInterface;kwargs...)

        Convenient method for calling $(EDDPotential.$func) with a Calculator constructed ad-hoc.
        """ $func(cell::Cell, cf::CellFeature, itf::AbstractNNInterface; kwargs...)
    end
end

get_energy_per_atom(cell::Cell, args...; kwargs...) =
    get_energy(cell, args...; kwargs...) / length(cell)
get_energy_per_atom(calc::AbstractCalc, args...; kwargs...) =
    get_energy(calc, args...; kwargs...) / length(get_cell(calc))
get_energy_std_per_atom(cell::Cell, args...; kwargs...) =
    get_energy_std(cell, args...; kwargs...) / length(cell)
get_energy_std_per_atom(calc::AbstractCalc, args...; kwargs...) =
    get_energy_std(calc, args...; kwargs...) / length(get_cell(calc))
