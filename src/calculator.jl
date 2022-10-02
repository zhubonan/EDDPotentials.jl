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


function set_cellmat!(calc::AC, cellmat)
    set_cellmat!(get_cell(calc), cellmat)
end

function set_positions!(calc::AC, positions)
    set_positions!(get_cell(calc), positions)
end

function Base.show(io::IO, c::AbstractCalc)
    println(io, "$(typeof(c)) for: ")
    Base.show(io, get_cell(c))
end

function Base.show(io::IO, m::MIME"text/plain", cw::AbstractCalc)
    println(io, "$(typeof(c)) for: ")
    Base.show(io, m, cw.cell)
end

### Concrete implementation

mutable struct NNCalc{T,N<:NeighbourList,M<:CellFeature,X<:AbstractNNInterface} <: AbstractCalc
    cell::Cell{T}
    last_cell::Cell{T}
    "NeighbourList"
    nl::N
    cf::M
    "Combined Feature Vector"
    v::Matrix{T}
    "Gradient of the feature vector"
    gv::Matrix{T}
    "Tuple of forces buffers"
    force_buffer::ForceBuffer{T}
    "Forces"
    forces::Matrix{T}
    "Stress"
    stress::Matrix{T}
    "atomic_energy"
    eng::Vector{T}
    "flat to ignore one body interactions or not"
    ignore_one_body::Bool
    "Has energy being calculated?"
    energy_calculated::Bool
    "Has forces being calculated?"
    forces_calculated::Bool
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

function is_equal(cell_a, cell_b)
    all(cellmat(cell_a) .== cellmat(cell_b)) && all(positions(cell_a) .== positions(cell_b)) && all(species(cell_a) .== species(cell_b))
end


function NNCalc(cell::Cell{T}, cf::CellFeature, nn::AbstractNNInterface; rcut=suggest_rcut(cf),
    nmax=500, savevec=true, ignore_one_body=false) where {T}
    nl = NeighbourList(cell, rcut, nmax; savevec)
    v = zeros(T, nfeatures(cf; ignore_one_body=false), length(cell))
    v2 = zeros(T, nfeatures(cf; ignore_one_body=true), length(cell))

    fb = ForceBuffer{T}(v2) # Buffer for force calculation 
    NNCalc(cell,
        deepcopy(cell),
        nl,
        cf,
        v,
        similar(v),  # Gradient of the input to the NN 
        fb,
        fb.forces,  # Forces
        fb.stress,  # Stress
        zeros(T, nions(cell)), # Energy
        ignore_one_body,
        false,
        false,
        nn
    )
end


function get_energy(calc::NNCalc; forces=false, rebuild_nl=true)
    calculate!(calc; forces, rebuild_nl)
    sum(calc.eng)
end

function get_forces(calc::NNCalc; rebuild_nl=true)
    calculate!(calc; forces=true, rebuild_nl)
    calc.forces
end

function get_stress(calc::NNCalc; rebuild_nl=true)
    calculate!(calc; forces=true, rebuild_nl)
    calc.stress
end

function _need_calc(calc::NNCalc, forces)
    if is_equal(get_cell(calc), calc.last_cell) && calc.energy_calculated
        if forces
            calc.forces_calculated && return false
        else
            return false
        end
    end
    return true
end



function calculate!(calc::NNCalc; forces=true, rebuild_nl=true)
    # Nothing to do if the cell has not changed since last time

    _need_calc(calc, forces) || return

    update_feature_vector!(calc; rebuild_nl, gradients=forces)

    # Energy evaluation
    calc.eng .= forward!(calc.nninterface, calc.v)[1, :]
    calc.energy_calculated = true
    calc.forces_calculated = false
    fill!(calc.stress, 0.0)
    fill!(calc.forces, 0.0)
    if forces
        backward!(calc.nninterface; gu=one(eltype(calc.v)), weight_and_bias=false)
        # Calculate the gradient of the feature vectors on the outputs (energies)
        gradinp!(calc.gv, calc.nninterface)
        # Apply chain rule to get the forces
        n1bd = feature_size(calc.cf)[1]
        # Force is only applicable on n-body features where N>1
        apply_chainrule!(calc.force_buffer, @view(calc.gv[n1bd+1:end, :]))
        # Scale stress by the volume
        calc.stress ./= volume(get_cell(calc))
        calc.forces_calculated = true
    end
    # Update as the last calculated cell
    copycell!(calc.cell, calc.last_cell)
end

"""
    update_feature_vector!(wt::CellWorkSpace)

Returns the updated the feature vectors after atomic displacements
"""
function update_feature_vector!(calc::NNCalc; rebuild_nl=true, gradients=true, global_minsep=0.01, maxvol=100)

    cell = get_cell(calc)
    nl = calc.nl

    # Update or rebuild the neighbour list
    rebuild_nl ? rebuild!(nl, cell) : update!(nl, cell)

    # Update the vectors
    one_body_vectors!(calc.v, cell, calc.cf)
    n1bd, n2bd, _ = feature_size(calc.cf)
    if gradients
        compute_two_body_fv_gv!(calc.force_buffer, calc.cf.two_body, cell; nl)
        compute_three_body_fv_gv!(calc.force_buffer, calc.cf.three_body, cell; nl, offset=n2bd)
        # Update total stress - simple summartion of the atomic contributions
        calc.force_buffer.stotv .= sum(calc.force_buffer.svec, dims=5)
    else
        feature_vector!(calc.force_buffer.fvec, calc.cf.two_body, cell; nl)
        feature_vector!(calc.force_buffer.fvec, calc.cf.three_body, cell; nl, offset=n2bd)
    end
    # Construct the combined feature vector that includes onebody interactions
    calc.v[n1bd+1:end, :] .= calc.force_buffer.fvec
    calc.v
end

"""
    get_pressure_gpa(vc::AbstractCalc) 

Return pressure in unit of GPa.
"""
function get_pressure_gpa(vc::AbstractCalc)
    eVAngToGPa(tr(EDDP.get_stress(vc)) / 3.0)
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
    lattice_factor::Float64
end

_need_calc(calc::VariableCellCalc, forces) = _need_calc(calc.calc, forces)
get_cell(c::VariableCellCalc) = get_cell(c.calc)
get_energy(c::VariableCellCalc; rebuild_nl=true, kwargs...) = get_energy(c.calc; rebuild_nl, kwargs...)

VariableCellCalc(calc) = VariableCellCalc(calc, Lattice(copy(cellmat(get_cell(calc)))), Float64(nions(get_cell(calc))))

raw"""
Obtain the deformation gradient matrix such that 

$$F C = C'$$
"""
function deformgradient(vc::VariableCellCalc)
    copy(
        transpose(
            transpose(cellmat(vc.orig_lattice)) \ transpose(cellmat(get_cell(vc)))
        )
    )
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
    stress = get_stress(vc.calc)
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
get_forces(vc::VariableCellCalc; rebuild_nl=true, kwargs...) = _get_forces_and_stress(vc; rebuild_nl, kwargs...)[1]
get_stress(vc::VariableCellCalc; rebuild_nl=true, kwargs...) = _get_forces_and_stress(vc; rebuild_nl, kwargs...)[2]


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

    # Set the positions
    cell.positions .= new_dgrad * pos
end


"Covert eV/Å^3 to GPa"
eVAngToGPa(x) = 160.21766208 * x

"""
    optimise!(calc::AbstractCalc)

Optimise the cell with LBFGS from Optim. Collect the trajectory if requested.
Note that the trajectory is collected for all force evaluations and may not 
corresponds to the actual iterations of the underlying LBFGS iterations.
"""
function optimise!(calc::AbstractCalc; show_trace=false, record_trajectory=false, stepmax=2.0, 
                   g_abstol=1e-6, f_reltol=0.0, successive_f_tol=2, 
                   method=LBFGS(; linesearch=HagerZhang(; alphamax=stepmax)))
    p0 = get_positions(calc)[:]
    traj = []

    "Energy"
    function fo(x, calc)
        set_positions!(calc, reshape(x, 3, :))
        get_energy(calc)
    end

    "Gradient"
    function go(x, calc)
        set_positions!(calc, reshape(x, 3, :))
        forces = get_forces(calc)
        # ∇E = -F
        forces .*= -1
        # Collect the trajectory if requested
        if record_trajectory
            cell = deepcopy(get_cell(calc))
            cell.metadata[:enthalpy] = get_energy(calc)
            cell.arrays[:forces] = get_forces(calc)
            push!(traj, cell)
        end
        forces
    end
    res = optimize(x -> fo(x, calc), x -> go(x, calc), p0, method, Optim.Options(; show_trace=show_trace, g_abstol, f_reltol, successive_f_tol); inplace=false)
    res, traj
end


"""
     check_global_minsep(nl::NeighbourList, threshold)

Check if there are atoms that are too close to each other.
"""
function check_global_minsep(nl::NeighbourList, threshold)
    for i in 1:length(nl.nneigh)
        for (_, _, d) in eachneighbour(nl, i)
            if d < threshold
                return false
            end
        end
    end
    return true
end
