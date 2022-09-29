import CellBase: set_cellmat!, set_positions!, get_cellmat, get_positions
using StatsBase: ZScoreTransform, transform!
abstract type AbstractCalc end
const AC = AbstractCalc

## Methods to be implemented for the Abstract type

function get_cell(ac::AC) end
function get_energy(ac::AC) end

function calculate!(ac::AC) end

function get_forces(calc::AC) end

function get_stress(calc::AC) end


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


function get_energy(calc::NNCalc;forces=false, rebuild_nl=true)
    calculate!(calc;forces, rebuild_nl)
    sum(calc.eng)
end

function get_forces(calc::NNCalc;rebuild_nl=true)
    calculate!(calc;forces=true, rebuild_nl)
    calc.forces
end

function get_stress(calc::NNCalc;rebuild_nl=true)
    calculate!(calc;forces=true, rebuild_nl)
    calc.stress
end

function _need_calc(calc, forces)
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
