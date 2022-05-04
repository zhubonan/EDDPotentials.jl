#=
Support code for using the potentials for efficient energy/force/stress
calculations.
=#
using Zygote: gradient as zgradient
import Base
import CellBase
import CellBase: rebuild!, update!
export rebuild!, update!

"""
Combination of a cell, its associated neighbour list and feature vectors

This is to allow efficient re-calculation of the feature vectors without reallocating memory.
"""
struct CellWorkSpace{T, N, M}
    cell::T
    nl::N
    cf::M
    # Feature vectors
    one_body
    two_body
    three_body
    v
    "Buffer for two body force calculations"
    two_body_fbuffer
    "Buffer for three body force calculations"
    three_body_fbuffer
    "Forces"
    forces
    "Stress"
    stress
    "atomic_energy"
    eng
    "flat to ignore one body interactions or not"
    ignore_one_body
end

"Number of features used in the CellWorkSpace"
nfeatures(cw::CellWorkSpace)  = size(cw.v, 1)

function Base.show(io::IO, cw::CellWorkSpace)
    println(io, "CellWorkspace for: ")
    Base.show(io, cw.cell)
end

function Base.show(io::IO, m::MIME"text/plain", cw::CellWorkSpace)
    println(io, "CellWorkspace for: ")
    Base.show(io, m, cw.cell)
end


"""
    ForceBuffer{T}

Buffer for storing forces and stress and support their calculations
"""
struct ForceBuffer{T}
    "dF/dri"
    gvec::Array{T, 4}
    "dF/dσi"
    svec::Array{T, 5}
    "dF/dσ"
    stotv::Array{T, 4}
    "Calculated forces"
    forces::Array{T, 2}
    "Calculated stress"
    stress::Array{T, 2}
end

"""
Initialise a buffer for computing forces
"""
function ForceBuffer{T}(nf, nat;ndims=3) where {T}
    gvec = zeros(T, nf, nat, ndims, nat)
    svec = zeros(T, nf, nat, ndims, ndims, nat)
    stotv = zeros(T, nf, nat, ndims, ndims)
    forces = zeros(ndims, nat)
    stress = zeros(ndims, ndims)
    ForceBuffer(gvec, svec, stotv, forces, stress)
end


function CellWorkSpace(cell::Cell;cf, rcut, nmax=100, savevec=true, ndims=3, ignore_one_body=true) 
    nl = NeighbourList(cell, rcut, nmax;savevec)
    us = unique(atomic_numbers(cell))
    one_body = zeros(length(us), nions(cell))
    two_body = feature_vector(cf.two_body, cell;nl=nl)
    three_body = feature_vector(cf.three_body, cell;nl=nl)

    if ignore_one_body
        v = vcat(two_body, three_body)
    else
        v = vcat(one_body, two_body, three_body)
    end

    CellWorkSpace(cell, nl, cf, one_body, two_body, three_body, v, 
    ForceBuffer{eltype(two_body)}(size(two_body, 1), nions(cell)),
    ForceBuffer{eltype(three_body)}(size(three_body, 1), nions(cell)),
    zeros(eltype(two_body), ndims, nions(cell)),
    zeros(eltype(two_body), ndims, ndims),
    zeros(eltype(two_body), nions(cell)),
    ignore_one_body,
    )
end

CellBase.rebuild!(cw::CellWorkSpace) = CellBase.rebuild!(cw.nl, cw.cell)
CellBase.update!(cw::CellWorkSpace) = CellBase.update!(cw.nl, cw.cell)

"""
    update_feature_vector!(wt::CellWorkSpace)

Update the feature vectors after atomic displacements.
"""
function update_feature_vector!(wt::CellWorkSpace;rebuild_nl=false, gradients=true)
    rebuild_nl ? rebuild!(wt) : update!(wt)

    one_body_vectors!(wt.one_body, wt.cell)
    if gradients
        compute_two_body_fv_gv!(wt.two_body, wt.two_body_fbuffer.gvec, wt.two_body_fbuffer.svec, wt.cf.two_body, wt.cell;wt.nl)
        compute_three_body_fv_gv!(wt.three_body, wt.three_body_fbuffer.gvec, wt.three_body_fbuffer.svec, wt.cf.three_body, wt.cell;wt.nl)
        # Update total stress - simple summartion of the atomic contributions
        wt.two_body_fbuffer.stotv .= sum(wt.two_body_fbuffer.svec, dims=5)
        wt.three_body_fbuffer.stotv .= sum(wt.three_body_fbuffer.svec, dims=5)
    else
        feature_vector!(wt.two_body, wt.cf.two_body, wt.cell;wt.nl)
        feature_vector!(wt.three_body, wt.cf.three_body, wt.cell;wt.nl)
    end
    # Block update
    i = 1
    if !wt.ignore_one_body
        l = size(wt.one_body, 1)
        wt.v[i: i+l-1, :] .= wt.one_body
        i += l
    end
    l = size(wt.two_body, 1)
    wt.v[i: i+l-1, :] .= wt.two_body
    i += l
    l = size(wt.three_body, 1)
    wt.v[i: i+l-1, :] .= wt.three_body
    wt.v
end

"""
    calculate_forces!(wt::CellWorkSpace, g2, g3)

Calculate the forces given gradeints of the total energy based on
dE / dF and dF / dxi

Args:
    - g2: partial derivatives of the two-body features
    - g3: partial derivatives of the three-body features
"""
function calculate_forces!(wt::CellWorkSpace, g2, g3)
    gv2 = wt.two_body_fbuffer.gvec
    gv3 = wt.three_body_fbuffer.gvec
    sv2 = wt.two_body_fbuffer.stotv
    sv3 = wt.three_body_fbuffer.stotv
    # Propagate forces
    _force_update!(wt.two_body_fbuffer.forces, gv2, g2)
    _force_update!(wt.three_body_fbuffer.forces, gv3, g3)
    # Propagate stress
    _stress_update!(wt.two_body_fbuffer.stress, sv2, g2)
    _stress_update!(wt.three_body_fbuffer.stress, sv3, g3)

    wt.forces .= wt.two_body_fbuffer.forces .+ wt.three_body_fbuffer.forces
    wt.stress .= wt.two_body_fbuffer.stress .+ wt.three_body_fbuffer.stress

    # Divide by the unit cell volume to get the stress
    wt.stress ./= volume(wt.cell)
    wt
end

"""
Apply transformation stored in the ModelEnsemble
"""
function apply_x_trans!(vec::AbstractVecOrMat, me::ModelEnsemble)
    if !isnothing(me.xt)
        StatsBase.transform!(me.xt, vec)
    end
    vec
end

"""
    calculate!(wt::CellWorkSpace, model::ModelEnsemble;forces=true, rebuild_nl=true)

Update the feature vectors and compute energy, forces and stresses (default).
"""
function calculate!(wt::CellWorkSpace, me::ModelEnsemble;forces=true, rebuild_nl=true)
    update_feature_vector!(wt;rebuild_nl, gradients=forces)

    # Ensure that we scale the feature vectors according to the one stored in the ModelEnsemble
    apply_x_trans!(wt.v, me)

    # Compute atomic energies
    wt.eng .= site_energies(me, wt)[:]

    if forces
        nf2 = sum(nfeatures, wt.cf.two_body)
        nf3 = sum(nfeatures, wt.cf.three_body)
        v = wt.v

        force_buf = zeros(size(wt.forces))
        stress_buf = zeros(size(wt.stress))

        # Obtain the gradients for each individual model
        for (ind, weight) in zip(me.models, me.weight)

            # Compute gradient from the model
            grad = zgradient(x -> sum(ind(x)), v)[1]
            if !isnothing(me.yt)
                grad .*= me.yt.scale
            end

            # Scale back to the original feature vectors
            if !isnothing(me.xt)
                grad[end-nf3-nf2+1:end, :] ./= me.xt.scale
            end

            # Locate the gradients for 2-body and 3-body terms
            g3 = grad[end-nf3+1:end, :]
            g2 = grad[end-nf3-nf2+1:end-nf3, :]
            # Compute the forces from chain rule
            calculate_forces!(wt, g2, g3)

            # Apply weighting from the ensemble
            force_buf .+= wt.forces * weight
            stress_buf .+= wt.stress * weight
        end
        # Set the forces
        wt.stress .= stress_buf
        wt.forces .= force_buf
    else
        fill!(wt.stress, 0.)
        fill!(wt.forces, 0.)
    end
end

"""
    _force_update!(buffer::Array{T, 2}, gv, g) where {T}

Propagate chain rule to obtain the forces
"""
function _force_update!(buffer::Array{T, 2}, gv, g) where {T}
    # Zero the buffer
    fill!(buffer, zero(T))
    for iat = 1:size(gv, 4)
        for j = 1:size(gv, 2)
            for i = 1:size(gv, 1)
                for _i = 1:size(buffer, 1)
                    buffer[_i, iat] += gv[i, j, _i, iat] * g[i, j] * -1  # F(xi) = -∇E(xi)
                end
            end
        end
    end
end

"""
    _stress_update!(buffer::Array{T, 2}, sv, s) where {T}

Propagate chain rule to obtain the stress
"""
function _stress_update!(buffer::Array{T, 2}, sv, s) where {T}
    # Zero the buffer
    fill!(buffer, zero(T))
    for j = 1:size(sv, 2)
        for i = 1:size(sv, 1)
            for _i = 1:3
                for _j = 1:3
                    buffer[_i, _j] += sv[i, j, _i, _j] .* s[i, j] * -1 # F(xi) = -∇E(xi)
                end
            end
        end
    end
end



"""
Per-site energy
"""
site_energies(m::ModelEnsemble, cw::CellWorkSpace) = site_energies(m, cw.v)

function site_energies(m::ModelEnsemble, inp::AbstractMatrix)
    eng = reshape(sum(m(inp) * w for (m, w) in zip(m.models, m.weight)), :, 1)
    if !isnothing(m.yt)
        StatsBase.reconstruct!(m.yt, eng)
    end
    eng
end

total_energy(m::ModelEnsemble, inp::AbstractMatrix) = sum(site_energies(m, inp))
total_energy(m::ModelEnsemble, cw::CellWorkSpace) = sum(site_energies(m, cw))

"""
    CellCalculator{T}

A calculator to support repetitive energy/forces/stress calculations with the same
model.
"""
struct CellCalculator{T}
    workspace::CellWorkSpace
    modelensemble::ModelEnsemble
    last_pos::Matrix{T}
    last_cellmat::Matrix{T}
end

function CellCalculator(cw::CellWorkSpace, me::ModelEnsemble)
    CellCalculator(cw, me, similar(positions(cw.cell)), similar(cellmat(cw.cell)))
end

"""
Calculate energy and forces

Return a tuple of energy, force, stress
"""
function calculate!(calc::CellCalculator;forces=true, rebuild_nl=true)
    # Detect any change of the structure
    should_calculate = any(calc.last_cellmat .!= cellmat(calc.workspace.cell)) || any(calc.last_pos .!= positions(calc.workspace.cell))
    if forces == true && sum(calc.workspace.forces) == 0. && sum(calc.workspace.stress) == 0.
        should_calculate = true
    end
    if should_calculate
        calculate!(calc.workspace, calc.modelensemble;forces, rebuild_nl)
    else
        return get_energy(calc.workspace), get_forces(calc.workspace), get_stress(calc.workspace)
    end
    # Update the last calculated positions
    calc.last_pos .= positions(calc.workspace.cell)
    calc.last_cellmat .= cellmat(calc.workspace.cell)
    return get_energy(calc.workspace), get_forces(calc.workspace), get_stress(calc.workspace)
end

# Getter for the energy/stress

get_cell(c::CellCalculator) = c.workspace.cell

get_forces(c::CellWorkSpace;make_copy=true) = make_copy ? copy(c.forces) : c.forces

function get_forces(m::CellCalculator;make_copy=true)
    calculate!(m)
    get_forces(m.workspace;make_copy)
end

get_energy(c::CellWorkSpace) = sum(c.eng)
function get_energy(m::CellCalculator)
    calculate!(m)
    get_energy(m.workspace)
end

get_stress(c::CellWorkSpace;make_copy=true) = make_copy ? copy(c.stress) : c.stress
function get_stress(c::CellCalculator;make_copy=true)
    calculate!(c)
    get_stress(c.workspace;make_copy)
end

"Get the positions of the cell (copy)"
CellBase.get_positions(c::CellCalculator) = CellBase.get_positions(get_cell(c))
"Get the positions of the cell (copy)"
CellBase.get_positions(c::CellWorkSpace) = CellBase.get_positions(c.cell)

# Setter
"Set the cell matrix"
CellBase.set_cellmat!(c::CellCalculator, cellmat) = CellBase.set_cellmat!(c.workspace.cell, cellmat)
"Set the positions"
CellBase.set_positions!(c::CellCalculator, pos) = CellBase.set_positions!(c.workspace.cell, pos)

#=
Optimisation

For optimisation we need to abstract away the details of operation
The only thing to be exposed is:

* a single value to be optimised (e.g. enthalpy)
* a vector controlling this value
* gradients of the value for each vector components 
=#

"Abstract type for filters of the cell"
abstract type CellFilter end

"""
Filter for including lattice vectors in the optimisation


Reference: 
 E. B. Tadmor, G. S. Smith, N. Bernstein, and E. Kaxiras,
            Phys. Rev. B 59, 235 (1999)

Base on ase.constraints.UnitCellFilter
"""
struct VariableLatticeFilter{T}
    calculator::CellCalculator
    orig_lattice::Lattice{T}
    lattice_factor::Float64
end

get_cell(c::VariableLatticeFilter) = get_cell(c.calculator)

VariableLatticeFilter(calc) = VariableLatticeFilter(calc, Lattice(copy(cellmat(get_cell(calc)))), Float64(nions(get_cell(calc))))

raw"""
Obtain the deformation gradient matrix such that 

$$F C = C'$$
"""
function deformgradient(cf::VariableLatticeFilter)
    copy(
        transpose(
        transpose(cellmat(cf.orig_lattice)) \ transpose(cellmat(get_cell(cf)))
        )
    )
end

"""
    get_positions(cf::VariableLatticeFilter)

Composed positions vectors including cell defromations
"""
function CellBase.get_positions(cf::VariableLatticeFilter)
    cell = get_cell(cf)
    dgrad = deformgradient(cf)
    apos = dgrad \ positions(cell)
    lpos = cf.lattice_factor .* dgrad'
    hcat(apos, lpos)
end

function _get_forces_and_stress(cf::VariableLatticeFilter;rebuild_nl)
    calculate!(cf.calculator;forces=true, rebuild_nl) 
    dgrad = deformgradient(cf)
    vol = volume(get_cell(cf))
    stress = get_stress(cf.calculator)
    forces = get_forces(cf.calculator)

    virial = (dgrad  \ (vol .* stress))
    # Deformed forces
    atomic_forces = dgrad * forces

    out_forces = hcat(atomic_forces, virial ./ cf.lattice_factor)
    out_stress = -virial / vol
    out_forces, out_stress
end

"""
Forces including the stress contributions
"""
get_forces(cf::VariableLatticeFilter;rebuild_nl=true) = _get_forces_and_stress(cf;rebuild_nl)[1]


"""
Update the positions passing through the filter
"""
function set_positions!(cf::VariableLatticeFilter, new)
    cell = get_cell(cf)
    nat = nions(cell)
    pos = new[:, 1:nat]
    new_dgrad = transpose(new[:, nat+1:end]) ./ cf.lattice_factor

    # Set the cell according to the deformation
    mat = cellmat(cell)
    mat .= new_dgrad * cellmat(cf.orig_lattice)

    # Set the positions
    cell.positions .= new_dgrad * pos
end

"Return the energy"
function get_energy(cf::VariableLatticeFilter;rebuild_nl=true)
    calculate!(cf.calculator;rebuild_nl)
    get_energy(cf.calculator)
end


# function stress2vot(mat)
#     [mat[1,1], mat[2,2], mat[3,3], ]
# end