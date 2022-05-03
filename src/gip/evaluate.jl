#=
Support code for using the potentials for effcient energy/force/stress
calculations.
=#
using Zygote
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
Buffer for storing forces and stress
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
        feature_vector_and_gradients!(wt.two_body, wt.two_body_fbuffer.gvec, wt.two_body_fbuffer.svec, wt.cf.two_body, wt.cell;wt.nl)
        feature_vector_and_gradients!(wt.three_body, wt.three_body_fbuffer.gvec, wt.three_body_fbuffer.svec, wt.cf.three_body, wt.cell;wt.nl)
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
TODO

Major refectoring needed:

- How to connect the gradient of the feature vectors to that of the model?
- need to ensure that the tranformations are applied correctly
- gradients obtained need to be modified by the chain rule of the transformation
"""
function calculate!(wt::CellWorkSpace, model::ModelEnsemble;forces=true, rebuild_nl=true)
    update_feature_vector!(wt;rebuild_nl, gradients=forces)
    wt.eng .= site_energies(model, wt)[:]

    nf2 = sum(nfeatures, wt.cf.two_body)
    nf3 = sum(nfeatures, wt.cf.three_body)
    if isnothing(model.xt)
        v = wt.v[end-nf2-nf3+1:end, :]
    else
        v = StatsBase.transform(model.xt, wt.v[end-nf2-nf3+1:end, :])
    end

    force_buf = zeros(size(wt.forces))
    stress_buf = zeros(size(wt.stress))
    if forces
        # Obtain the graents for each individual model
        for (ind, weight) in zip(model.models, model.weight)
            get_eng(x) = sum(ind(x))
            grad = Zygote.gradient(get_eng, v)[1]
            if !isnothing(model.yt)
                grad .*= model.yt.scale
            end

            # Scale back to the original feature vectors
            if !isnothing(model.xt)
                grad[end-nf3-nf2+1:end, :] ./= model.xt.scale
            end
            g3 = grad[end-nf3+1:end, :]
            g2 = grad[end-nf3-nf2+1:end-nf3, :]
            calculate_forces!(wt, g2, g3)
            # Weight since the energies are linear combinations
            force_buf .+= wt.forces * weight
            stress_buf .+= wt.stress * weight
        end
        # Set the forces
        wt.stress .= stress_buf
        wt.forces .= force_buf
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
Calculate energy, force and stress
"""
function site_energies(m::ModelEnsemble, cw::CellWorkSpace)
    # Ensure the input vector is in the right size
    if isnothing(m.xt) || (m.xt.len == size(cw.v, 1))
        v = cw.v
    else
        v = cw.v[end-m.xt.len + 1:end, :]
    end
    site_energies(m, v)
end

function site_energies(m::ModelEnsemble, inp::AbstractMatrix)
    if isnothing(m.xt)
        x = inp
    else
        x = StatsBase.transform(m.xt, inp)
    end
    eng = sum(m(x) * w for (m, w) in zip(m.models, m.weight))
    if !isnothing(m.yt)
        out = StatsBase.reconstruct(m.yt, reshape(eng, :, 1))
    else
        out = reshape(eng, :, 1)
    end
    out
end

total_energy(m::ModelEnsemble, inp::AbstractMatrix) = sum(site_energies(m, inp))
total_energy(m::ModelEnsemble, cw::CellWorkSpace) = sum(site_energies(m, cw))

"""
A calculator to compute energy/forces/stress of the structure
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
function calculate!(m::CellCalculator;forces=true, rebuild_nl=true)
    # Detect any change of the structure
    if any(m.last_cellmat .!= cellmat(m.workspace.cell)) || any(m.last_pos .!= positions(m.workspace.cell))
        calculate!(m.workspace, m.modelensemble;forces, rebuild_nl)
    else
        return get_energy(m.workspace), get_forces(m.workspace), get_stress(m.workspace)
    end
    # Update the last calculated positions
    m.last_pos .= positions(m.workspace.cell)
    m.last_cellmat .= cellmat(m.workspace.cell)
    return get_energy(m.workspace), get_forces(m.workspace), get_stress(m.workspace)
end

# Getter for the energy/stress

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

get_positions(c::CellCalculator;make_copy=true) = get_positions(c.workspace; make_copy)
get_positions(c::CellWorkSpace;make_copy=true) = make_copy ? copy(positions(c.cell)) : positions(c.cell)

# Setter
CellBase.set_cellmat!(c::CellCalculator, cellmat) = CellBase.set_cellmat!(c.workspace.cell, cellmat)
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

"Filter for including lattice vectors in the optimisation"
struct VariableLatticeFilter
    calculator
    orig_lattice::Lattice
    lattice_factor::Float64
end

get_cell(c::CellCalculator) = c.workspace.cell
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


function get_positions(cf::VariableLatticeFilter)
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

    virial = (dgrad  \ (-vol .* stress))
    # Deformed forces
    atomic_forces = dgrad * forces

    out_forces = hcat(atomic_forces, virial ./ cf.lattice_factor)
    out_stress = -virial / vol
    out_forces, out_stress
end

get_forces(cf::VariableLatticeFilter;rebuild_nl=true) = _get_forces_and_stress(cf;rebuild_nl)[1]

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