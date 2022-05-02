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
end

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


function CellWorkSpace(cell::Cell;cf, rcut, nmax=100, savevec=true, ndims=3) 
    nl = NeighbourList(cell, rcut, nmax;savevec)
    us = unique(atomic_numbers(cell))
    one_body = zeros(length(us), nions(cell))
    two_body = feature_vector(cf.two_body, cell;nl=nl)
    three_body = feature_vector(cf.three_body, cell;nl=nl)
    CellWorkSpace(cell, nl, cf, one_body, two_body, three_body, vcat(one_body, two_body, three_body), 
    ForceBuffer{eltype(two_body)}(size(two_body, 1), nions(cell)),
    ForceBuffer{eltype(three_body)}(size(three_body, 1), nions(cell)),
    zeros(eltype(two_body), ndims, nions(cell)),
    zeros(eltype(two_body), ndims, ndims),
    zeros(eltype(two_body), nions(cell))
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
    l = size(wt.one_body, 1)
    wt.v[i: i+l-1, :] .= wt.one_body
    i += l
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
    v = StatsBase.transform(model.xt, wt.v[end-nf2-nf3+1:end, :])

    force_buf = similar(wt.forces)
    stress_buf = similar(wt.stress)
    fill!(force_buf, 0.)
    if forces
        # Obtain the graents for each individual model
        for (ind, weight) in zip(model.models, model.weight)
            get_eng(x) = sum(ind(x)) * model.yt.scale[1]
            grad = Zygote.gradient(get_eng, v)[1]

            # Scale back to the original feature vectors
            grad[end-nf3-nf2+1:end, :] ./= model.xt.scale
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
    if m.xt.len == size(cw.v, 1)
        v = cw.v
    else
        v = cw.v[end-m.xt.len + 1:end, :]
    end
    site_energies(m, v)
end

function site_energies(m::ModelEnsemble, inp::AbstractMatrix)
    x = StatsBase.transform(m.xt, inp)
    eng = sum(m(x) * w for (m, w) in zip(m.models, m.weight))
    StatsBase.reconstruct(m.yt, reshape(eng, :, 1))
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
        return get_energy(m), get_forces(m), get_stress(m)
    end
    # Update the last calculated positions
    m.last_pos .= positions(m.workspace.cell)
    m.last_cellmat .= cellmat(m.workspace.cell)
    return get_energy(m), get_forces(m), get_stress(m)
end

# Getter for the energy/stress

get_forces(c::CellWorkSpace) = c.forces
get_forces(m::CellCalculator) = get_forces(m.workspace)
get_energy(c::CellWorkSpace) = sum(c.eng)
get_energy(m::CellCalculator) = get_energy(m.workspace)
get_stress(c::CellWorkSpace) = c.stress
get_stress(c::CellCalculator) = get_stress(c.workspace)
get_positions(c::CellCalculator) = positions(c.workspace.cell)

# Setter
CellBase.set_cellmat!(c::CellCalculator, cellmat) = CellBase.set_cellmat!(c.workspace.cell, cellmat)
CellBase.set_positions!(c::CellCalculator, pos) = CellBase.set_positions!(c.workspace.cell, pos)