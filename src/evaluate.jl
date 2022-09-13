#=
Support code for using the potentials for efficient energy/force/stress
calculations.
=#
using Zygote: gradient as zgradient
using Optim: LBFGS, optimize
using LineSearches: HagerZhang
using LinearAlgebra
import Base
import CellBase
import CellBase: rebuild!, update!
export rebuild!, update!

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
    forces = zeros(T, ndims, nat)
    stress = zeros(T, ndims, ndims)
    ForceBuffer(gvec, svec, stotv, forces, stress)
end

const WORKSPACE_T = Float64

"""
Combination of a cell, its associated neighbour list and feature vectors

This is to allow efficient re-calculation of the feature vectors without reallocating memory.
"""
struct CellWorkSpace{A, T, N, M}
    cell::Cell{T}
    nl::N
    cf::M
    # Feature vectors
    one_body::Matrix{A}
    two_body::Matrix{A}
    three_body::Matrix{A}
    v::Matrix{A}
    "Buffer for two body force calculations"
    two_body_fbuffer::ForceBuffer{A}
    "Buffer for three body force calculations"
    three_body_fbuffer::ForceBuffer{A}
    "Forces"
    forces::Matrix{A}
    "Stress"
    stress::Matrix{A}
    "atomic_energy"
    eng::Vector{A}
    "flat to ignore one body interactions or not"
    ignore_one_body::Bool
end

get_cell(cw::CellWorkSpace) = cw.cell

"Number of features used in the CellWorkSpace"
nfeatures(cw::CellWorkSpace)  = size(cw.v, 1)

nbodyfeatures(cw::CellWorkSpace, nbody) = nbodyfeatures(cw.cf, nbody)

function Base.show(io::IO, cw::CellWorkSpace)
    println(io, "CellWorkspace for: ")
    Base.show(io, cw.cell)
end

function Base.show(io::IO, m::MIME"text/plain", cw::CellWorkSpace)
    println(io, "CellWorkspace for: ")
    Base.show(io, m, cw.cell)
end


function CellWorkSpace{A}(cell::Cell;cf, rcut, nmax=500, savevec=true, ndims=3, ignore_one_body=true) where {A}
    nl = NeighbourList(cell, rcut, nmax;savevec)
    us = unique(atomic_numbers(cell))
    one_body = zeros(A, length(us), nions(cell))
    two_body = convert.(A, feature_vector(cf.two_body, cell;nl=nl))
    three_body = convert.(A, feature_vector(cf.three_body, cell;nl=nl))

    if ignore_one_body
        v = vcat(two_body, three_body)
    else
        v = vcat(one_body, two_body, three_body)
    end

    CellWorkSpace(cell, nl, cf, one_body, two_body, three_body, v, 
    ForceBuffer{eltype(two_body)}(size(two_body, 1), nions(cell)),
    ForceBuffer{eltype(three_body)}(size(three_body, 1), nions(cell)),
    zeros(eltype(two_body), ndims, nions(cell)),  # Forces 
    zeros(eltype(two_body), ndims, ndims),  # Stress
    zeros(eltype(two_body), nions(cell)),
    ignore_one_body,
    )
end

function CellWorkSpace(cell::Cell;cf, rcut, nmax=500, savevec=true, ndims=3, ignore_one_body=true)
    CellWorkSpace{WORKSPACE_T}(cell::Cell;cf, rcut, nmax, savevec, ndims, ignore_one_body)
end

CellBase.rebuild!(cw::CellWorkSpace) = CellBase.rebuild!(cw.nl, cw.cell)
CellBase.update!(cw::CellWorkSpace) = CellBase.update!(cw.nl, cw.cell)

"""
    update_feature_vector!(wt::CellWorkSpace)

Update the feature vectors after atomic displacements.
"""
function update_feature_vector!(wt::CellWorkSpace;rebuild_nl=true, gradients=true, global_minsep=0.01, maxvol=100)
    check_cell_volume(wt.cell;threshold=maxvol) || throw(ErrorException("The cell volume is unrealistically high!"))
    rebuild_nl ? rebuild!(wt) : update!(wt)
    check_global_minsep(wt.nl, global_minsep) || throw(ErrorException("There are atoms closer than $(global_minsep)!"))

    # Sanity check - do we have atoms squashed togeter?
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
    chainrule_forces!(forces, stress, b2, b3, g2::AbstractMatrix, g3::AbstractMatrix)

Calculate the forces given gradients of the total energy based on
dE / dF and dF / dxi

Accumulate the (weighted) results in the matrices passed.
Args:
    - b2: Buffer for two body derivatives with atomic positions
    - b3: Buffer for three body derivatives with atomic positions
    - g2: partial derivatives of the two-body features as input of the model
    - g3: partial derivatives of the three-body features as input of the model
"""
function chainrule_forces!(forces, stress, b2, b3, g2::AbstractMatrix, g3::AbstractMatrix;weight=1)
    gv2 = b2.gvec
    gv3 = b3.gvec
    sv2 = b2.stotv
    sv3 = b3.stotv
    # Propagate forces
    _force_update!(b2.forces, gv2, g2)
    _force_update!(b3.forces, gv3, g3)
    # Propagate stress
    _stress_update!(b2.stress, sv2, g2)
    _stress_update!(b3.stress, sv3, g3)

    # Accumulate weighted forces
    forces .+= (b2.forces .+ b3.forces) .* weight
    stress .+= (b2.stress .+ b3.stress) .* weight 

    forces, stress
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
struct CellCalculator{A, T, N, M}
    workspace::CellWorkSpace{A, T, N, M}
    modelensemble::ModelEnsemble
    last_pos::Matrix{T}
    last_cellmat::Matrix{T}
    backprop_buffers::Vector
    g2::Matrix{A}
    g3::Matrix{A}
end

function CellCalculator(cw::CellWorkSpace, me::ModelEnsemble)
    # Setup the buffer for backpropagation
    buffer = []
    for model in me.models
        push!(buffer, ChainGradients(model, nions(get_cell(cw))))
    end

    nf2 = sum(nfeatures, cw.cf.two_body)
    nf3 = sum(nfeatures, cw.cf.three_body)
    g3 = zeros(eltype(cw.v), nf3, size(cw.v, 2))
    g2 = zeros(eltype(cw.v), nf2, size(cw.v, 2))
    CellCalculator(cw, me, similar(positions(cw.cell)), similar(cellmat(cw.cell)), buffer, g2, g3)
end

"""
Calculate energy only
"""
function calculate_energy!(wt::CellWorkSpace, me::ModelEnsemble; rebuild_nl=true)
    # Ensure that we scale the feature vectors according to the one stored in the ModelEnsemble
    apply_x_trans!(wt.v, me)
    # Compute atomic energies
    wt.eng .= site_energies(me, wt)[:]
    fill!(wt.stress, 0.)
    fill!(wt.forces, 0.)
end

"""
    calculate!(wt::CellWorkSpace, model::ModelEnsemble;forces=true, rebuild_nl=true)

Update the feature vectors and compute energy, forces and stresses (default).
"""
function calculate_energy_and_force!(cw::CellWorkSpace, me::ModelEnsemble, backprop_buffers, g2, g3; rebuild_nl=true)

    # Calling this should zero the stress and force matrices
    calculate_energy!(cw, me;rebuild_nl)

    # Obtain the gradients for each individual model
    for (model, weight, buffer) in zip(me.models, me.weight, backprop_buffers)
        calculate_forces!(cw, g2, g3, buffer, model, cw.v;xt=me.xt, yt=me.yt, weight=weight)
    end
    # Stress needs to be divided by the volume
    cw.stress ./= volume(get_cell(cw))
end

function calculate_forces!(cw::CellWorkSpace, g2, g3, buffer::ChainGradients, model::Chain, v::AbstractMatrix;yt, xt, weight=1)
    # Compute gradient from the model
    forward!(buffer, model, v)
    # Back propagate can skip computing the gradients of weight and bias
    backward!(buffer, model;gu=one(eltype(v)), weight_and_bias=false)
    grad = input_gradient(buffer.layers[1])  # For the total force

    # Scale the gradient
    if !isnothing(yt)
        grad .*= yt.scale
    end

    # Scale back to the original feature vectors
    if !isnothing(xt)
        grad ./= xt.scale
    end

    nf3 = size(g3, 1)
    nf2 = size(g2, 1)
    # Locate the gradients for 2-body and 3-body terms
    g3 .= grad[end-nf3+1:end, :]
    g2 .= grad[end-nf3-nf2+1:end-nf3, :]

    # Compute the forces from chain rule
    chainrule_forces!(cw.forces, cw.stress, cw.two_body_fbuffer, cw.three_body_fbuffer, g2, g3;weight)
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
        update_feature_vector!(calc.workspace;rebuild_nl, gradients=forces)
        if forces
            calculate_energy_and_force!(calc.workspace, calc.modelensemble, calc.backprop_buffers, calc.g2, calc.g3;rebuild_nl)
        else
            calculate_energy!(calc.workspace, calc.modelensemble;rebuild_nl)
        end

        calc.last_pos .= positions(calc.workspace.cell)
        calc.last_cellmat .= cellmat(calc.workspace.cell)
    end
    return get_energy(calc.workspace), get_forces(calc.workspace), get_stress(calc.workspace)
end

# Getter for the energy/stress

"Return the underlying Cell object"
get_cell(c::CellCalculator) = c.workspace.cell

"Get the forces"
get_forces(c::CellWorkSpace;make_copy=true) = make_copy ? copy(c.forces) : c.forces

"Get the forces"
function get_forces(m::CellCalculator;make_copy=true)
    calculate!(m)
    get_forces(m.workspace;make_copy)
end

"Get the energy"
get_energy(c::CellWorkSpace) = sum(c.eng)
function get_energy(m::CellCalculator)
    calculate!(m)
    get_energy(m.workspace)
end

"Get the stress tensor in native units"
get_stress(c::CellWorkSpace;make_copy=true) = make_copy ? copy(c.stress) : c.stress

"Get the stress tensor in native units"
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


"""
    _get_forces_and_stress(cf::VariableLatticeFilter;rebuild_nl)

Construct force and stress for the filter object
"""
function _get_forces_and_stress(cf::VariableLatticeFilter;rebuild_nl)
    calculate!(cf.calculator;forces=true, rebuild_nl) 
    dgrad = deformgradient(cf)
    vol = volume(get_cell(cf))
    stress = get_stress(cf.calculator;make_copy=false)
    forces = get_forces(cf.calculator;make_copy=false)

    virial = (dgrad  \ (vol .* stress))
    # Deformed forces
    atomic_forces = dgrad * forces

    out_forces = hcat(atomic_forces, virial ./ cf.lattice_factor)
    out_stress = -virial / vol
    out_forces, out_stress
end

"""
Forces including the stress contributions that is consistent with the augmented positions
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
    set_cellmat!(cell, new_dgrad * cellmat(cf.orig_lattice))

    # Set the positions
    cell.positions .= new_dgrad * pos
end

"Return the energy of the VariableLatticeFilter"
function get_energy(cf::VariableLatticeFilter;rebuild_nl=true)
    calculate!(cf.calculator;rebuild_nl)
    get_energy(cf.calculator)
end


"Covert eV/Å^3 to GPa"
eVAngToGPa(x) = 160.21766208 * x

"""
    get_pressure_gpa(vc::Union{VariableLatticeFilter, CellCalculator}) 

Return pressure in unit of GPa.
"""
function get_pressure_gpa(vc::Union{VariableLatticeFilter, CellCalculator}) 
    eVAngToGPa(tr(EDDP.get_stress(vc)) / 3.)
end


"""
    optimise_cell!(vc)

Optimise the cell with LBFGS from Optim. Collect the trajectory if requested.
Note that the trajectory is collected for all force evaluations and may not 
corresponds to the actual iterations of the underlying LBFGS iterations.
"""
function optimise_cell!(vc;show_trace=false, record_trajectory=false, stepmax=2.0, g_abstol=1e-6, f_reltol=0.0, successive_f_tol=2)
    p0 = get_positions(vc)[:]
    traj = []
    function fo(x, vc)
        set_positions!(vc, reshape(x, 3, :))
        get_energy(vc)
    end

    function go(x, vc)
        set_positions!(vc, reshape(x, 3, :))
        forces = get_forces(vc)
        # ∇E = -F
        forces .*= -1  
        # Collect the trajectory if requsted
        if record_trajectory
            cell = deepcopy(get_cell(vc))
            cell.metadata[:enthalpy] = get_energy(vc)
            cell.arrays[:forces] = get_forces(vc.calculator)
            push!(traj, cell)
        end
        forces
    end
    lbfgs = LBFGS(;linesearch = HagerZhang(;alphamax=stepmax))
    res = optimize(x -> fo(x, vc), x -> go(x, vc), p0, lbfgs, Optim.Options(;show_trace=show_trace, g_abstol, f_reltol, successive_f_tol); inplace=false)
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

"""
    check_cell_volume(cell)
Check if the volume of the cell is reasonable
"""
function check_cell_volume(cell; threshold=100)
    vol_per_atom = volume(cell) / nions(cell)
    vol_per_atom < 100
end