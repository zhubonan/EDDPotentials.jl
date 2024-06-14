#=
Various tool functions for workflow managements
=#

using CellBase: rattle!, reduce, Composition
using NLSolversBase
using ProgressMeter: @showprogress
import CellBase: write_res
using StatsBase
using Base.Threads
using JLD2
using Dates
using UUIDs

"""
    update_metadata!(vc::AbstractCalc, label;symprec=1e-2)

Update the metadata attached to a `Cell`` object
"""
function update_metadata!(vc::AbstractCalc, label; symprec=1e-2)
    this_cell = get_cell(vc)
    # Set metadata
    this_cell.metadata[:enthalpy] = get_enthalpy(vc)
    this_cell.metadata[:volume] = volume(this_cell)
    this_cell.metadata[:pressure] = get_pressure_gpa(vc)
    this_cell.metadata[:label] = label
    symm = CellBase.get_international(this_cell, symprec)
    this_cell.metadata[:symm] = "($(symm))"
    # Write to the file
    vc
end

"""
    write_res(path, vc::VariableCellCalc;symprec=1e-2, label="EDDPotentials")

Write structure in VariableCellCalc as SHELX file.
"""
function write_res(path, vc::VariableCellCalc; symprec=1e-2, label="EDDPotentials")
    update_metadata!(vc, label; symprec)
    write_res(path, get_cell(vc))
end



"""
Check if a feature vector already present in an array of vectors
"""
function is_unique_fvec(all_fvecs, fvec; tol=1e-2, lim=5)
    match = false
    for ref in all_fvecs
        dist = CellBase.fingerprint_distance(ref, fvec; lim)
        if dist < tol
            match = true
            break
        end
    end
    !match
end

ensure_dir(path) = isdir(path) || mkdir(path)

function get_label(seedname)
    dt = Dates.format(now(), "yy-mm-dd-HH-MM-SS")
    suffix = string(uuid4())[end-7:end]
    "$(seedname)-$(dt)-$(suffix)"
end

"Return the *stea* part of a file name"
stem(x::AbstractString) = splitext(splitpath(x)[end])[1]

swapext(fname, new) = splitext(fname)[1] * new



raw"""

Generate LJ like pair-wise interactions

```math
F = \alpha(-2f(r, rc)^a + f(r, rc)^{2a})

The equilibrium position is at ``r_c/2``.
```

Support only single a element for now.
"""
function lj_like_calc(cell::Cell; α=1.0, a=6, rc=3.0)
    elem = unique(species(cell))
    @assert length(elem) == 1 "Only works for single specie Cell for now."
    cf = EDDPotentials.CellFeature(elem, p2=[a, 2a], p3=[], q3=[], rcut2=rc)
    model = EDDPotentials.LinearInterface([0, -2, 1.0] .* α)
    EDDPotentials.NNCalc(cell, cf, model)
end


### Finite difference testing


function _alter_pos(f, cell, pos, args...)
    pb = get_positions(cell)
    set_positions!(cell, reshape(pos, size(get_positions(cell))...))
    #cell.positions[:] .= vec(pos)
    out = f((cell, pos, args...))
    set_positions!(cell, pb)
    return out
end

function _alter_pos_vc(f, vc, pos, args...)
    pb = get_positions(vc)
    set_positions!(vc, reshape(pos, size(pb)...))
    out = f((vc, pos, args...))
    set_positions!(vc, pb)
    return out
end

function _alter_strain(f, cell, s, args...)
    cb = get_cellmat(cell)
    pb = get_positions(cell)
    smat = diagm([1.0, 1.0, 1.0])
    smat[:] .+= s
    set_cellmat!(cell, smat * cb; scale_positions=true)
    out = f((cell, s, args...))
    set_cellmat!(cell, cb; scale_positions=true)
    set_positions!(cell, pb)
    out
end


function _fd_features_strain(calc, s)
    _alter_strain(calc, s) do x
        EDDPotentials.update_feature_vector!(calc; rebuild_nl=true, compute_gradients=true)
        copy(calc.workspace.fvec)
    end
end

function _fd_features(calc, s)
    _alter_pos(calc, s) do _
        EDDPotentials.update_feature_vector!(calc; rebuild_nl=true, compute_gradients=true)
        copy(calc.workspace.fvec)
    end
end

function _fd_energy(calc, p)
    _alter_pos(calc, p) do _
        calc.param.energy_calculated = false
        get_energy(calc)
    end
end

function _fd_energy_vc(calc, p)
    _alter_pos_vc(calc, p) do _
        calc.calc.param.energy_calculated = false
        eng = get_energy(calc)
        eng
    end
end

function _fd_strain(calc, p)
    _alter_strain(get_cell(calc), p) do _
        get_energy(calc)
    end
end



"""
    gvec_index_transfer(workspace)

Sum the components of the gradients collected from neighbour of each atom:
(ndim, nvec, nmax_neigh, natoms) -> (ndim, nvec, natoms, natoms)
"""
function gvec_index_transfer(workspace)
    gvec = workspace.gvec
    # Recover the gradient in the dFj/dri format
    gvec_ideal = zeros(size(gvec, 1), size(gvec, 2), size(gvec, 4), size(gvec, 4))
    for iat in axes(gvec, 4)
        for j = 1:workspace.gvec_nn[iat]
            # Transfor neighbour local index to atom index
            jat = workspace.gvec_index[j, iat]  # index of the atoms that has moved
            gvec_ideal[:, :, iat, jat] .+= gvec[:, :, j, iat]
        end
    end
    gvec_ideal
end


"""
    fd_desc(cf, cell)

    Compute the finite difference gradient of the features with respect to the positions of the atoms.
Return the gradient of the feature indexed by [dir, nf, j, i], where i is tha atom that is moved and
j is the gradient of the feature vector of the j atom as a result of the movement of i.
"""
function fd_desc(cf, cell)

    fb = EDDPotentials.compute_fv_gv(cf, cell)
    gvec = fb.gvec # dFi/drj order

    # Recover the gradient in the dFj/dri format
    gvec_ideal = gvec_index_transfer(fb)

    diff = zeros(size(gvec, 1), size(gvec, 2), natoms(cell), natoms(cell))  # dFj/dri
    for iat = 1:natoms(cell)
        for dir = 1:3
            dcell = deepcopy(cell)
            dcell.positions[dir, iat] += 1e-6
            fb = EDDPotentials.compute_fv_gv(cf, dcell)
            fv1 = copy(fb.fvec)

            dcell = deepcopy(cell)
            dcell.positions[dir, iat] -= 1e-6
            fb = EDDPotentials.compute_fv_gv(cf, dcell)
            fv2 = copy(fb.fvec)
            diff[dir, :, :, iat] .= (fv1 - fv2) / 2e-6
        end
    end

    return diff, gvec_ideal

end


# Test with final force creation
"""
    force_gradient(cf, cell)

    Finite difference gradient of the energies with respect to the positions of the atoms.
Returns the gradient from finite difference and analytical computation. A random linear model is
used for prediction.
"""
function force_gradient(cf, cell, core_size=0)

    if core_size != 0
        core = CoreRepulsion(core_size)
    else
        core = nothing
    end
    fb = EDDPotentials.compute_fv_gv(cf, cell, core=core)
    gvec = fb.gvec # dFi/drj order
    # coefficients such that param * fvec = energies
    param = rand(1, size(fb.gvec, 2))
    gv = repeat(transpose(param), 1, natoms(cell))
    # Compute forces
    EDDPotentials._force_update!(fb, gv; offset=length(cf.elements))
    EDDPotentials._stress_update!(fb, gv; offset=length(cf.elements))
    forces = copy(fb.tot_forces)

    # dE/dri
    diff = zeros(size(gvec, 1), natoms(cell))
    for iat = 1:natoms(cell)
        for dir = 1:3
            dcell = deepcopy(cell)
            dcell.positions[dir, iat] += 1e-6
            fb = EDDPotentials.compute_fv_gv(cf, dcell, core=core)
            # Compute the energy
            e1 = sum(param * fb.fvec)
            if core !== nothing
                e1 += sum(fb.hardcore.ecore)
            end

            dcell = deepcopy(cell)
            dcell.positions[dir, iat] -= 1e-6
            fb = EDDPotentials.compute_fv_gv(cf, dcell, core=core)
            e2 = sum(param * fb.fvec)
            if core !== nothing
                e2 += sum(fb.hardcore.ecore)
            end
            # Compute the energy
            diff[dir, iat] = (e1 - e2) / 2e-6

        end
    end
    return diff, forces
end



"""
    stress_gradient(cf, cell)

    Finite difference gradient of the energies with respect to cell deformations.
Returns the gradient from finite difference and analytical computation. A random linear model is
used for prediction.
"""
function stress_gradient(cf, cell, core_size=0)

    if core_size != 0
        core = CoreRepulsion(core_size)
    else
        core = nothing
    end
    fb = EDDPotentials.compute_fv_gv(cf, cell)
    # coefficients such that param * fvec = energies
    param = rand(1, size(fb.gvec, 2))
    gv = repeat(transpose(param), 1, natoms(cell))
    # Compute forces
    EDDPotentials._force_update!(fb, gv; offset=length(cf.elements))
    EDDPotentials._stress_update!(fb, gv; offset=length(cf.elements))
    stress = copy(fb.tot_stress)


    smat_orig = diagm([1.0, 1.0, 1.0])

    # dE/dri
    diff = zeros(Float64, 3, 3)
    for i = 1:3
        for j = 1:3
            dcell = deepcopy(cell)
            smat = copy(smat_orig)
            smat[i, j] += 1e-6
            set_cellmat!(dcell, smat * cellmat(dcell); scale_positions=true)
            fb = EDDPotentials.compute_fv_gv(cf, dcell)
            # Compute the energy
            e1 = sum(param * fb.fvec)
            if core !== nothing
                e1 += sum(fb.hardcore.ecore)
            end

            dcell = deepcopy(cell)
            smat = copy(smat_orig)
            smat[i, j] -= 1e-6
            set_cellmat!(dcell, smat * cellmat(dcell); scale_positions=true)
            fb = EDDPotentials.compute_fv_gv(cf, dcell)
            e2 = sum(param * fb.fvec)
            if core !== nothing
                e2 += sum(fb.hardcore.ecore)
            end

            # Compute the energy
            diff[i, j] = (e1 - e2) / 2e-6
        end
    end
    return diff, stress
end

function allclose(x, y; kwargs...)
    all(isapprox.(x, y; kwargs...))
end


"""
Perform finite difference tests on a given calculator.
"""
function test_finite_difference(calc::NNCalc)

    diff, grad = fd_desc(calc.cf, get_cell(calc))
    desc_ok = allclose(grad, diff, atol=1e-7)
    desc_ok || @warn "Finite difference gradient of the descriptor is not correct"


    forces = copy(EDDPotentials.get_forces(calc))
    stress = copy(EDDPotentials.get_stress(calc))
    # Test the total force
    p0 = get_positions(calc)
    od = OnceDifferentiable(x -> _fd_energy(calc, x), p0, _fd_energy(calc, p0))
    grad = NLSolversBase.gradient(od, p0)
    force_ok = allclose(grad, -forces, atol=1e-4)
    force_ok || @warn "Finite difference gradient of the total force is not correct"

    # Test the total stress
    s0 = zeros(3, 3)[:]
    od = OnceDifferentiable(
        x -> _fd_strain(calc, x),
        s0,
        _fd_strain(calc, s0),
        inplace=false,
    )
    grad = NLSolversBase.gradient(od, s0) ./ volume(get_cell(calc))
    stress_ok = allclose(grad, -vec(stress), atol=1e-3)
    stress_ok || @warn "finite difference gradient of the total stress is not correct"

    # Test wrapper
    # NOTE Somehow this is needed here - possible BUG?
    vc = EDDPotentials.VariableCellCalc(calc)
    epos = EDDPotentials.get_positions(vc)
    eforce = copy(EDDPotentials.get_forces(vc))

    od = OnceDifferentiable(
        x -> _fd_energy_vc(vc, x),
        epos,
        _fd_energy_vc(vc, epos);
        inplace=false,
    )
    grad = NLSolversBase.gradient(od, epos)
    vc_force_ok = allclose(grad, -eforce, atol=1e-3)
    vc_force_ok ||
        @warn "finite difference gradient of the total force is not correct with variable cell"

    vc_force_ok && force_ok && stress_ok && desc_ok
end
