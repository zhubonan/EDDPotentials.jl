#=
Various tool functions for workflow managements
=#

using CellBase: rattle!, reduce, Composition
using NLSolversBase
using Printf
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
    this_cell.metadata[:enthalpy] = eddp_enthalpy(vc)
    this_cell.metadata[:volume] = volume(this_cell)
    this_cell.metadata[:pressure] = eddp_pressure_gpa(vc)
    if !haskey(this_cell.metadata, :label)
        this_cell.metadata[:label] = label
    end
    symm = CellBase.get_international(this_cell, symprec)
    this_cell.metadata[:symm] = "($(symm))"
    # Write to the file
    vc
end

"""
    write_res(path, vc::AbstractCalc;symprec=1e-2, label="EDDPotentials")

Write structure as SHELX file.
"""
function write_res(path, vc::AbstractCalc; symprec=1e-2, label="EDDPotentials")
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

function ensure_dir(path::AbstractString)
    parent = dirname(path)
    if isempty(parent) || parent == "."
        isdir(path) || mkdir(path)
    else
        target = contains(basename(path), ".") ? parent : path
        isdir(target) || mkpath(target)
    end
    return path
end

const _ensure_dir = ensure_dir

mutable struct TrainingContext
    itf::AbstractNNInterface
    feature_spec
    elemental_energies::Dict{Symbol,Float64}
    checkpoint_every::Int
    checkpoint_prefix::AbstractString
    show_progress::Bool
    log_file::AbstractString
    rec::Vector{Tuple{Float64,Float64}}
    time_start::Float64
    last_time::Float64
    counter::Int
end

function TrainingContext(itf, feature_spec, elemental_energies, checkpoint_every, checkpoint_prefix, show_progress, log_file)
    TrainingContext(itf, feature_spec, elemental_energies, checkpoint_every, checkpoint_prefix, show_progress, log_file, Tuple{Float64,Float64}[], time(), time(), 1)
end

function progress_callback!(ctx::TrainingContext, rmse_train, rmse_val, extra_info="")
    if ctx.show_progress || !isempty(ctx.log_file)
        tnow = time()
        elapsed = tnow - ctx.time_start
        loop_time = tnow - ctx.last_time
        ctx.last_time = tnow
        line = @sprintf "Epoch: %d %3.3f %3.3f RMSE Train %10.5f eV | Val %10.5f eV%s\n" ctx.counter loop_time elapsed rmse_train rmse_val extra_info
        ctx.show_progress && print(line)
        isempty(ctx.log_file) || (open(ctx.log_file, "a") do f; write(f, line); end)
        flush(stdout)
    end
    push!(ctx.rec, (rmse_train, rmse_val))
    if ctx.checkpoint_every > 0 && !isempty(ctx.checkpoint_prefix) && ctx.feature_spec !== nothing && ctx.counter % ctx.checkpoint_every == 0
        ckpt_path = ensure_dir(@sprintf("%s_%04d.json", ctx.checkpoint_prefix, ctx.counter))
        save_model_artifact(ckpt_path, ctx.itf; feature_spec=ctx.feature_spec, elemental_energies=ctx.elemental_energies, metadata=Dict("epoch" => ctx.counter))
    end
    return
end

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
    cf = CellFeature(elem, p2=[a, 2a], p3=[], q3=[], rcut2=rc)
    model = LinearInterface([0, -2, 1.0] .* α)
    NNCalc(cell, cf, model)
end

_lj_inv_r_cutoff(r, rcut) = r <= rcut ? inv(r) : zero(r)
_lj_g_inv_r_cutoff(r, rcut) = r <= rcut ? -inv(r * r) : zero(r)
_lj_const_cutoff(r, rcut) = r <= rcut ? one(r) : zero(r)
_lj_g_const_cutoff(r, rcut) = zero(r)
_lj_linear_cutoff(r, rcut) = r <= rcut ? r : zero(r)
_lj_g_linear_cutoff(r, rcut) = r <= rcut ? one(r) : zero(r)

function _lj_normalize_elements(elements)
    sort(unique(map(Symbol, collect(elements))))
end

function _lj_pair_parameter(value::Number, ::Symbol, ::Symbol, ::AbstractString)
    value
end

function _lj_pair_parameter(values::AbstractDict, si::Symbol, sj::Symbol, name::AbstractString)
    if haskey(values, (si, sj)) && haskey(values, (sj, si))
        values[(si, sj)] == values[(sj, si)] || throw(
            ArgumentError(
                "Conflicting LJ parameter `$name` for pair ($(si), $(sj)) and its reverse.",
            ),
        )
    end
    if haskey(values, (si, sj))
        return values[(si, sj)]
    elseif haskey(values, (sj, si))
        return values[(sj, si)]
    end
    throw(ArgumentError("Missing LJ parameter `$name` for pair ($(si), $(sj))."))
end

function _lj_pair_parameter(value, ::Symbol, ::Symbol, name::AbstractString)
    throw(ArgumentError("Unsupported LJ parameter container for `$name`: $(typeof(value))."))
end

function _lj_pair_energy(r, sigma, epsilon; p_rep=12, p_att=6)
    4epsilon * ((sigma / r)^p_rep - (sigma / r)^p_att)
end

function _lj_pair_energy_derivative(r, sigma, epsilon; p_rep=12, p_att=6)
    rep = -p_rep * sigma^p_rep * r^(-(p_rep + 1))
    att = p_att * sigma^p_att * r^(-(p_att + 1))
    4epsilon * (rep + att)
end

function _lj_pair_linear_weights(sigma, epsilon, rcut; p_rep=12, p_att=6)
    vcut = _lj_pair_energy(rcut, sigma, epsilon; p_rep, p_att)
    dvcut = _lj_pair_energy_derivative(rcut, sigma, epsilon; p_rep, p_att)
    (
        2epsilon * sigma^p_rep,
        -2epsilon * sigma^p_att,
        (-vcut + rcut * dvcut) / 2,
        -dvcut / 2,
    )
end

"""
    lj_baseline_feature(elements; rcut, p_rep=12, p_att=6)

Construct an exact two-body feature basis for a force-shifted Lennard-Jones model.
"""
function lj_baseline_feature(elements; rcut, p_rep::Integer=12, p_att::Integer=6)
    rcut > 0 || throw(ArgumentError("`rcut` must be positive."))
    p_rep > 0 || throw(ArgumentError("`p_rep` must be positive."))
    p_att > 0 || throw(ArgumentError("`p_att` must be positive."))
    elems = _lj_normalize_elements(elements)
    two_body_features = TwoBodyFeature[]

    for si in elems
        for sj in elems
            push!(
                two_body_features,
                TwoBodyFeature(
                    _lj_inv_r_cutoff,
                    _lj_g_inv_r_cutoff,
                    (Int(p_rep),),
                    (si, sj),
                    rcut,
                ),
            )
            push!(
                two_body_features,
                TwoBodyFeature(
                    _lj_inv_r_cutoff,
                    _lj_g_inv_r_cutoff,
                    (Int(p_att),),
                    (si, sj),
                    rcut,
                ),
            )
            push!(
                two_body_features,
                TwoBodyFeature(
                    _lj_const_cutoff,
                    _lj_g_const_cutoff,
                    (1,),
                    (si, sj),
                    rcut,
                ),
            )
            push!(
                two_body_features,
                TwoBodyFeature(
                    _lj_linear_cutoff,
                    _lj_g_linear_cutoff,
                    (1,),
                    (si, sj),
                    rcut,
                ),
            )
        end
    end

    CellFeature(elems, Tuple(two_body_features), ())
end

"""
    lj_baseline_weights(elements; sigma=1.0, epsilon=1.0, rcut, p_rep=12, p_att=6)

Build the fixed `LinearInterface` weights for an exact force-shifted Lennard-Jones model.
"""
function lj_baseline_weights(elements; sigma=1.0, epsilon=1.0, rcut, p_rep::Integer=12, p_att::Integer=6)
    elems = _lj_normalize_elements(elements)
    weights = zeros(Float64, length(elems))

    for si in elems
        for sj in elems
            sigma_ij = _lj_pair_parameter(sigma, si, sj, "sigma")
            epsilon_ij = _lj_pair_parameter(epsilon, si, sj, "epsilon")
            append!(
                weights,
                Float64.(
                    _lj_pair_linear_weights(sigma_ij, epsilon_ij, rcut; p_rep, p_att),
                ),
            )
        end
    end
    weights
end

"""
    lj_baseline_model(elements; sigma=1.0, epsilon=1.0, rcut, p_rep=12, p_att=6)

Construct a fixed-weight linear model for a force-shifted Lennard-Jones baseline.
"""
function lj_baseline_model(elements; kwargs...)
    LinearInterface(lj_baseline_weights(elements; kwargs...))
end

"""
    lj_baseline_calc(cell::Cell; sigma=1.0, epsilon=1.0, rcut, p_rep=12, p_att=6, elemental_energies=Dict())

Construct an `NNCalc` that reproduces a force-shifted Lennard-Jones baseline.
"""
function lj_baseline_calc(
    cell::Cell;
    sigma=1.0,
    epsilon=1.0,
    rcut,
    p_rep::Integer=12,
    p_att::Integer=6,
    elemental_energies=Dict(),
)
    elems = _lj_normalize_elements(species(cell))
    cf = lj_baseline_feature(elems; rcut, p_rep, p_att)
    model = lj_baseline_model(elems; sigma, epsilon, rcut, p_rep, p_att)
    NNCalc(cell, cf, model; core=nothing, elemental_energies)
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
        update_feature_vector!(calc; rebuild_nl=true, compute_gradients=true)
        copy(calc.workspace.fvec)
    end
end

function _fd_features(calc, s)
    _alter_pos(calc, s) do _
        update_feature_vector!(calc; rebuild_nl=true, compute_gradients=true)
        copy(calc.workspace.fvec)
    end
end

function _fd_energy(calc, p)
    _alter_pos(calc, p) do _
        calc.param.energy_calculated = false
        eddp_energy(calc)
    end
end

function _fd_energy_vc(calc, p)
    _alter_pos_vc(calc, p) do _
        eddp_energy(calc)
    end
end

function _fd_strain(calc, p)
    _alter_strain(get_cell(calc), p) do _
        eddp_energy(calc)
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
function fd_desc(cf, cell, step=1e-6)

    fb = compute_fv_gv(cf, cell)
    gvec = fb.gvec # dFi/drj order

    # Recover the gradient in the dFj/dri format
    gvec_ideal = gvec_index_transfer(fb)

    diff = zeros(size(gvec, 1), size(gvec, 2), natoms(cell), natoms(cell))  # dFj/dri
    for iat = 1:natoms(cell)
        for dir = 1:size(cell.positions, 1)
            dcell = deepcopy(cell)
            dcell.positions[dir, iat] += step
            fb = compute_fv_gv(cf, dcell)
            fv1 = copy(fb.fvec)

            dcell = deepcopy(cell)
            dcell.positions[dir, iat] -= step
            fb = compute_fv_gv(cf, dcell)
            fv2 = copy(fb.fvec)
            diff[dir, :, :, iat] .= (fv1 - fv2) / step / 2
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
    fb = compute_fv_gv(cf, cell, core=core)
    gvec = fb.gvec # dFi/drj order
    # coefficients such that param * fvec = energies
    param = rand(1, size(fb.gvec, 2))
    gv = repeat(transpose(param), 1, natoms(cell))
    # Compute forces
    n1bd = one_body_offset(cf)
    _force_update!(fb, gv; offset=n1bd)
    _stress_update!(fb, gv; offset=n1bd)
    forces = copy(fb.tot_forces)

    # dE/dri
    diff = zeros(size(gvec, 1), natoms(cell))
    for iat = 1:natoms(cell)
        for dir = 1:size(cell.positions, 1)
            dcell = deepcopy(cell)
            dcell.positions[dir, iat] += 1e-6
            fb = compute_fv_gv(cf, dcell, core=core)
            # Compute the energy
            e1 = sum(param * fb.fvec)
            if core !== nothing
                e1 += sum(fb.hardcore.ecore)
            end

            dcell = deepcopy(cell)
            dcell.positions[dir, iat] -= 1e-6
            fb = compute_fv_gv(cf, dcell, core=core)
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
    stress_gradient(cf, cell, core_size=0, step=1e-6)

    Finite difference gradient of the energies with respect to cell deformations.
Returns the gradient from finite difference and analytical computation. A random linear model is
used for prediction.
"""
function stress_gradient(cf, cell, core_size=0, step=1e-6)

    if core_size != 0
        core = CoreRepulsion(core_size)
    else
        core = nothing
    end
    fb = compute_fv_gv(cf, cell)
    # coefficients such that param * fvec = energies
    param = rand(1, size(fb.gvec, 2))
    gv = repeat(transpose(param), 1, natoms(cell))
    # Compute forces
    n1bd = one_body_offset(cf)
    _force_update!(fb, gv; offset=n1bd)
    _stress_update!(fb, gv; offset=n1bd)
    stress = copy(fb.tot_stress)


    smat_orig = diagm([1.0, 1.0, 1.0])

    # dE/dri
    diff = zeros(Float64, 3, 3)
    for i = 1:3
        for j = 1:3
            dcell = deepcopy(cell)
            smat = copy(smat_orig)
            smat[i, j] += step
            set_cellmat!(dcell, smat * cellmat(dcell); scale_positions=true)
            fb = compute_fv_gv(cf, dcell)
            # Compute the energy
            e1 = sum(param * fb.fvec)
            if core !== nothing
                e1 += sum(fb.hardcore.ecore)
            end

            dcell = deepcopy(cell)
            smat = copy(smat_orig)
            smat[i, j] -= step
            set_cellmat!(dcell, smat * cellmat(dcell); scale_positions=true)
            fb = compute_fv_gv(cf, dcell)
            e2 = sum(param * fb.fvec)
            if core !== nothing
                e2 += sum(fb.hardcore.ecore)
            end

            # Compute the energy
            diff[i, j] = (e1 - e2) / (2 * step)
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


    forces = copy(eddp_forces(calc))
    stress = copy(eddp_stress(calc))
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
    vc = MetaCalculator(calc, [CellRelaxationModifier(get_cell(calc))])
    epos = get_positions(vc)
    eforce = copy(eddp_forces(vc))

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
