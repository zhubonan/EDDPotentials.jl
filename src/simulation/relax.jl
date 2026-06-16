using LinearAlgebra
using StatsBase: mean
using GarishPrint
using ExtXYZ
#=
Relaxation of a structure
=#

using Printf
using Configurations

"""
    RelaxOption

High-level relaxation options used by `relax!` and RSS/link workflows.

`stress_threshold` uses the optimizer's internal stress unit, eV/Å³. Prefer
the user-facing `stress_threshold_gpa`.

`external_pressure` is deprecated and uses the optimizer's internal stress
unit, eV/Å³. Use `external_pressure_gpa` for hydrostatic external pressure in
GPa; positive values are compressive and negative values are tensile. Set only
one pressure option: using `external_pressure_gpa` together with a non-default
`external_pressure` is an error.
"""
@option mutable struct RelaxOption <: EDDPotentialsOption
    method::String = "tpsd"
    energy_threshold::Float64 = 1e-6
    convergence::String = "step_energy"
    force_threshold::Float64 = 1e-2
    stress_threshold::Float64 = GPaToeVAng(0.1)
    stress_threshold_gpa::Float64 = -1.0
    external_pressure::Vector{Float64} = [0.0]
    external_pressure_gpa::Union{Nothing,Float64} = nothing
    relax_cell::Bool = true
    keep_trajectory::Bool = false
    verbose::Bool = false
    iterations::Int = 1000
    maxstep::Float64 = 0.2
    max_cell_strain::Float64 = 0.0125
    cell_damping::Float64 = 1.0
    step_constraints::Vector{Any} = Any[]
    honor_cell_constraints::Bool = true
    symmetry::String = "auto"
    symmetry_tolerance::Float64 = 1e-5
    trajectory_file::String = ""
    trajectory_format::String = "csv_extxyz"
    trajectory_every::Int = 1
    hyper_ndim::Int = 0
    hyper_alpha_start::Float64 = 1e-4
    hyper_schedule_every::Int = 50
    hyper_target_iterations::Int = 0
    hyper_alpha_ramp_factor::Float64 = 5.0
    hyper_max_alpha::Float64 = 1.0
    hyper_ramp_alpha::Bool = true
    hyper_pos_tol::Float64 = 1e-4
    hyper_max_vol_change::Float64 = 3.0
    hyper_extra_dim_tol::Float64 = 0.1
    hyper_dynamic_alpha::Bool = false
    hyper_force_ratio::Float64 = 2.0
    hyper_phase1_alpha::Float64 = 1e-4
    hyper_phase1_fraction::Float64 = 0.5
end

struct Relax{T}
    calc::T
    options::RelaxOption
    trajectory::Vector{Any}
end


Relax(calc, options=RelaxOption()) = Relax(calc, options, [])

function Base.show(io::IO, text::MIME"text/plain", x::Relax)
    println(io, "Relax:")
    println(io, "Calc:")
    show(io, text, x.calc)
    println(io, "Options:")
    if isa(io, GarishPrint.GarishIO)
        pprint_struct(GarishPrint.GarishIO(io; include_defaults=true), text, x.options)
    else
        pprint_struct(
            GarishPrint.GarishIO(io; include_defaults=true, color=false),
            text,
            x.options,
        )
    end
end


struct RelaxResult
    relax::Relax
    dE::Float64
    fmax::Float64
    smax::Float64
    iterations::Int
    converged::Bool
end

function Base.show(io::IO, text::MIME"text/plain", x::RelaxResult)
    pprint_struct(GarishPrint.GarishIO(io; color=false, include_defaults=true), text, x)
end

precompile(Base.show, (Base.TTY, MIME"text/plain", RelaxResult))
precompile(Base.show, (Base.TTY, MIME"text/plain", Relax))

"""
    _method_to_native_fn(method::String)

Map a RelaxOption method string to a native optimizer function for hyperspace relaxation.
"""
function _method_to_native_fn(method::String)
    if method in ("tpsd", "TPSD")
        return opt_tpsd
    elseif method in ("fire", "FIRE")
        return opt_fire
    else
        return opt_tpsd
    end
end

"""
    _hyper_max_iterations(opts::RelaxOption)

Return the effective max iterations for hyperspace relaxation.
Uses `hyper_target_iterations` if set (>0), otherwise falls back to `iterations`.
"""
_hyper_max_iterations(opts::RelaxOption) = opts.hyper_target_iterations > 0 ? opts.hyper_target_iterations : opts.iterations

const _SYMMETRY_MODES = ("auto", "on", "off")

function _normalize_symmetry_mode(mode::AbstractString)
    normalized = lowercase(strip(mode))
    normalized in _SYMMETRY_MODES ||
        throw(ArgumentError("RelaxOption.symmetry must be one of auto, on, or off; got `$mode`"))
    normalized
end

_cell_wants_symmetry(cell) = get(cell.metadata, :symmetry_generate, false) === true

_stress_threshold_internal(opts::RelaxOption) =
    opts.stress_threshold_gpa >= 0 ? GPaToeVAng(opts.stress_threshold_gpa) : opts.stress_threshold

function _external_pressure_internal(opts::RelaxOption)
    if opts.external_pressure_gpa !== nothing
        if opts.external_pressure != [0.0]
            throw(ArgumentError("Set only one of RelaxOption.external_pressure_gpa or external_pressure. external_pressure uses internal eV/Å³ units."))
        end
        return Float64[GPaToeVAng(opts.external_pressure_gpa)]
    end
    if opts.external_pressure != [0.0]
        Base.depwarn("RelaxOption.external_pressure is deprecated because it uses internal eV/Å³ units. Use external_pressure_gpa for hydrostatic pressure in GPa; positive values are compressive and negative values are tensile.", :external_pressure)
    end
    opts.external_pressure
end

_external_stress_matrix(opts::RelaxOption) = external_stress_matrix(_external_pressure_internal(opts))

function _normalize_relax_convergence_mode(mode::AbstractString)
    _normalize_convergence_mode(mode)
end

const _TRAJECTORY_FORMATS = ("csv_extxyz", "csv", "extxyz")

function _normalize_trajectory_format(format::AbstractString)
    normalized = lowercase(strip(format))
    normalized in _TRAJECTORY_FORMATS ||
        throw(ArgumentError("RelaxOption.trajectory_format must be one of csv_extxyz, csv, or extxyz; got `$format`"))
    normalized
end

_trajectory_enabled(opts::RelaxOption) = opts.keep_trajectory || !isempty(strip(opts.trajectory_file))

function _trajectory_stem(path::AbstractString)
    stem = strip(path)
    for suffix in (".relax.extxyz", ".relax.csv", ".extxyz", ".csv")
        if endswith(stem, suffix)
            return stem[1:(lastindex(stem)-length(suffix))]
        end
    end
    stem
end

function _trajectory_paths(path::AbstractString, format::AbstractString)
    stem = _trajectory_stem(path)
    paths = Dict{Symbol,String}()
    if format in ("csv_extxyz", "csv")
        paths[:csv] = "$(stem).relax.csv"
    end
    if format in ("csv_extxyz", "extxyz")
        paths[:extxyz] = "$(stem).relax.extxyz"
    end
    paths
end

function _annotate_relax_metadata!(cell, result::RelaxResult, opt=nothing)
    cell.metadata[:relax_iterations] = result.iterations
    cell.metadata[:relax_converged] = result.converged
    cell.metadata[:relax_stop_reason] = result.converged ? "converged" : "iteration_limit"
    cell.metadata[:relax_fmax] = result.fmax
    cell.metadata[:relax_smax_gpa] = result.smax
    if opt !== nothing && hasproperty(opt, :step_energy_de)
        cell.metadata[:relax_step_energy] = opt.step_energy_de
        cell.metadata[:relax_derun] = opt.step_energy_derun
    else
        cell.metadata[:relax_step_energy] = result.dE
        cell.metadata[:relax_derun] = result.dE > 0 ? log10(result.dE) : -Inf
    end
    info = get(cell.metadata, :info, nothing)
    if !(info isa Dict)
        info = Dict{Any,Any}()
        cell.metadata[:info] = info
    end
    info[:relax_iterations] = cell.metadata[:relax_iterations]
    info[:relax_converged] = cell.metadata[:relax_converged]
    info[:relax_stop_reason] = cell.metadata[:relax_stop_reason]
    info[:relax_fmax] = cell.metadata[:relax_fmax]
    info[:relax_smax_gpa] = cell.metadata[:relax_smax_gpa]
    info[:relax_step_energy] = cell.metadata[:relax_step_energy]
    info[:relax_derun] = cell.metadata[:relax_derun]
    cell
end

function _write_relax_trajectory!(re::Relax, opts::RelaxOption)
    isempty(re.trajectory) && return nothing
    isempty(strip(opts.trajectory_file)) && return nothing
    format = _normalize_trajectory_format(opts.trajectory_format)
    paths = _trajectory_paths(opts.trajectory_file, format)
    if haskey(paths, :csv)
        _write_relax_csv(paths[:csv], re.trajectory)
    end
    if haskey(paths, :extxyz)
        _write_relax_extxyz(paths[:extxyz], re.trajectory, get_cell(re.calc))
    end
    nothing
end

function _write_relax_csv(path::AbstractString, frames)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "iteration,energy,enthalpy,fmax,smax_gpa,step_energy_de,step_energy_derun,step_norm,max_cell_strain,optimizer_step,convergence,final")
        for frame in frames
            @printf(io,
                "%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%s,%d\n",
                frame.iteration,
                frame.energy,
                frame.enthalpy,
                frame.fmax,
                frame.smax_gpa,
                frame.step_energy_de,
                frame.step_energy_derun,
                frame.step_norm,
                frame.max_cell_strain,
                frame.optimizer_step,
                frame.convergence,
                frame.is_final ? 1 : 0)
        end
    end
    path
end

_xyz_string(x) = replace(string(x), "\"" => "\\\"")

function _frame_matrix3(mat)
    out = zeros(Float64, 3, 3)
    if mat !== nothing
        out .= mat[1:3, 1:3]
    end
    out
end

function _write_relax_extxyz(path::AbstractString, frames, cell)
    mkpath(dirname(path))
    atom_species = string.(species(cell))
    nat = length(atom_species)
    open(path, "w") do io
        for frame in frames
            pos = zeros(Float64, 3, nat)
            dpos = min(3, size(frame.positions, 1))
            pos[1:dpos, :] .= frame.positions[1:dpos, :]
            frc = zeros(Float64, 3, nat)
            if frame.forces !== nothing
                dfrc = min(3, size(frame.forces, 1))
                frc[1:dfrc, :] .= frame.forces[1:dfrc, :]
            end
            H = frame.cell_matrix === nothing ? _frame_matrix3(cellmat(cell)) : _frame_matrix3(frame.cell_matrix)
            lattice = join(vec(transpose(H)), " ")
            label = get(cell.metadata, :label, "")
            label_part = isempty(string(label)) ? "" : " label=\"$(_xyz_string(label))\""
            println(io, nat)
            println(io,
                "Lattice=\"$lattice\" Properties=species:S:1:pos:R:3:force:R:3 energy=$(frame.energy) enthalpy=$(frame.enthalpy) iteration=$(frame.iteration) fmax=$(frame.fmax) smax_gpa=$(frame.smax_gpa) step_energy_de=$(frame.step_energy_de) step_energy_derun=$(frame.step_energy_derun) step_norm=$(frame.step_norm) max_cell_strain=$(frame.max_cell_strain) optimizer_step=$(frame.optimizer_step) convergence=\"$(frame.convergence)\" final=$(frame.is_final ? "T" : "F")$(label_part)")
            for i in 1:nat
                @printf(io, "%s %.17g %.17g %.17g %.17g %.17g %.17g\n",
                    atom_species[i],
                    pos[1, i], pos[2, i], pos[3, i],
                    frc[1, i], frc[2, i], frc[3, i])
            end
        end
    end
    path
end

function _resolved_step_constraints(calc::AbstractCalc, opts::RelaxOption)
    mode = _normalize_symmetry_mode(opts.symmetry)
    constraints = Any[opts.step_constraints...]
    cell = get_cell(calc)
    if opts.honor_cell_constraints
        append!(constraints, _cell_step_constraints(cell))
    end
    mode == "off" && return constraints

    if mode == "auto" && !_cell_wants_symmetry(cell)
        return constraints
    end

    sym_cell = cell
    if size(CellBase.positions(cell), 1) > 3
        sym_cell = remove_dimensions(deepcopy(cell))
    end

    sym_constraint = SymmetryConstraint(sym_cell; symprec=opts.symmetry_tolerance)
    if sym_constraint.active
        push!(constraints, sym_constraint)
    elseif mode == "on"
        @warn "Symmetry-aware relaxation requested, but no nontrivial symmetry was detected."
    end
    constraints
end

"""
    _relax_hyper!(calc::AbstractCalc, opts::RelaxOption)

Perform hyperspace relaxation on a calculator with a bound cell.

Extracts model components from the calculator, adds extra dimensions,
builds a `MetaCalculator` with `ExtraDimensionPenaltyModifier`, runs
`scheduled_penalty_relax!`, then does a 3D cleanup relaxation.

Supports `EDDPCalculator` and `MetaCalculator` (extracts source).
"""
function _relax_hyper!(calc::AbstractCalc, opts::RelaxOption)
    relax_fn = _method_to_native_fn(opts.method)
    smat = _external_stress_matrix(opts)

    if calc isa MetaCalculator
        return _relax_hyper!(calc.source, opts)
    end

    calc isa EDDPCalculator || error(
        "Hyperspace relaxation not supported for calculator type $(typeof(calc))"
    )
    cell = get_cell(calc)
    hcell = CellBase.add_dimensions(deepcopy(cell), opts.hyper_ndim)
    hcell.metadata[:symmetry_generate] = _cell_wants_symmetry(cell)
    P = smat[1, 1]
    metacalc = create_calculator(calc, hcell; modifiers=[
        ExtraDimensionPenaltyModifier(opts.hyper_alpha_start; start_dim=4),
    ])
    max_iter = _hyper_max_iterations(opts)
    hyper_step_constraints = _resolved_step_constraints(metacalc, opts)
    convergence = _normalize_relax_convergence_mode(opts.convergence)
    parentmod = parentmodule(@__MODULE__)
    parentmod.scheduled_penalty_relax!(metacalc;
        schedule_every=opts.hyper_schedule_every,
        max_iterations=max_iter,
        alpha_start=opts.hyper_alpha_start,
        relax_fn,
        relax_kwargs=(
            relax_cell=opts.relax_cell,
            ext_pressure=smat,
            f_tol=opts.force_threshold,
            s_tol=_stress_threshold_internal(opts),
            e_tol=opts.energy_threshold,
            convergence,
            maxstep=opts.maxstep,
            max_cell_strain=opts.max_cell_strain,
            cell_damping=opts.cell_damping,
            step_constraints=hyper_step_constraints,
        ),
        alpha_ramp_factor=opts.hyper_alpha_ramp_factor,
        max_alpha=opts.hyper_max_alpha,
        pos_tol=opts.hyper_pos_tol,
        ramp_alpha=opts.hyper_ramp_alpha,
        max_vol_change=opts.hyper_max_vol_change,
        extra_dim_tol=opts.hyper_extra_dim_tol,
        dynamic_alpha=opts.hyper_dynamic_alpha,
        force_ratio=opts.hyper_force_ratio,
    )
    _relax_hyper_finalize!(metacalc, opts)
end

function _relax_hyper_finalize!(metacalc, opts)
    hcell = get_cell(metacalc)
    cell_3d = remove_dimensions(hcell)
    positions(cell_3d) = positions(cell_3d)[1:3, :]

    if opts.verbose
        pos = positions(hcell)
        D = size(pos, 1)
        nat = size(pos, 2)
        avg_extra = D > 3 ? mean(norm(pos[4:end, a]) for a in 1:nat) : 0.0
        @printf("\n╔══ 3D cleanup  ⟨|extra|⟩=%.4f ══╗\n", avg_extra)
    end

    source_calc = metacalc isa MetaCalculator ? metacalc.source : metacalc
    nmax_val = source_calc isa EDDPCalculator ? source_calc.nmax : source_calc.nl.nmax
    core_val = source_calc isa EDDPCalculator ? source_calc.core_repulsion :
        source_calc.workspace.hardcore.core_repulsion
    calc_3d = EDDPCalculator(cell_3d,
        source_calc.cf,
        source_calc.nninterface;
        nmax=nmax_val,
        core_repulsion=core_val,
        elemental_energies=source_calc.elemental_energies,
    )

    cleanup_opts = deepcopy(opts)
    cleanup_opts.hyper_ndim = 0
    cleanup_opts.relax_cell = false
    re = Relax(calc_3d, cleanup_opts)
    try
        return relax!(re)
    catch
        re2 = Relax(calc_3d, cleanup_opts)
        return RelaxResult(re2, 0.0, 0.0, -1.0, 0, false)
    end
end

"""
    stress_matrix(vector=AbstractVector)

Build a external stress matrix from vector inputs.
"""
function external_stress_matrix(vector::AbstractVector)
    @assert length(vector) in [6, 1] "Expect a vector with length 6 or 1"
    if length(vector) == 1
        a = vector[1]
        return diagm([a, a, a])
    elseif length(vector) == 6
        a, b, c, d, e, f = vector
        return [
            a b c
            b d e
            c e f
        ]
    end
end
external_pressure_matrix(x::Real) = external_stress_matrix([x])


"""
    multirelax!(re::Relax;itermax=2000, restartmax=5)

Perform relaxation and restart if it does not converge. This is useful when the
algorithm suffer from vanishing step sizes and a reset of the state of the optimiser
helps.
"""
function multirelax!(re::Relax; max_iter=2000, max_restart=5)
    local outcome
    itertotal = 0
    for _ = 1:max_restart
        outcome = relax!(re)
        itertotal += outcome.iterations
        if outcome.converged || itertotal >= max_iter
            break
        end
    end
    RelaxResult(
        outcome.relax,
        outcome.dE,
        outcome.fmax,
        outcome.smax,
        itertotal,
        outcome.converged,
    )
end


"""
    relax!(re::Relax)

Perform relaxation for a `Relax` object.
"""
function relax!(re::Relax)
    (; method, energy_threshold, convergence, force_threshold) = re.options
    (; relax_cell, verbose, maxstep, max_cell_strain, cell_damping, step_constraints) = re.options
    (; iterations) = re.options
    calc = re.calc

    smat = _external_stress_matrix(re.options)
    relax_fn = _method_to_native_fn(method)
    resolved_constraints = _resolved_step_constraints(calc, re.options)
    convergence = _normalize_relax_convergence_mode(convergence)
    _normalize_trajectory_format(re.options.trajectory_format)
    trajectory_every = max(re.options.trajectory_every, 1)
    capture_trajectory = _trajectory_enabled(re.options)

    opt = relax_fn(calc;
        f_tol=force_threshold,
        s_tol=_stress_threshold_internal(re.options),
        itermax=iterations,
        e_tol=energy_threshold,
        convergence,
        require_residual_convergence=true,
        maxstep=maxstep,
        max_cell_strain=max_cell_strain,
        cell_damping=cell_damping,
        trace=verbose ? 1 : 0,
        relax_cell=relax_cell,
        ext_pressure=smat,
        keep_trajectory=capture_trajectory,
        trajectory_every,
        step_constraints=resolved_constraints,
    )

    if capture_trajectory
        empty!(re.trajectory)
        append!(re.trajectory, opt.trajectory)
    end

    E = eddp_energy(calc)
    F = _projected_forces(opt, calc)
    fmax = maximum(norm.(eachcol(F)))
    de = hasproperty(opt, :step_energy_derun) ? 10.0^opt.step_energy_derun : 0.0
    if isnan(de) || isinf(de)
        de = 0.0
    end

    smax = -1.0
    if relax_cell
        S = _projected_stress(opt, calc)
        smax = eVAngToGPa(maximum(abs.(S)))
    end

    optimizer_converged = converged(opt, calc)
    residual_converged = fmax <= force_threshold
    if relax_cell
        residual_converged &= smax <= eVAngToGPa(_stress_threshold_internal(re.options))
    end
    result = RelaxResult(re, de, fmax, smax, opt.iteration, optimizer_converged && residual_converged)
    _annotate_relax_metadata!(get_cell(calc), result, opt)
    _write_relax_trajectory!(re, re.options)
    result
end

function (re::Relax)(;kwargs...) 
    for (key, value) in kwargs
        setproperty!(re.options, key, value)
    end
    relax!(re)
end

"""
    relax!(calc::AbstractCalc)

Optimise the cell using the Optim interface. Collect the trajectory if requested.
Note that the trajectory is collected for all force evaluations and may not 
corresponds to the actual iterations of the underlying LBFGS iterations.

When `hyper_ndim > 0` is passed as a keyword argument, hyperspace relaxation
is performed instead of standard relaxation.
"""
function relax!(calc::AbstractCalc; multi=true, relax_cell=true, method="tpsd", kwargs...)
    opts = RelaxOption(; method, relax_cell, kwargs...)
    if opts.hyper_ndim > 0
        return _relax_hyper!(calc, opts)
    end
    relax = Relax(calc, opts)
    if multi
        multirelax!(relax)
    else
        relax!(relax)
    end
end
