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

High-level relaxation options used by `Relax`, `relax!`, RSS, and hot-AIRSS.

Unit convention:
- `force_threshold` and `maxstep` use eV/Å and Å.
- `stress_threshold` and `external_pressure` use GPa.
- Native optimizer kwargs passed through `optimizer_kwargs` keep native units;
  in particular `s_tol`/`ext_pressure` are not accepted there and must be set
  through the high-level fields above.
"""
@option mutable struct RelaxOption <: EDDPotentialsOption
    method::String = "tpsd"
    energy_threshold::Float64 = 1e-5
    force_threshold::Float64 = 1e-2
    stress_threshold::Float64 = 1e-1
    external_pressure::Vector{Float64} = [0.0]
    relax_cell::Bool = true
    keep_trajectory::Bool = false
    verbose::Bool = false
    iterations::Int = 1000
    maxstep::Float64 = 0.2
    maxstrain::Float64 = 0.02
    cell_damping::Float64 = 0.1
    optimizer_kwargs::Dict{String,Any} = Dict{String,Any}()
    step_constraints::Vector{Any} = Any[]
    trajectory_file::String = ""
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
    method_lc = lowercase(method)
    if method_lc == "tpsd"
        return opt_tpsd
    elseif method_lc == "fire"
        return opt_fire
    else
        throw(ArgumentError("Unknown relaxation method: $method"))
    end
end

const _RELAX_OPTION_ALIASES = Dict{Symbol,Symbol}(
    :f_tol => :force_threshold,
    :s_tol => :stress_threshold,
    :e_tol => :energy_threshold,
    :itermax => :iterations,
    :max_iterations => :iterations,
)

const _RELAX_METHOD_KWARG_ALIASES = Dict{Symbol,Symbol}(
    :alpha_tol => :α_tol,
    :nmin => :Nmin,
)

const _RELAX_METHOD_KWARGS = Set{Symbol}([
    :dt,
    :dtmax,
    :Nmin,
    :nmin,
    :finc,
    :fdec,
    :astart,
    :fa,
    :downhill_check,
    :α_tol,
    :alpha_tol,
])

const _COMMON_NATIVE_OPTIMIZER_KWARGS = Set{Symbol}([
    :f_tol,
    :s_tol,
    :e_tol,
    :itermax,
    :trace,
    :verbose,
    :maxstep,
    :maxstrain,
    :relax_cell,
    :ext_pressure,
    :cell_damping,
    :callback,
    :keep_trajectory,
    :trajectory,
    :step_constraints,
])

_kw_symbol(key::Symbol) = key
_kw_symbol(key::AbstractString) = Symbol(key)
_kw_symbol(key) = Symbol(key)

function _canonical_relax_kwargs(kwargs)
    data = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(kwargs))

    for (alias, canonical) in _RELAX_OPTION_ALIASES
        haskey(data, alias) || continue
        value = pop!(data, alias)
        if haskey(data, canonical) && data[canonical] != value
            throw(ArgumentError("Conflicting relaxation options `$alias` and `$canonical`"))
        end
        data[canonical] = value
    end

    method_kwargs = Dict{String,Any}()
    if haskey(data, :optimizer_kwargs)
        existing = pop!(data, :optimizer_kwargs)
        existing isa AbstractDict ||
            throw(ArgumentError("optimizer_kwargs must be a dictionary"))
        for (key, value) in pairs(existing)
            method_kwargs[string(_kw_symbol(key))] = value
        end
    end

    for key in collect(keys(data))
        key in _RELAX_METHOD_KWARGS || continue
        normalized = get(_RELAX_METHOD_KWARG_ALIASES, key, key)
        strkey = string(normalized)
        value = pop!(data, key)
        if haskey(method_kwargs, strkey) && method_kwargs[strkey] != value
            throw(ArgumentError("Conflicting optimizer_kwargs value for `$strkey`"))
        end
        method_kwargs[strkey] = value
    end

    if !isempty(method_kwargs)
        data[:optimizer_kwargs] = method_kwargs
    end

    (; data...)
end

function _normalized_method_kwargs(opts::RelaxOption)
    out = Dict{Symbol,Any}()
    for (key, value) in pairs(opts.optimizer_kwargs)
        sym = get(_RELAX_METHOD_KWARG_ALIASES, _kw_symbol(key), _kw_symbol(key))
        sym in _COMMON_NATIVE_OPTIMIZER_KWARGS &&
            throw(ArgumentError("`optimizer_kwargs` contains common relaxation option `$sym`; set it directly on RelaxOption"))
        if haskey(out, sym) && out[sym] != value
            throw(ArgumentError("Conflicting optimizer_kwargs value for `$sym`"))
        end
        out[sym] = value
    end
    out
end

function _optimizer_kwargs(
    opts::RelaxOption;
    itermax_override=nothing,
    verbose_override=nothing,
    trajectory=nothing,
    callback=nothing,
    relax_cell_override=nothing,
    ext_pressure_override=nothing,
    include_itermax=true,
    include_logging=true,
)
    method_lc = lowercase(opts.method)
    method_lc in ("tpsd", "fire") ||
        throw(ArgumentError("Unknown relaxation method: $(opts.method)"))

    verbose_value = verbose_override === nothing ? opts.verbose : verbose_override
    relax_cell_value = relax_cell_override === nothing ? opts.relax_cell : relax_cell_override
    ext_pressure_value = ext_pressure_override === nothing ?
                         _external_stress_matrix_internal(opts.external_pressure) :
                         ext_pressure_override

    kwargs = Dict{Symbol,Any}(
        :f_tol => opts.force_threshold,
        :s_tol => GPaToeVAng(opts.stress_threshold),
        :e_tol => opts.energy_threshold,
        :maxstep => opts.maxstep,
        :maxstrain => opts.maxstrain,
        :relax_cell => relax_cell_value,
        :ext_pressure => ext_pressure_value,
        :cell_damping => opts.cell_damping,
        :keep_trajectory => opts.keep_trajectory,
        :step_constraints => opts.step_constraints,
    )
    if include_itermax
        kwargs[:itermax] = itermax_override === nothing ? opts.iterations : itermax_override
    end
    if include_logging
        if method_lc == "tpsd"
            kwargs[:trace] = verbose_value ? 1 : 0
        else
            kwargs[:verbose] = verbose_value
        end
    end
    trajectory === nothing || (kwargs[:trajectory] = trajectory)
    callback === nothing || (kwargs[:callback] = callback)

    merge!(kwargs, _normalized_method_kwargs(opts))
    (; kwargs...)
end

"""
    _hyper_max_iterations(opts::RelaxOption)

Return the effective max iterations for hyperspace relaxation.
Uses `hyper_target_iterations` if set (>0), otherwise falls back to `iterations`.
"""
_hyper_max_iterations(opts::RelaxOption) = opts.hyper_target_iterations > 0 ? opts.hyper_target_iterations : opts.iterations

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
    smat = _external_stress_matrix_internal(opts.external_pressure)

    if calc isa MetaCalculator
        return _relax_hyper!(calc.source, opts)
    end

    calc isa EDDPCalculator || error(
        "Hyperspace relaxation not supported for calculator type $(typeof(calc))"
    )
    cell = get_cell(calc)
    hcell = CellBase.add_dimensions(deepcopy(cell), opts.hyper_ndim)
    P = smat[1, 1]
    metacalc = create_calculator(calc, hcell; modifiers=[
        ExtraDimensionPenaltyModifier(opts.hyper_alpha_start; start_dim=4),
    ])
    max_iter = _hyper_max_iterations(opts)
    parentmod = parentmodule(@__MODULE__)
    relax_kwargs = _optimizer_kwargs(opts;
        ext_pressure_override=smat,
        include_itermax=false,
        include_logging=false,
    )

    parentmod.scheduled_penalty_relax!(metacalc;
        schedule_every=opts.hyper_schedule_every,
        max_iterations=max_iter,
        alpha_start=opts.hyper_alpha_start,
        relax_fn,
        relax_kwargs,
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
    external_stress_matrix(vector::AbstractVector)

Expand a scalar or six-component stress vector into a 3x3 symmetric matrix.
This helper is unit-neutral. `RelaxOption.external_pressure` values are GPa
and are converted to native eV/Å³ by `_optimizer_kwargs`.
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

_external_stress_matrix_internal(vector::AbstractVector) =
    GPaToeVAng.(external_stress_matrix(vector))


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
    (; relax_cell) = re.options
    calc = re.calc

    relax_fn = _method_to_native_fn(re.options.method)
    optimizer_kwargs = _optimizer_kwargs(re.options;
        trajectory=re.options.keep_trajectory ? re.trajectory : nothing,
    )

    opt = relax_fn(calc; optimizer_kwargs...)

    E = eddp_energy(calc)
    F = _projected_forces(opt, calc)
    fmax = maximum(norm.(eachcol(F)))
    de = 0.0
    if isnan(de) || isinf(de)
        de = 0.0
    end

    smax = -1.0
    if relax_cell
        S = _projected_stress(opt, calc)
        smax = eVAngToGPa(maximum(abs.(S)))
    end

    converged(opt, calc) ? (opts = "Converged") : (opts = "Unconverged")
    RelaxResult(re, de, fmax, smax, opt.iteration, converged(opt, calc))
end

function (re::Relax)(;kwargs...) 
    normalized = _canonical_relax_kwargs(kwargs)
    for (key, value) in pairs(normalized)
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
    opts = RelaxOption(; method, relax_cell, _canonical_relax_kwargs(kwargs)...)
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
