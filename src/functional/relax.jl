using LinearAlgebra
using StatsBase: mean
using GarishPrint
using ExtXYZ
#=
Relaxation of a structure
=#

using Printf
using Optim: optimize, LBFGS, BFGS
using Optim.LineSearches: BackTracking
using Configurations

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

"""
    _relax_hyper!(calc::AbstractCalc, opts::RelaxOption)

Perform hyperspace relaxation on a calculator with a bound cell.

Extracts model components from the calculator, adds extra dimensions,
builds a `MetaCalculator` with `ExtraDimensionPenaltyModifier` and
`CellRelaxationModifier`, runs `scheduled_penalty_relax!`, then does a
3D cleanup relaxation.

Supports `EDDPCalculator`, `NNCalc`, and `MetaCalculator` (extracts source).
"""
function _relax_hyper!(calc::AbstractCalc, opts::RelaxOption)
    relax_fn = _method_to_native_fn(opts.method)
    smat = external_stress_matrix(opts.external_pressure)

    if calc isa EDDPCalculator
        cell = get_cell(calc)
        hcell = CellBase.add_dimensions(deepcopy(cell), opts.hyper_ndim)
        P = smat[1, 1]
        metacalc = create_calculator(calc, hcell; modifiers=[
            ExtraDimensionPenaltyModifier(opts.hyper_alpha_start; start_dim=4),
            CellRelaxationModifier(hcell; external_pressure=smat),
        ])
        max_iter = _hyper_max_iterations(opts)
        scheduled_penalty_relax!(metacalc;
            schedule_every=opts.hyper_schedule_every,
            max_iterations=max_iter,
            alpha_start=opts.hyper_alpha_start,
            relax_fn,
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
    elseif calc isa MetaCalculator
        _relax_hyper_from_nncalc!(calc.source, opts)
    elseif calc isa NNCalc
        _relax_hyper_from_nncalc!(calc, opts)
    else
        error("Hyperspace relaxation not supported for calculator type $(typeof(calc))")
    end
end

function _relax_hyper_from_nncalc!(nncalc::NNCalc, opts::RelaxOption)
    relax_fn = _method_to_native_fn(opts.method)
    cell = get_cell(nncalc)

    hcell = CellBase.add_dimensions(deepcopy(cell), opts.hyper_ndim)
    source = NNCalc(hcell, nncalc.cf, nncalc.nninterface;
        nmax=nncalc.nl.nmax,
        core=nncalc.workspace.hardcore.core_repulsion,
        elemental_energies=nncalc.elemental_energies,
    )

    penalty = ExtraDimensionPenaltyModifier(opts.hyper_alpha_start; start_dim=4)
    smat = external_stress_matrix(opts.external_pressure)
    cell_relax = CellRelaxationModifier(hcell; external_pressure=smat)
    metacalc = MetaCalculator(source, [penalty, cell_relax])

    max_iter = _hyper_max_iterations(opts)
    scheduled_penalty_relax!(metacalc;
        schedule_every=opts.hyper_schedule_every,
        max_iterations=max_iter,
        alpha_start=opts.hyper_alpha_start,
        relax_fn,
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
    calc_3d = NNCalc(cell_3d,
        source_calc.cf,
        source_calc.nninterface;
        nmax=source_calc.nl.nmax,
        core=source_calc.workspace.hardcore.core_repulsion,
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
    (; method, energy_threshold, force_threshold, stress_threshold) = re.options
    (; external_pressure, relax_cell, keep_trajectory, verbose, maxstep, trajectory_file) = re.options
    (; iterations) = re.options
    trajectory = re.trajectory
    calc = re.calc

    if method == "tpsd"
        _method = TwoPointSteepestDescent(; maxstep)
    elseif method == "lbfgs"
        _method = LBFGS(; linesearch=BackTracking(; maxstep))
    elseif method == "bfgs"
        _method = BFGS(; linesearch=BackTracking(; maxstep))
    elseif method == "cg"
        _method = ConjugateGradient(; linesearch=BackTracking(maxstep=0.2))
    elseif method == "fire"
        _method = TwoPointSteepestDescent(; maxstep)
    else
        throw(KeyError("Unknown `method`: $(method)"))
    end

    smat = external_stress_matrix(external_pressure)

    # Initial position vector

    if relax_cell
        if is_variable_cell(calc)
            # MetaCalculator with CellRelaxationModifier — already handles
            # variable-cell positions; apply external pressure via modifier.
            mod = get_cell_relaxation_modifier(calc)
            if !isnothing(mod)
                mod.external_pressure .= smat
            end
            _calc = calc
        else
            # Wrap in MetaCalculator with CellRelaxationModifier
            _calc = MetaCalculator(calc, [CellRelaxationModifier(get_cell(calc); external_pressure=smat)])
        end
    else
        # Do nothing 
        _calc = calc
    end
    # Initial position
    p0 = get_positions(_calc)[:]

    "Energy"
    function fo(x, calc)
        set_positions!(calc, reshape(x, CellBase.n_dimensions(get_cell(calc)), :))
        eddp_energy(calc)
    end

    "Gradient"
    function go(x, calc)
        set_positions!(calc, reshape(x, CellBase.n_dimensions(get_cell(calc)), :))
        if keep_trajectory
            cell = deepcopy(get_cell(calc))
            cell.metadata[:energy] = eddp_energy(calc)
            cell.metadata[:force] = eddp_forces(calc)[:, 1:length(cell)]
            push!(trajectory, cell)
        end
        forces = eddp_forces(calc)
        # ∇E = -F
        forces .* -1
    end

    start_time = time()
    "Callback function to control the stopping of the optimisation"
    last_energy = eddp_energy(_calc)

    smax = 0.0
    fmax = 0.0
    de = 0.0
    converged = false

    """
        callback(x)

    Callback function to print relaxation information as well as checking the convergence
    as specified.
    """
    function callback(x)
        eng = eddp_energy(_calc)
        de = abs((eng - last_energy) / length(get_cell(_calc)))
        last_energy = eng
        forces = eddp_forces(_calc)
        fmax = maximum(norm.(eachcol(forces)))
        stress = eVAngToGPa.(optimizer_effective_stress(_calc))
        smax = maximum(abs.(stress))
        elapsed = time() - start_time

        if !isempty(trajectory_file)
            # Write trajectory file 
            xyzdict = ExtXYZ.write_dict(CellBase.remove_dimensions(get_cell(calc)))
            #xyzdict[:arrays][:forces] = forces
            ExtXYZ.write_frame(trajectory_file, xyzdict; append=true)
        end

        # Check for convergence criteria
        fmax < force_threshold ? fok = "T" : fok = "F"
        de < energy_threshold ? eok = "T" : eok = "F"
        smax < stress_threshold ? sok = "T" : sok = "F"

        # Print basic information
        if verbose
            if x.iteration == 0
                @printf "* Iter                    E               dE         |F|        Smax       Time         Conv\n"
                #        * Iter 0000: -7.85900821e+02  6.82121026e-14 6.63946e-03 4.20335e-02      0.222   |F|:T dE:T Smax:T
            end
            @printf "* Iter %04d: %15.8e %15.8e %10.5e %10.5e %10.3f   " x.iteration eng de fmax smax elapsed
            @printf "dE:%s |F|:%s Smax:%s\n" eok fok sok
        end


        # Break if the convergence criteria is reached
        if relax_cell
            if (fok == "T") && (eok == "T") && (sok == "T")
                converged = true
                return true
            end
        else
            if (fok == "T") && (eok == "T")
                converged = true
                return true
            end
        end
        return false
    end

    res = optimize(
        x -> fo(x, _calc),
        x -> go(x, _calc),
        p0,
        _method,
        Optim.Options(; callback, iterations);
        inplace=false,
    )

    if !relax_cell
        smax = -1.0
    end


    RelaxResult(re, de, fmax, smax, res.iterations, converged)
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

