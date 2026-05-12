#=
Routines for performing random structure searching using models
=#

using Printf
using Configurations

"""
    RssPruneOption

Options for pruning random-structure-search output before writing structures.
Pruning is disabled by default.
"""
@option mutable struct RssPruneOption <: EDDPotentialsOption
    enabled::Bool = false
    pool_size::Int = 100
    keep_fraction::Float64 = 0.10
    dedup_tol::Float64 = 0.10
    fingerprint_lim::Float64 = 5.0
    min_stable_pool_size::Int = 50
    stable_window::Int = 20
    mean_abs_tol::Float64 = 1e-3
    median_abs_tol::Float64 = 1e-3
end

function _rss_candidate(cell)
    epa = Float64(cell.metadata[:enthalpy]) / length(cell)
    comp = reduce_composition(Composition(cell))
    (; cell, epa, comp)
end

function _rss_median(values)
    sorted = sort(collect(values))
    n = length(sorted)
    mid = n ÷ 2
    isodd(n) ? sorted[mid+1] : (sorted[mid] + sorted[mid+1]) / 2
end

_rss_pool_mean(pool) = sum(c -> c.epa, pool) / length(pool)
_rss_pool_median(pool) = _rss_median(c.epa for c in pool)

function _rss_should_flush(pool, stats_history, prune::RssPruneOption)
    length(pool) >= max(prune.pool_size, 1) && return (:pool_size, true)
    prune.stable_window > 0 || return (:none, false)
    length(pool) >= prune.min_stable_pool_size || return (:none, false)
    length(stats_history) > prune.stable_window || return (:none, false)

    current = stats_history[end]
    previous = stats_history[end-prune.stable_window]
    mean_stable = abs(current.mean - previous.mean) <= prune.mean_abs_tol
    median_stable = abs(current.median - previous.median) <= prune.median_abs_tol
    return (:stable_statistics, mean_stable && median_stable)
end

function _validate_rss_prune_option(prune::RssPruneOption)
    prune.pool_size >= 1 ||
        throw(ArgumentError("`rss_prune.pool_size` must be >= 1"))
    isfinite(prune.keep_fraction) && 0.0 < prune.keep_fraction <= 1.0 ||
        throw(ArgumentError("`rss_prune.keep_fraction` must be in (0, 1]"))
    isfinite(prune.dedup_tol) && prune.dedup_tol >= 0.0 ||
        throw(ArgumentError("`rss_prune.dedup_tol` must be finite and >= 0"))
    isfinite(prune.fingerprint_lim) && prune.fingerprint_lim > 0.0 ||
        throw(ArgumentError("`rss_prune.fingerprint_lim` must be finite and > 0"))
    prune.min_stable_pool_size >= 1 ||
        throw(ArgumentError("`rss_prune.min_stable_pool_size` must be >= 1"))
    prune.stable_window >= 0 ||
        throw(ArgumentError("`rss_prune.stable_window` must be >= 0"))
    isfinite(prune.mean_abs_tol) && prune.mean_abs_tol >= 0.0 ||
        throw(ArgumentError("`rss_prune.mean_abs_tol` must be finite and >= 0"))
    isfinite(prune.median_abs_tol) && prune.median_abs_tol >= 0.0 ||
        throw(ArgumentError("`rss_prune.median_abs_tol` must be finite and >= 0"))
    return prune
end

function _rss_is_unique_fingerprint(fingerprints, fp, prune::RssPruneOption)
    for ref in fingerprints
        if CellBase.fingerprint_distance(ref, fp; lim=prune.fingerprint_lim) < prune.dedup_tol
            return false
        end
    end
    return true
end

function _rss_select_pruned(pool, prune::RssPruneOption, remaining::Int)
    isempty(pool) && return (eltype(pool)[], 0)
    target = max(1, ceil(Int, prune.keep_fraction * length(pool)))
    quota = min(remaining, target, length(pool))
    quota <= 0 && return (eltype(pool)[], 0)

    ranked = sort(pool; by=c -> c.epa)
    selected = eltype(pool)[]
    fingerprints = Any[]
    duplicates = 0
    for cand in ranked
        fp = try
            CellBase.fingerprint(cand.cell)
        catch err
            @warn "Cannot fingerprint RSS candidate; keeping it as unique." exception=(err, catch_backtrace())
            push!(selected, cand)
            length(selected) >= quota && break
            continue
        end

        if _rss_is_unique_fingerprint(fingerprints, fp, prune)
            push!(selected, cand)
            push!(fingerprints, fp)
            length(selected) >= quota && break
        else
            duplicates += 1
        end
    end
    return selected, duplicates
end

function _rss_write_candidate(
    cand,
    outdir::AbstractString,
    packed::Bool,
    packed_outfile::Union{Nothing,String},
    mode::AbstractString,
)
    cell = cand.cell
    if packed
        write_res(packed_outfile, cell, mode)
    else
        label = string(get(cell.metadata, :label, get_label("rss")))
        write_res(joinpath(outdir, "$(label).res"), cell, "w")
    end
    return 1
end

function _rss_write_candidates!(
    candidates,
    outdir,
    packed,
    packed_outfile,
    mode,
    remaining,
)
    written = 0
    for cand in candidates
        written >= remaining && break
        written += _rss_write_candidate(cand, outdir, packed, packed_outfile, mode)
    end
    return written
end

function _rss_log_flush(pool, selected, duplicates, written, total_kept, reason)
    epas = [c.epa for c in pool]
    @info "RSS prune flush" reason=reason raw_candidates=length(pool) mean_epa=sum(epas) / length(epas) median_epa=_rss_median(epas) min_epa=minimum(epas) max_epa=maximum(epas) duplicates selected=length(selected) written=written total_kept=total_kept
end

"""
    build_and_relax(seedfile::AbstractString, ensemble, cf;timeout=10, nmax=500, pressure_gpa=0., 

Build structure using `buildcell` and return the relaxed structure.

- `init_structure_transform`: A function that transforms the initial structure. If `nothing` is returned, skip this generated structure.
"""
function build_and_relax(
    seedfile::AbstractString,
    ensemble,
    cf;
    timeout=10,
    nmax=1000,
    core_size=0.5,
    init_structure_transform=nothing,
    relax_option=RelaxOption(),
    elemental_energies=Dict{Symbol,Any}(),
    relax_mode="standard",
    hot_kwargs=NamedTuple(),
    seed_text_transform=nothing,
)
    cell = build_one(seedfile; timeout, init_structure_transform, seed_text_transform)

    _is_hyper = relax_option.hyper_ndim > 0 || relax_mode == "hyperspace"

    if _is_hyper
        ndim = relax_option.hyper_ndim > 0 ? relax_option.hyper_ndim : 3
        alpha_start = relax_option.hyper_alpha_start
        schedule_every = relax_option.hyper_schedule_every
        target_iterations = _hyper_max_iterations(relax_option)
        phase1_alpha = relax_option.hyper_phase1_alpha
        phase1_frac = relax_option.hyper_phase1_fraction
        pressure_gpa = relax_option.external_pressure[1]
        trace = relax_option.verbose

        cell = CellBase.add_dimensions(cell, ndim)
        source = EDDPCalculator(cell, cf, ensemble; nmax,
            core_repulsion=_make_core_repulsion(core_size), elemental_energies)
        penalty = ExtraDimensionPenaltyModifier(alpha_start; start_dim=4)
        p_clamped = GPaToeVAng(max(pressure_gpa, 0.1))
        P_ext = [p_clamped 0 0; 0 p_clamped 0; 0 0 p_clamped]

        phase1_iters = round(Int, target_iterations * phase1_frac)
        phase2_iters = target_iterations - phase1_iters

        trace && @printf("\n╔══ Phase 1: Position-only relaxation  α=%.1e  iters=%d  ndim=%d ══╗\n",
            phase1_alpha, phase1_iters, ndim)

        metacalc_pos = MetaCalculator(source, [penalty])
        set_modifier_parameter!(metacalc_pos, ExtraDimensionPenaltyModifier, :alpha, phase1_alpha)
        p1_ramp = relax_option.hyper_dynamic_alpha ? true : false
        scheduled_penalty_relax!(metacalc_pos;
            schedule_every, max_iterations=phase1_iters,
            alpha_start=phase1_alpha, relax_fn=opt_fire, trace,
            ramp_alpha=p1_ramp,
            alpha_ramp_factor=relax_option.hyper_alpha_ramp_factor,
            max_alpha=relax_option.hyper_max_alpha,
            dynamic_alpha=relax_option.hyper_dynamic_alpha,
            force_ratio=relax_option.hyper_force_ratio)

        trace && @printf("\n╔══ Phase 2: Variable-cell relaxation  α=%.1e  iters=%d ══╗\n",
            alpha_start, phase2_iters)

        metacalc_full = MetaCalculator(source, [penalty])
        scheduled_penalty_relax!(metacalc_full;
            schedule_every, max_iterations=phase2_iters,
            alpha_start, relax_fn=opt_fire, trace,
            relax_kwargs=(relax_cell=true, ext_pressure=P_ext),
            ramp_alpha=relax_option.hyper_ramp_alpha,
            alpha_ramp_factor=relax_option.hyper_alpha_ramp_factor,
            max_alpha=relax_option.hyper_max_alpha,
            pos_tol=relax_option.hyper_pos_tol,
            max_vol_change=relax_option.hyper_max_vol_change,
            extra_dim_tol=relax_option.hyper_extra_dim_tol,
            dynamic_alpha=relax_option.hyper_dynamic_alpha,
            force_ratio=relax_option.hyper_force_ratio)

        cell = remove_dimensions(get_cell(metacalc_full))
        positions(cell) = positions(cell)[1:3, :]

        trace && @printf("\n╔══ Phase 3: 3D cleanup ══╗\n")

        try
            re3d_opt = deepcopy(relax_option)
            re3d_opt.hyper_ndim = 0
            calc3d = EDDPCalculator(cell, cf, ensemble; nmax,
                core_repulsion=_make_core_repulsion(core_size), elemental_energies)
            re3d = Relax(calc3d, re3d_opt)
            return relax!(re3d)
        catch
        end
        calc = EDDPCalculator(cell, cf, ensemble; nmax,
            core_repulsion=_make_core_repulsion(core_size), elemental_energies)
        fallback_opt = deepcopy(relax_option)
        fallback_opt.hyper_ndim = 0
        re = Relax(calc, fallback_opt)
        return relax!(re)

    elseif relax_mode == "hot"
        calc = EDDPCalculator(cell, cf, ensemble; nmax,
            core_repulsion=_make_core_repulsion(core_size), elemental_energies)
        re = Relax(calc, relax_option)
        result = relax!(re)
        cycles = get(hot_kwargs, :n_rounds, 10)
        md_temp = get(hot_kwargs, :md_temperature, Float64(300.0))
        md_steps = get(hot_kwargs, :md_steps, 1000)
        md_dt = get(hot_kwargs, :md_timestep, Float64(0.5))
        relax_cell = get(hot_kwargs, :relax_cell, true)
        hot_opts = HotAIRSSOption(
            cycles, 200, 1e-2, 1e-1, 1e-5, "tpsd", relax_cell, [0.0],
            md_steps, md_temp, md_dt, "langevin", 0.01, 50.0, nothing, false)
        hot_airss!(re.calc, hot_opts)
        return result

    else  # standard
        calc = EDDPCalculator(cell, cf, ensemble; nmax,
            core_repulsion=_make_core_repulsion(core_size), elemental_energies)
        re = Relax(calc, relax_option)
        return relax!(re)
    end
end


"""
    run_rss(seedfile, ensemble, cf;max=1, outdir="./", ...)

Perform random structure searching using the seed file.
Glob expression is allowed for the `seedfile` argument to select random from a list of
seeds.

- `init_structure_transform`: A function that transforms the initial structure. If `nothing` is returned, skip this generated structure.
"""
function _run_rss(
    seedfile::Union{AbstractString,Vector},
    ensemble::AbstractNNInterface,
    cf::CellFeature;
    seedfile_weights::Vector{Float64}=[1.0],
    show_progress=false,
    max=1,
    outdir="./",
    packed=false,
    ensemble_std_max=-1.0,
    ensemble_std_min=-1.0,
    init_structure_transform=nothing,
    composition_engmin=Dict{Composition,Float64}(),
    eng_threshold=-1.0,
    niggli_reduce_output=true,
    max_err=10,
    pressure_gpa=0.001,
    pressure_gpa_range=nothing,
    relax_option=RelaxOption(),
    core_size=0.5,
    elemental_energies=Dict{Symbol,Any}(),
    relax_mode="standard",
    hot_kwargs=NamedTuple(),
    seed_text_transform=nothing,
    prune_option=RssPruneOption(),
    prune_random_composition=false,
)
    _relax_option = deepcopy(relax_option)
    nkept = 0

    isdir(outdir) || mkdir(outdir)
    packed_outfile = nothing
    if packed
        # Select the first resolved seed file and use it as the name
        name = _select_seed(seedfile, seedfile_weights)[2][1]
        label = EDDPotentials.get_label(EDDPotentials.stem(name))
        # Name of the packed out file
        packed_outfile = joinpath(outdir, "$(label).packed.res")
        mode = "a"
    else
        mode = "w"
    end
    if show_progress
        pmeter = Progress(max)
    end
    pruning_enabled = prune_option.enabled
    pruning_enabled && _validate_rss_prune_option(prune_option)
    prune_disabled_warned = false
    if pruning_enabled && prune_random_composition
        @warn "RSS pruning disabled because formula sampling/random-composition generation is active."
        pruning_enabled = false
        prune_disabled_warned = true
    end
    pool = []
    stats_history = NamedTuple[]
    pool_comp = nothing
    _pressure_gpa = pressure_gpa
    while nkept < max
        # Use randomly chosen pressure
        if pressure_gpa_range !== nothing
            _pressure_gpa =
                rand() * (pressure_gpa_range[2] - pressure_gpa_range[1]) +
                pressure_gpa_range[1]
        end
        # Select the actual seeds
        this_seed = _select_seed(seedfile, seedfile_weights)[1]
        # Apply the external pressure setting
        _relax_option.external_pressure = Float64[_pressure_gpa]
        # Build the random structure and relax it
        res = build_and_relax_one(
            this_seed,
            ensemble,
            cf;
            max_err,
            init_structure_transform,
            relax_option,
            core_size,
            elemental_energies,
            relax_mode,
            hot_kwargs,
            seed_text_transform,
        )

        calc = res.relax.calc

        # Check for ensemble error and act
        # This prunes structure that we are too confident, e.g. nothing to learn from
        if ensemble_std_min > 0.0 || ensemble_std_max > 0.0
            estd = eddp_energy_std(calc) / length(get_cell(calc))
            if ensemble_std_min > 0.0 && estd < ensemble_std_min
                @info "Ensemble standard deviation $(estd) is too small ($(ensemble_std_min))!"
                continue
            end
            if ensemble_std_max > 0.0 && estd > ensemble_std_max
                @info "Ensemble standard deviation $(estd) is too large ($(ensemble_std_max))!"
                continue
            end
        end

        # Check if the final energy is low enough
        # Use this with case as pathological structures may prevent normal
        # structures being accepted. Ideally use it with `ensemble_std_min`.
        if eng_threshold > 0
            if check_energy_threshold!(composition_engmin, calc, eng_threshold) == false
                continue
            end
        end

        # Update the label of the structure
        label = get_label(stem(this_seed))
        EDDPotentials.update_metadata!(calc, label)

        # Write output file
        cell = get_cell(calc)
        # Run niggli reduction - skip the loop if failed.
        if niggli_reduce_output
            try
                cell = niggli_reduce_cell(cell)
            catch err
                continue
            end
        end

        cand = _rss_candidate(cell)
        if pruning_enabled
            if pool_comp === nothing
                pool_comp = cand.comp
            elseif cand.comp != pool_comp
                if !prune_disabled_warned
                    @warn "RSS pruning disabled because multiple reduced compositions were detected in the candidate pool."
                    prune_disabled_warned = true
                end
                written = _rss_write_candidates!(
                    pool,
                    outdir,
                    packed,
                    packed_outfile,
                    mode,
                    max - nkept,
                )
                nkept += written
                if show_progress
                    for _ = 1:written
                        ProgressMeter.next!(pmeter)
                    end
                end
                empty!(pool)
                empty!(stats_history)
                pool_comp = nothing
                pruning_enabled = false
            end
        end

        if pruning_enabled
            push!(pool, cand)
            push!(stats_history, (; mean=_rss_pool_mean(pool), median=_rss_pool_median(pool)))
            reason, should_flush = _rss_should_flush(pool, stats_history, prune_option)
            if should_flush
                remaining = max - nkept
                selected, duplicates = _rss_select_pruned(pool, prune_option, remaining)
                written = _rss_write_candidates!(
                    selected,
                    outdir,
                    packed,
                    packed_outfile,
                    mode,
                    remaining,
                )
                nkept += written
                _rss_log_flush(pool, selected, duplicates, written, nkept, reason)
                if show_progress
                    for _ = 1:written
                        ProgressMeter.next!(pmeter)
                    end
                end
                empty!(pool)
                empty!(stats_history)
                pool_comp = nothing
            end
        elseif nkept < max
            written = _rss_write_candidate(cand, outdir, packed, packed_outfile, mode)
            nkept += written
            if show_progress
                ProgressMeter.next!(pmeter)
            end
        end
    end
end


"""
    build_and_relax_one(seedfile::AbstractString, ensemble, cf::CellFeature;)

Build and relax a single structure, ensure that the process *does* generate a new structure.
"""
function build_and_relax_one(
    seedfile::AbstractString,
    ensemble,
    cf::CellFeature;
    init_structure_transform=nothing,
    max_err=10,
    relax_option=RelaxOption(),
    core_size=0.5,
    elemental_energies=Dict{Symbol,Any}(),
    relax_mode="standard",
    hot_kwargs=NamedTuple(),
    seed_text_transform=nothing,
)
    nerr = 1
    local res
    while true
        try
            res = build_and_relax(
                seedfile,
                ensemble,
                cf;
                init_structure_transform,
                relax_option,
                core_size,
                elemental_energies,
                relax_mode,
                hot_kwargs,
                seed_text_transform,
            )
        catch err
            isa(err, InterruptException) && throw(err)
            if typeof(err) <: ProcessFailedException
                @warn " `buildcell` failed to make the structure"
            else
                @warn "relaxation errored with $err"
            end
            if nerr >= max_err
                @error "Maximum $(max_err) consecutive errors reached!"
                throw(err)
            end
            nerr += 1
            continue
        end

        break
    end
    res
end

"""
    check_energy_threshold!(composition_engmin, calc, threshold)

Check if the structure is too high in energy and update the per-composition lowest
energy seen so far.
"""
function check_energy_threshold!(composition_engmin, calc, threshold)
    # Is the energy low enough?
    epa = eddp_energy(calc) / length(get_cell(calc))
    reduced_comp = reduce_composition(Composition(get_cell(calc)))
    if reduced_comp in keys(composition_engmin)
        # Update the lowest energy seen so far
        if epa < composition_engmin[reduced_comp]
            composition_engmin[reduced_comp] = epa
        end

        # Reject the structure if the energy is too high
        if eng_threshold > 0 && epa > (composition_engmin[reduced_comp] + threshold)
            return false
        end
    else
        # No records so far - update the record
        composition_engmin[reduced_comp] = epa
    end
    return true
end
