using Printf
using StatsBase: mean

"""
    hyper_get_calc(cell, bu::Builder; alpha=1e-3) -> MetaCalculator

Construct a calculator for hyperdimensional relaxation. Wraps an
[`EDDPCalculator`](@ref) in a [`MetaCalculator`](@ref) with
`ExtraDimensionPenaltyModifier`. Cell relaxation is handled by the
optimizer via the `relax_cell` parameter.
"""
function hyper_get_calc(cell, bu::Builder; alpha=1e-3)
    source = EDDPCalculator(
        cell,
        bu.cf,
        load_ensemble(bu);
        nmax=500,
        core_repulsion=CoreRepulsion(bu.state.core_size),
        elemental_energies=bu.state.elemental_energies,
    )
    penalty = ExtraDimensionPenaltyModifier(alpha; start_dim=4)
    MetaCalculator(source, [penalty])
end

function bP(press_gpa)
    p = GPaToeVAng(max(press_gpa, 0.1))
    [p 0 0; 0 p 0; 0 0 p]
end

"""
    HyperRoundDiagnostics

Per-round diagnostics recorded during `scheduled_penalty_relax!`.

# Fields
- `round::Int`: Round number (1-based)
- `alpha_before::Float64`: Penalty coefficient at start of round
- `alpha_after::Float64`: Penalty coefficient set for next round
- `iterations_used::Int`: Iterations actually consumed this round
- `converged_early::Bool`: Whether the optimizer converged before `schedule_every`
- `max_total_force::Float64`: Maximum force norm across all dimensions and atoms
- `max_3d_force::Float64`: Maximum force norm in 3D subspace
- `max_extra_force::Float64`: Maximum force norm in extra dimensions
- `avg_extra_dim_norm::Float64`: Mean extra-dimensional coordinate magnitude across atoms
- `extra_norm::Float64`: Maximum extra-dimensional coordinate magnitude across atoms
- `dr_3d::Float64`: Maximum 3D displacement of any atom during this round
- `energy::Float64`: Total energy at end of round
"""
struct HyperRoundDiagnostics
    round::Int
    alpha_before::Float64
    alpha_after::Float64
    iterations_used::Int
    converged_early::Bool
    max_total_force::Float64
    max_3d_force::Float64
    max_extra_force::Float64
    avg_extra_dim_norm::Float64
    extra_norm::Float64
    dr_3d::Float64
    energy::Float64
end

"""
    scheduled_penalty_relax!(
        calc::MetaCalculator;
        schedule_every=10,
        max_iterations=500,
        alpha_start=1e-4,
        relax_fn=opt_tpsd,
        relax_kwargs=(;),
        trace=false,
        alpha_ramp_factor=5.0,
        max_alpha=1.0,
        pos_tol=1e-4,
        ramp_alpha=true,
        max_vol_change=3.0,
    )

Relax a structure with a scheduled extra-dimension penalty.

After each round of `schedule_every` iterations, the penalty coefficient α
is multiplied by `alpha_ramp_factor` (capped at `max_alpha`), unless
`ramp_alpha=false`.

**Termination conditions** (checked after each round):
- Extra dimensions collapsed: `extra_norm < extra_dim_tol` → terminate
- Two consecutive rounds with `dr_3d < pos_tol` (no 3D movement) → terminate
- Volume out of bounds (factor `max_vol_change` from initial) → restore and terminate
- Cell blow-up (any 3D interatomic distance > 20 Å) → restore and terminate

**Per-round diagnostics** (`HyperRoundDiagnostics`) are recorded and returned.

Returns a NamedTuple `(results, extra_dim_norms, round_diagnostics)`.
"""
function scheduled_penalty_relax!(
    calc::MetaCalculator;
    schedule_every=10,
    max_iterations=500,
    alpha_start=1e-4,
    relax_fn=opt_tpsd,
    relax_kwargs=(;),
    trace=false,
    alpha_ramp_factor=5.0,
    max_alpha=1.0,
    pos_tol=1e-4,
    ramp_alpha=true,
    max_vol_change=3.0,
    extra_dim_tol=0.1,
    dynamic_alpha=false,
    force_ratio=2.0,
)
    cell = get_cell(calc)
    schedule_every > 0 || throw(ArgumentError("schedule_every must be positive"))
    max_iterations >= 0 || throw(ArgumentError("max_iterations must be non-negative"))
    nrounds = cld(max_iterations, schedule_every)
    D = size(positions(cell), 1)
    nat = size(positions(cell), 2)

    set_modifier_parameter!(calc, ExtraDimensionPenaltyModifier, :alpha, alpha_start)

    results = []
    extra_d = Float64[]
    diagnostics = HyperRoundDiagnostics[]
    prev_stationary = false
    vol_init = CellBase.volume(cell)

    used_iterations = 0
    for i in 1:nrounds
        round_itermax = min(schedule_every, max_iterations - used_iterations)
        round_itermax <= 0 && break

        pos_before = copy(positions(cell))
        cmat_before = copy(cellmat(cell))
        pos_3d_round_start = copy(positions(cell)[1:3, :])

        alpha = get_modifier_parameter(calc, ExtraDimensionPenaltyModifier, :alpha)

        extra_norm = D > 3 ? maximum(norm.(eachcol(positions(cell)[4:end, :]))) : 0.0

        if trace
            E = eddp_energy(calc)
            vol = CellBase.volume(cell)
            pos = positions(cell)
            d3d = Float64[]
            for a=1:nat, b=a+1:nat
                push!(d3d, norm(pos[1:3, a] .- pos[1:3, b]))
            end
            min3d = isempty(d3d) ? 0.0 : minimum(d3d)
            max3d = isempty(d3d) ? 0.0 : maximum(d3d)
            d_all = Float64[]
            for a=1:nat, b=a+1:nat
                push!(d_all, norm(pos[:, a] .- pos[:, b]))
            end
            min_all = isempty(d_all) ? 0.0 : minimum(d_all)
            max_all = isempty(d_all) ? 0.0 : maximum(d_all)
            avg_extra = D > 3 ? mean(norm(pos[4:end, a]) for a in 1:nat) : 0.0
            @printf("\n── Round %d/%d  α=%.4e  E=%.1f  |extra|=%.4f  ⟨|extra|⟩=%.4f  vol=%.1f  ndim=%d ──\n",
                i, nrounds, alpha, E, extra_norm, avg_extra, vol, D)
        end
        push!(extra_d, extra_norm)

        res = relax_fn(calc; itermax=round_itermax, relax_kwargs..., trace=trace)
        push!(results, res)

        iterations_used = res.iteration
        used_iterations += iterations_used
        converged_early = iterations_used < round_itermax

        F = eddp_forces(calc)
        pos = positions(cell)
        E = eddp_energy(calc)

        max_total_force = maximum(norm.(eachcol(F)))
        if D > 3
            max_3d_force = maximum(norm.(eachcol(F[1:3, :])))
            max_extra_force = maximum(norm.(eachcol(F[4:end, :])))
            avg_extra_dim_norm = mean(norm(pos[4:end, a]) for a in 1:nat)
        else
            max_3d_force = max_total_force
            max_extra_force = 0.0
            avg_extra_dim_norm = 0.0
        end

        dr_3d = maximum(norm.(eachcol(pos[1:3, :] .- pos_3d_round_start)))

        if dynamic_alpha && D > 3
            max_extra_coord = maximum(norm.(eachcol(pos[4:end, :])))
            raw_forces = eddp_forces(calc.source)
            max_phys_extra_f = maximum(norm.(eachcol(raw_forces[4:end, :])))
            if max_extra_coord > 1e-12 && max_phys_extra_f > 1e-12
                new_alpha = min(force_ratio * max_phys_extra_f / (2 * max_extra_coord), max_alpha)
                new_alpha = max(new_alpha, alpha)
            elseif max_phys_extra_f <= 1e-12 && max_extra_coord > extra_dim_tol
                new_alpha = min(alpha * alpha_ramp_factor, max_alpha)
            else
                new_alpha = alpha
            end
        elseif ramp_alpha
            new_alpha = min(alpha * alpha_ramp_factor, max_alpha)
        else
            new_alpha = alpha
        end

        diag = HyperRoundDiagnostics(
            i, alpha, new_alpha, iterations_used, converged_early,
            max_total_force, max_3d_force, max_extra_force,
            avg_extra_dim_norm, extra_norm, dr_3d, E,
        )
        push!(diagnostics, diag)

        trace && @info "Hyper round $i done  α=$(round(alpha, digits=4))→$(round(new_alpha, digits=4))  iters=$iterations_used  dr_3d=$(round(dr_3d, digits=4))  |F|=$(round(max_total_force, digits=4))  |F3d|=$(round(max_3d_force, digits=4))  |Fext|=$(round(max_extra_force, digits=4))  |extra|=$(round(extra_norm, digits=2))"

        extra_norm_post = D > 3 ? maximum(norm.(eachcol(pos[4:end, :]))) : 0.0
        if extra_dim_tol > 0 && extra_norm_post < extra_dim_tol
            trace && @info "Hyper: extra dimensions collapsed (|extra|=$(round(extra_norm_post, digits=5)) < tol=$(extra_dim_tol)), terminating"
            break
        end

        this_stationary = dr_3d < pos_tol
        if this_stationary && prev_stationary
            trace && @info "Hyper: two consecutive stationary rounds (dr_3d=$(round(dr_3d, digits=5))), terminating"
            break
        end
        prev_stationary = this_stationary

        # Check volume limits to prevent cell collapse / blow-up
        vol_now = CellBase.volume(cell)
        if vol_now > vol_init * max_vol_change || vol_now < vol_init / max_vol_change
            trace && @info "Hyper: volume out of limits (init=$(round(vol_init, digits=1)) now=$(round(vol_now, digits=1)) factor=$max_vol_change), restoring to previous round"
            set_cellmat!(cell, cmat_before; scale_positions=false)
            set_positions!(cell, pos_before)
            calc.energy_dirty = true
            calc.forces_dirty = true
            calculate!(calc; forces=true)
            break
        end

        # Detect cell blow-up: stop if 3D max distance exceeds 20 Å
        try
            pos3d = positions(get_cell(calc))[1:3, :]
            max3d_now = 0.0
            for a=1:nat, b=a+1:nat
                d = norm(pos3d[:, a] .- pos3d[:, b])
                max3d_now = max(max3d_now, d)
            end
            if max3d_now > 20.0
                trace && @info "Hyper: cell blow-up detected (max3d=$(round(max3d_now, digits=1)) Å), restoring to previous round"
                set_cellmat!(cell, cmat_before; scale_positions=false)
                set_positions!(cell, pos_before)
                calc.energy_dirty = true
                calc.forces_dirty = true
                calculate!(calc; forces=true)
                break
            end
        catch
        end

        set_modifier_parameter!(calc, ExtraDimensionPenaltyModifier, :alpha, new_alpha)
    end

    (results=results, extra_dim_norms=extra_d, round_diagnostics=diagnostics)
end

"""
    hyper_gen_relax(bu; ndim=3, outfile=nothing, label=nothing,
                    schedule_every=50, target_iterations=500,
                    alpha_start=1e-4,
                    relax_kwargs=(;),
                    dynamic_alpha=false, force_ratio=2.0, max_alpha=1.0)

Generate a random structure, add extra dimensions, and relax using a
scheduled penalty that gradually forces the extra coordinates back to
zero.

Returns a NamedTuple `(results, extra_dim_norms, round_diagnostics)`.

# Keyword Arguments
- `ndim::Int`: Number of extra dimensions (default 3)
- `outfile`: If set, write the relaxed 3D structure as a `.res` file
- `label`: Label for the output `.res` file (required if `outfile` is set)
- `schedule_every::Int`: Iterations between penalty schedule updates
- `target_iterations::Int`: Maximum relaxation iterations
- `alpha_start::Float64`: Starting penalty coefficient
- `relax_kwargs`: Additional keyword arguments passed to the optimizer
- `dynamic_alpha::Bool`: Enable force-adaptive penalty scheduling
- `force_ratio::Float64`: Target penalty-to-physical force ratio
- `max_alpha::Float64`: Maximum penalty coefficient
- `ramp_alpha::Bool`: Ramp alpha by `alpha_ramp_factor` each round
- `alpha_ramp_factor::Float64`: Multiplicative factor for alpha ramping
- `pos_tol::Float64`: 3D displacement tolerance for stationarity check
- `extra_dim_tol::Float64`: Extra-dim norm tolerance for early termination
"""
function hyper_gen_relax(bu; ndim=3, outfile=nothing, label=nothing,
                    schedule_every=50, target_iterations=500,
                    alpha_start=1e-4, relax_fn=opt_tpsd, relax_kwargs=(;),
                    dynamic_alpha=false, force_ratio=2.0, max_alpha=1.0,
                    ramp_alpha=true, alpha_ramp_factor=5.0,
                    pos_tol=1e-4, extra_dim_tol=0.1)

    structure = build_one(joinpath(bu.state.workdir, bu.rss.seedfile))
    if ndim > 0
        structure = add_dimensions(structure, ndim)
    end
    vcalc = hyper_get_calc(structure, bu; alpha=alpha_start)

    result = scheduled_penalty_relax!(vcalc;
                    schedule_every,
                    max_iterations=target_iterations,
                    alpha_start,
                    relax_fn,
                    relax_kwargs,
                    dynamic_alpha,
                    force_ratio,
                    max_alpha,
                    ramp_alpha,
                    alpha_ramp_factor,
                    pos_tol,
                    extra_dim_tol)
    if outfile !== nothing
        structure_3d = remove_dimensions(structure)
        vcalc3d = hyper_get_calc(structure_3d, bu)
        @assert !isnothing(label) "label must be set for the output SHELX file"
        write_res(outfile, vcalc3d; label=label)
    end
    result
end

"""
    hyper_gen_relax(
        calc::EDDPCalculator,
        cell::Cell;
        ndim=3,
        alpha_start=1e-4,
        schedule_every=50,
        target_iterations=500,
        external_pressure_GPa=0.0,
        relax_fn=opt_tpsd,
        relax_kwargs=(;),
        dynamic_alpha=false,
        force_ratio=2.0,
        max_alpha=1.0,
    )

Create a hyperdimensional calculator from an [`EDDPCalculator`](@ref) and
run the scheduled-penalty relaxation.

This overload uses [`create_calculator`](@ref) to bind the model to the given
`cell` with the appropriate modifier pipeline, avoiding the need for a
[`Builder`](@ref) configuration.

Returns a NamedTuple `(results, extra_dim_norms, round_diagnostics)`.
"""
function hyper_gen_relax(
    calc::EDDPCalculator,
    cell::Cell;
    ndim=3,
    alpha_start=1e-4,
    schedule_every=50,
    target_iterations=500,
    external_pressure_GPa=0.0,
    relax_fn=opt_tpsd,
    relax_kwargs=(;),
    dynamic_alpha=false,
    force_ratio=2.0,
    max_alpha=1.0,
    ramp_alpha=true,
    alpha_ramp_factor=5.0,
    pos_tol=1e-4,
    extra_dim_tol=0.1,
)
    hcell = deepcopy(cell)
    if ndim > 0
        hcell = add_dimensions(hcell, ndim)
    end

    P = GPaToeVAng(external_pressure_GPa)

    vcalc = create_calculator(calc, hcell; modifiers=[
        ExtraDimensionPenaltyModifier(alpha_start; start_dim=4),
    ])

    P_mat = diagm([P, P, P])
    _inner_kwargs = (; relax_cell=true, ext_pressure=P_mat, relax_kwargs...)

    scheduled_penalty_relax!(vcalc;
        schedule_every,
        max_iterations=target_iterations,
        alpha_start,
        relax_fn,
        relax_kwargs=_inner_kwargs,
        dynamic_alpha,
        force_ratio,
        max_alpha,
        ramp_alpha,
        alpha_ramp_factor,
        pos_tol,
        extra_dim_tol,
    )
end
