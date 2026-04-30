"""
    hyper_get_calc(cell, bu::Builder; alpha=1e-3) -> MetaCalculator

Construct a calculator for hyperdimensional relaxation. Wraps an
[`NNCalc`](@ref) in a [`MetaCalculator`](@ref) with
[`ExtraDimensionPenaltyModifier`](@ref) and
[`CellRelaxationModifier`](@ref) so that cell degrees of freedom are
also relaxed.
"""
function hyper_get_calc(cell, bu::Builder; alpha=1e-3)
    source = NNCalc(
        cell,
        bu.cf,
        load_ensemble(bu);
        nmax=500,
        core=CoreRepulsion(bu.state.core_size),
        elemental_energies=bu.state.elemental_energies,
    )
    penalty = ExtraDimensionPenaltyModifier(alpha; start_dim=4)
    P_external = bP(bu.rss.pressure_gpa)
    cell_relax = CellRelaxationModifier(cell; external_pressure=P_external)
    MetaCalculator(source, [penalty, cell_relax])
end

function bP(press_gpa)
    # Clamp to minimum 0.1 GPa to prevent cell blow-up during hyperspace
    p = GPaToeVAng(max(press_gpa, 0.1))
    [p 0 0; 0 p 0; 0 0 p]
end

"""
    _get_penalty_schedule(; n_hold=5, multiplier=0.1, cap_factor=2.0)

Return a closure that ramps the penalty coefficient `alpha` from its
current value. The first `n_hold` steps keep `alpha` unchanged; after
that `alpha` is updated as `max(current * cap_factor, step * multiplier)`.
"""
function _get_penalty_schedule(; n_hold=5, multiplier=0.1, cap_factor=2.0)
    n = 0
    function func(x)
        n += 1
        if n < n_hold
            return x
        end
        return min(n * multiplier, x * cap_factor)
    end
    return func
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
        n_hold=0,
        multiplier=1.0,
        cap_factor=10.0,
    )

Relax a structure with a scheduled extra-dimension penalty. Every
`schedule_every` iterations the penalty coefficient α is updated:

- First `n_hold` rounds: α stays at `alpha_start`
- Subsequent rounds: α = min(n * multiplier, α * cap_factor)

With `n_hold=0`, the ramp starts immediately: α grows from 1e-4 through
~0.01 to ~1.0 over ~4 rounds, forcing extra dimensions to zero.

The penalty is accessed and mutated through `get_modifier_parameter` /
`set_modifier_parameter!` on the `ExtraDimensionPenaltyModifier`
contained in `calc`.

Returns `(results, extra_dim_norms)` where `results` is a vector of
optimizer-return values and `extra_dim_norms` records the maximum norm
of the extra-dimensional coordinates before each schedule block.
"""
function scheduled_penalty_relax!(
    calc::MetaCalculator;
    schedule_every=10,
    max_iterations=500,
    alpha_start=1e-4,
    relax_fn=opt_tpsd,
    relax_kwargs=(;),
    trace=false,
    n_hold=0,
    multiplier=0.5,
    cap_factor=20.0,
    max_vol_change=3.0,
)
    cell = get_cell(calc)
    nrounds = div(max_iterations, schedule_every)

    # Build penalty schedule from kwargs (n_hold, multiplier, cap_factor)
    sf = _get_penalty_schedule(; n_hold, multiplier, cap_factor)

    set_modifier_parameter!(calc, ExtraDimensionPenaltyModifier, :alpha, alpha_start)

    results = []
    extra_d = []
    converged = false
    extra_norm_prev = Inf
    vol_init = CellBase.volume(cell)

    for i in 1:nrounds
        # Save state before this round (to restore if it goes wrong)
        pos_before = copy(positions(cell))
        cmat_before = copy(cellmat(cell))

        extra_norm = maximum(norm.(eachcol(positions(cell)[4:end, :])))
        alpha = get_modifier_parameter(calc, ExtraDimensionPenaltyModifier, :alpha)
        if trace
            E = eddp_energy(calc)
            vol = CellBase.volume(cell)
            pos = positions(cell)
            nat = size(pos, 2)
            D = size(pos, 1)
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
            @info "Hyper $i/$nrounds  α=$(round(alpha, digits=4))  E=$(round(E, digits=1))  |extra|=$(round(extra_norm, digits=2))  vol=$(round(vol, digits=1))  min3d=$(round(min3d, digits=2))  max3d=$(round(max3d, digits=2))  min_all=$(round(min_all, digits=2))  max_all=$(round(max_all, digits=2))  ndim=$D"
        end
        push!(extra_d, extra_norm)

        if extra_norm < 0.5 && extra_norm_prev < 0.5
            trace && @info "Hyper: extra dimensions converged (|extra|=$(round(extra_norm, digits=3)))"
            converged = true
            break
        end
        extra_norm_prev = extra_norm

        res = relax_fn(calc; itermax=schedule_every, relax_kwargs..., trace=trace)
        push!(results, res)

        # Check volume limits to prevent cell collapse / blow-up
        vol_now = CellBase.volume(cell)
        if vol_now > vol_init * max_vol_change || vol_now < vol_init / max_vol_change
            trace && @info "Hyper: volume out of limits (init=$(round(vol_init, digits=1)) now=$(round(vol_now, digits=1)) factor=$max_vol_change), restoring to previous round"
            set_cellmat!(calc, cmat_before)
            set_positions!(calc, pos_before)
            # Reset cached energy/forces so next calculate! picks up restored state
            calculate!(calc; forces=true)
            break
        end

        new_alpha = sf(alpha)
        # Cap alpha to prevent penalty overwhelm (forces ~ new_alpha * extra_dim)
        new_alpha = min(new_alpha, 0.5)
        # Detect cell blow-up: stop if 3D max distance exceeds 20 Å
        try
            pos3d = positions(get_cell(calc))[1:3, :]
            nat = size(pos3d, 2)
            max3d_now = 0.0
            for a=1:nat, b=a+1:nat
                d = norm(pos3d[:, a] .- pos3d[:, b])
                max3d_now = max(max3d_now, d)
            end
            if max3d_now > 20.0
                trace && @info "Hyper: cell blow-up detected (max3d=$(round(max3d_now, digits=1)) Å), restoring to previous round"
                set_cellmat!(calc, cmat_before)
                set_positions!(calc, pos_before)
                calculate!(calc; forces=true)
                break
            end
        catch
        end
        set_modifier_parameter!(calc, ExtraDimensionPenaltyModifier, :alpha, new_alpha)
    end

    (results=results, extra_dim_norms=extra_d)
end

"""
    hyper_gen_relax(bu; ndim=3, outfile=nothing, label=nothing,
                    schedule_every=50, target_iterations=500,
                    alpha_start=1e-4,
                    relax_kwargs=(;))

Generate a random structure, add extra dimensions, and relax using a
scheduled penalty that gradually forces the extra coordinates back to
zero.

Returns `(results, extra_dim_norms)` where `results` is a vector of
optimizer-return values and `extra_dim_norms` records the maximum norm
of the extra-dimensional coordinates before each schedule block.

# Keyword Arguments
- `ndim::Int`: Number of extra dimensions (default 3)
- `outfile`: If set, write the relaxed 3D structure as a `.res` file
- `label`: Label for the output `.res` file (required if `outfile` is set)
- `schedule_every::Int`: Iterations between penalty schedule updates
- `target_iterations::Int`: Maximum relaxation iterations
- `alpha_start::Float64`: Starting penalty coefficient
- `relax_kwargs`: Additional keyword arguments passed to the optimizer
"""
function hyper_gen_relax(bu; ndim=3, outfile=nothing, label=nothing,
                    schedule_every=50, target_iterations=500,
                    alpha_start=1e-4, relax_fn=opt_tpsd, relax_kwargs=(;))

    structure = build_one(joinpath(bu.state.workdir, bu.rss.seedfile))
    if ndim > 0
        structure = add_dimensions(structure, ndim)
    end
    vcalc = hyper_get_calc(structure, bu; alpha=alpha_start)

    results, dnorm = scheduled_penalty_relax!(vcalc;
                    schedule_every,
                    max_iterations=target_iterations,
                    schedule_function=_get_penalty_schedule(),
                    alpha_start,
                    relax_fn,
                    relax_kwargs)
    if outfile !== nothing
        structure_3d = remove_dimensions(structure)
        vcalc3d = hyper_get_calc(structure_3d, bu)
        @assert !isnothing(label) "label must be set for the output SHELX file"
        write_res(outfile, vcalc3d; label=label)
    end
    results, dnorm
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
    )

Create a hyperdimensional calculator from an [`EDDPCalculator`](@ref) and
run the scheduled-penalty relaxation.

This overload uses [`create_calculator`](@ref) to bind the model to the given
`cell` with the appropriate modifier pipeline, avoiding the need for a
[`Builder`](@ref) configuration.

Returns `(results, extra_dim_norms)` as in the Builder overload.
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
)
    hcell = deepcopy(cell)
    if ndim > 0
        hcell = add_dimensions(hcell, ndim)
    end

    P = GPaToeVAng(external_pressure_GPa)

    vcalc = create_calculator(calc, hcell; modifiers=[
        ExtraDimensionPenaltyModifier(alpha_start; start_dim=4),
        CellRelaxationModifier(hcell; external_pressure=diagm([P, P, P])),
    ])

    scheduled_penalty_relax!(vcalc;
        schedule_every,
        max_iterations=target_iterations,
        alpha_start,
        relax_fn,
        relax_kwargs,
    )
end
