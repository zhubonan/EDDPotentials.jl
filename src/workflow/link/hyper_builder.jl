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
        core_repulsion=CoreRepulsion(bu.state.rss.core_size),
        elemental_energies=bu.state.elemental_energies,
    )
    penalty = ExtraDimensionPenaltyModifier(alpha; start_dim=4)
    MetaCalculator(source, [penalty])
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
"""
function hyper_gen_relax(bu::Builder; ndim=3, outfile=nothing, label=nothing,
                    schedule_every=50, target_iterations=500,
                    alpha_start=1e-4, relax_fn=opt_tpsd, relax_kwargs=(;),
                    dynamic_alpha=false, force_ratio=2.0, max_alpha=1.0,
                    ramp_alpha=true, alpha_ramp_factor=5.0,
                    pos_tol=1e-4, extra_dim_tol=0.1)

    structure = build_one(joinpath(bu.state.workdir, bu.rss.seedfile))
    if ndim > 0
        structure = CellBase.add_dimensions(structure, ndim)
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
