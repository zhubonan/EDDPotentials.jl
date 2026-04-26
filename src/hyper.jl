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
    P = GPaToeVAng(press_gpa)
    diagm([P, P, P])
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
        schedule_function = _get_penalty_schedule(),
        alpha_start = 1e-4,
        relax_fn = opt_tpsd,
        relax_kwargs = (;),
    )

Relax a structure with a scheduled extra-dimension penalty. Every
`schedule_every` iterations the penalty coefficient is updated by
calling `schedule_function(alpha)`, which should return the new value.

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
    schedule_function=_get_penalty_schedule(),
    alpha_start=1e-4,
    relax_fn=opt_tpsd,
    relax_kwargs=(;),
)
    cell = get_cell(calc)

    set_modifier_parameter!(calc, ExtraDimensionPenaltyModifier, :alpha, alpha_start)

    results = []
    extra_d = []

    for _ in 1:div(max_iterations, schedule_every)
        push!(extra_d,
            maximum(norm.(eachcol(positions(cell)[4:end, :]))))
        res = relax_fn(calc; itermax=schedule_every, relax_kwargs...)
        push!(results, res)
        alpha = get_modifier_parameter(calc, ExtraDimensionPenaltyModifier, :alpha)
        set_modifier_parameter!(calc, ExtraDimensionPenaltyModifier, :alpha,
            schedule_function(alpha))
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
                    alpha_start=1e-4, relax_kwargs=(;))

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
                    relax_kwargs)
    if outfile !== nothing
        structure_3d = remove_dimensions(structure)
        vcalc3d = hyper_get_calc(structure_3d, bu)
        @assert !isnothing(label) "label must be set for the output SHELX file"
        write_res(outfile, vcalc3d; label=label)
    end
    results, dnorm
end
