"""
    hyper_get_calc(cell, bu::Builder) -> VariableCellCalc

Construct a calculator with extra-dimensional degrees of freedom for
hyperdimensional relaxation. Wraps an [`NNCalc`](@ref) with a [`DimPenalty`](@ref)
inside a [`MultiCalc`](@ref), then wraps in a [`VariableCellCalc`](@ref).
"""
function hyper_get_calc(cell, bu::Builder)
   calc = NNCalc(
        cell,
        bu.cf,
        load_ensemble(bu);
        nmax=500,
        core=CoreRepulsion(bu.state.core_size),
        elemental_energies=bu.state.elemental_energies,
    )

    dcalc = DimPenalty(calc;alpha=1e-3)
    calc = MultiCalc([calc, dcalc])
    VariableCellCalc(calc)
end

 """
Scheduling function for increasing extra dimension penalty
"""
function _get_penalty_schedule()
    n = 0
    function func(x)
        n += 1
        if n < 5
            return x
        end
        return min(n * 0.1 ,x * 2)
    end
    return func
end
    
"""
    hyper_gen_relax(bu; ndim=3, outfile=nothing, label=nothing,
                    schedule_every=50, target_iterations=500,
                    relax_kwargs=Dict{Symbol, Any}())

Generate a random structure, add extra dimensions, and relax using a scheduled
penalty that gradually forces the extra coordinates back to zero. Returns
`(results, dnorm)` where `results` is the relaxation result and `dnorm` is the
final norm of the extra-dimensional coordinates.

# Keyword Arguments
- `ndim::Int`: Number of extra dimensions (default 3)
- `outfile`: If set, write the relaxed 3D structure as a `.res` file
- `label`: Label for the output `.res` file (required if `outfile` is set)
- `schedule_every::Int`: Iterations between penalty schedule updates
- `target_iterations::Int`: Maximum relaxation iterations
- `relax_kwargs`: Additional keyword arguments passed to [`RelaxOption`](@ref)
"""
function hyper_gen_relax(bu; ndim=3, outfile=nothing, label=nothing, 
                    schedule_every=50, target_iterations=500, 
                    relax_kwargs=Dict{Symbol, Any}()
                    )

    structure = build_one(joinpath(bu.state.workdir, bu.rss.seedfile))
    if ndim > 0
        structure = add_dimensions(structure, ndim)
    end
    vcalc = hyper_get_calc(structure, bu)
    re = Relax(vcalc, RelaxOption(
        external_pressure=[bu.rss.pressure_gpa],
        relax_kwargs...
        ))
    # This is the dimensionality penalty type
    dimp = vcalc.calc.calcs[2]

    results, dnorm = scheduled_penalty_relax!(re, 
                    dimp;
                    schedule_every,
                    max_iterations=target_iterations,
                    schedule_function=_get_penalty_schedule());
    if outfile !== nothing
        structure_3d = remove_dimensions(structure)
        vcalc3d = get_calc(structure_3d, bu)
        @assert !isnothing(label) "label must be set for the output SHELX file"
        write_res(outfile, vcalc3d; label=label)
    end
    results, dnorm
end
