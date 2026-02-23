"""
Return a calculation with additional dimensions
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
Generate and relax the structure with additional dimensions
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
