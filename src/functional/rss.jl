#=
Routines for performing random structure searching using models
=#

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
    core_size=1.0,
    init_structure_transform=nothing,
    relax_option=RelaxOption(),
    elemental_energies=Dict{Symbol,Any}(),
)
    cell = build_one(seedfile; timeout, init_structure_transform)
    calc = EDDPotentials.NNCalc(cell, cf, ensemble; nmax, core=CoreRepulsion(core_size), elemental_energies)
    re = Relax(calc, relax_option)
    relax!(re)
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
)
    _relax_option = deepcopy(relax_option)
    i = 1

    isdir(outdir) || mkdir(outdir)
    if packed
        # Select the first resolved seed file and use it as the name
        name = _select_seed(seedfile, seedfile_weights)[2][1]
        label = EDDPotentials.get_label(EDDPotentials.stem(name))
        # Name of the packed out file
        outfile = joinpath(outdir, "$(label).packed.res")
        mode = "a"
    else
        mode = "w"
    end
    if show_progress
        pmeter = Progress(max)
    end
    _pressure_gpa = pressure_gpa
    while i <= max
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
        )

        calc = res.relax.calc

        # Check for ensemble error and act
        # This prunes structure that we are too confident, e.g. nothing to learn from
        if ensemble_std_min > 0.0 || ensemble_std_max > 0.0
            estd = get_energy_std(calc) / length(get_cell(calc))
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

        if !packed
            outfile = joinpath(outdir, "$(label).res")
        end

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
        write_res(outfile, cell, mode)
        i += 1
        if show_progress
            ProgressMeter.next!(pmeter)
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
    epa = get_energy(calc) / length(get_cell(calc))
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
