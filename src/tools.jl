#=
Various tool functions for workflow managements
=#

using CellBase: rattle!, reduce, Composition
using ProgressMeter: @showprogress
import CellBase: write_res
using StatsBase
using Base.Threads
using JLD2
using Dates
using UUIDs

"""
    update_metadata!(vc::AbstractCalc, label;symprec=1e-2)

Update the metadata attached to a `Cell`` object
"""
function update_metadata!(vc::AbstractCalc, label; symprec=1e-2)
    this_cell = get_cell(vc)
    # Set metadata
    this_cell.metadata[:enthalpy] = get_enthalpy(vc)
    this_cell.metadata[:volume] = volume(this_cell)
    this_cell.metadata[:pressure] = get_pressure_gpa(vc)
    this_cell.metadata[:label] = label
    symm = CellBase.get_international(this_cell, symprec)
    this_cell.metadata[:symm] = "($(symm))"
    # Write to the file
    vc
end

"""
    write_res(path, vc::VariableCellCalc;symprec=1e-2, label="EDDP")

Write structure in VariableCellCalc as SHELX file.
"""
function write_res(path, vc::VariableCellCalc; symprec=1e-2, label="EDDP")
    update_metadata!(vc, label; symprec)
    write_res(path, get_cell(vc))
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
    nmax=500,
    pressure_gpa=0.0,
    relax_cell=true,
    show_trace=false,
    method=TwoPointSteepestDescent(),
    core_size=1.0,
    init_structure_transform=nothing,
    kwargs...,
)

    cell = build_one(seedfile; timeout, init_structure_transform)

    calc = EDDP.NNCalc(cell, cf, ensemble; nmax, core=CoreReplusion(core_size))
    vc, res = relax!(calc; relax_cell, pressure_gpa, show_trace, method, kwargs...)
    vc, res
end

"""
    build_one(seedfile;timeout=10, init_structure_transform=nothing)

Build a single structure via `buildcell`.
"""
function build_one(seedfile; timeout=10, init_structure_transform=nothing, max_attemp=100)
    local cell::Cell{Float64}
    i = 1
    while true
        if i >= max_attemp
            throw(ErrorException("Maximum attempt for building structure exceeded!"))
        end
        lines = open(seedfile, "r") do seed
            try
                cellout = read(
                    pipeline(`timeout $(timeout) buildcell`, stdin=seed, stderr=devnull),
                    String,
                )
                split(cellout, "\n")
            catch err
                if typeof(err) <: ProcessFailedException
                    @warn " `buildcell` failed to make the structure"
                else
                    throw(err)
                end
            end
        end
        isnothing(lines) && continue

        # Generate a unique label
        cell = CellBase.read_cell(lines)

        if !isnothing(init_structure_transform)
            cell = init_structure_transform(cell)
            if isnothing(cell)
                # This generated structure is no good....
                i += 1
                continue
            end
        end
        break
    end
    return cell
end

GPaToeVAng(x) = x / 160.21766208

"""
    relax!(calc::NNCalc;relax_cell=true, show_trace, method, opt_kwargs...)

Relax the structure of the calculator.
"""
function relax!(
    calc::NNCalc;
    relax_cell=true,
    pressure_gpa=0.0,
    show_trace=false,
    method=TwoPointSteepestDescent(),
    out_label="eddp-output",
    opt_kwargs...,
)

    if relax_cell
        p = pressure_gpa / 160.21766208
        ext = diagm([p, p, p])
        vc = EDDP.VariableCellCalc(calc, external_pressure=ext)
        # Run optimisation
        res = EDDP.optimise!(vc; show_trace, method, opt_kwargs...)
    else
        vc = calc
        res = EDDP.optimise!(calc; show_trace, method, opt_kwargs...)
    end
    update_metadata!(vc, out_label)
    vc, res
end


"""
    build_random_structures(seedfile, outdir;n=1, show_progress=false, timeout=60, seedfile_weights)

Build multiple random structures in the target folder. A glob pattern may be used for the
`seedfile` argument.

"""
function build_random_structures(
    seedfile::Union{AbstractString,Vector},
    outdir;
    n=1,
    show_progress=false,
    timeout=60,
    outfmt="res",
    seedfile_weights=[1.0],
)
    i = 0
    if show_progress
        prog = Progress(n)
    end
    while i < n
        this_seed = _select_seed(seedfile, seedfile_weights)[1]
        cell = build_one(this_seed; timeout)
        label = EDDP.get_label(EDDP.stem(this_seed))
        cell.metadata[:label] = label
        if outfmt == "res"
            write_res(joinpath(outdir, "$(label).res"), cell)
        else
            CellBase.write_cell(joinpath(outdir, "$(label).cell"), cell)
        end
        i += 1
        if show_progress
            ProgressMeter.update!(prog)
        end
    end
end

"""
    _select_seed(names::AbstractVector, weights::AbstractVector)

Select a random seed from a vector of names and expand glob pattern if needed, and the resolved paths.

# Examples 

Selecting seeds with pattern `Si-*.cell` with equal weightings.
```julia
_select_seed(["Si-*.cell"], [1.0])
```

"""
function _select_seed(
    names::AbstractVector,
    weights::AbstractVector=repeat([1.0], length(names)),
)

    actual_names = String[]
    actual_weights = Float64[]
    for (i, name) in enumerate(names)
        for j in glob_allow_abs(name)
            push!(actual_names, j)
            if i <= length(weights)
                i_ = i
            else
                i_ = length(weights)
            end
            push!(actual_weights, weights[i_])
        end
    end
    actual_weights ./= sum(actual_weights)
    @assert !isempty(actual_names) "No valid file found with $names"
    sample(actual_names, Weights(actual_weights)), actual_names
end

_select_seed(names::AbstractString, weights=[1.0]) = _select_seed([names], weights)


"""
    run_rss(seedfile, ensemble, cf;max=1, outdir="./", kwargs...)

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
    kwargs...,
)
    i = 1

    isdir(outdir) || mkdir(outdir)
    if packed
        # Select the first resolved seed file and use it as the name
        name = _select_seed(seedfile, seedfile_weights)[2][1]
        label = EDDP.get_label(EDDP.stem(name))
        # Name of the packed out file
        outfile = joinpath(outdir, "$(label).packed.res")
        mode = "a"
    else
        mode = "w"
    end
    if show_progress
        pmeter = Progress(max)
    end

    while i <= max
        # Use randomly chosen pressure
        if pressure_gpa_range !== nothing
            pressure_gpa =
                rand() * (pressure_gpa_range[2] - pressure_gpa_range[1]) +
                pressure_gpa_range[1]
        end
        # Select the actual seeds
        this_seed = _select_seed(seedfile, seedfile_weights)[1]
        # Build the random structure and relax it
        vc, res = build_and_relax_one(
            this_seed,
            ensemble,
            cf;
            max_err,
            init_structure_transform,
            pressure_gpa,
            kwargs...,
        )

        # Check for ensemble error and act
        # This prunes structure that we are too confident, e.g. nothing to learn from
        if ensemble_std_min > 0.0 || ensemble_std_max > 0.0
            estd = get_energy_std(vc.calc) / length(get_cell(vc))
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
            if check_energy_threshold!(composition_engmin, vc, eng_threshold) == false
                continue
            end
        end

        # Update the label of the structure
        label = get_label(stem(this_seed))
        EDDP.update_metadata!(vc, label)

        if !packed
            outfile = joinpath(outdir, "$(label).res")
        end

        # Write output file
        cell = get_cell(vc)
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
    kwargs...,
)
    nerr = 1
    local vc
    local res
    while true
        try
            vc, res =
                build_and_relax(seedfile, ensemble, cf; init_structure_transform, kwargs...)
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
    vc, res
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


"""
Check if a feature vector already present in an array of vectors
"""
function is_unique_fvec(all_fvecs, fvec; tol=1e-2, lim=5)
    match = false
    for ref in all_fvecs
        dist = CellBase.fingerprint_distance(ref, fvec; lim)
        if dist < tol
            match = true
            break
        end
    end
    !match
end

ensure_dir(path) = isdir(path) || mkdir(path)

function get_label(seedname)
    dt = Dates.format(now(), "yy-mm-dd-HH-MM-SS")
    suffix = string(uuid4())[end-7:end]
    "$(seedname)-$(dt)-$(suffix)"
end

"Return the *stea* part of a file name"
stem(x) = splitext(splitpath(x)[end])[1]

swapext(fname, new) = splitext(fname)[1] * new


"""
    shake_res(files::Vector, nshake::Int, amp::Real)

Shake the given structures and write new files with suffix `-shake-N.res`.

"""
function shake_res(files::Vector, nshake::Int, amp::Real, cellamp::Real=0.02)
    for f in files
        cell = read_res(f)
        pos_backup = get_positions(cell)
        cellmat_backup = get_cellmat(cell)
        label = cell.metadata[:label]
        for i = 1:nshake
            rattle!(cell, amp)
            rattle_cell!(cell, cellamp)
            cell.metadata[:label] = label * "-shake-$i"
            write_res(splitext(f)[1] * "-shake-$i.res", cell)
            # Reset the original cellmatrix and positions
            set_cellmat!(cell, cellmat_backup)
            set_positions!(cell, pos_backup)
        end
    end
end

"""
    rattle_cell(cell::Cell, amp)

Rattle the cell shape based on random fractional changes on the cell parameters.
"""
function rattle_cell!(cell::Cell, amp)
    local new_cellpar
    i = 0
    while true
        new_cellpar = [x * (1 + rand() * amp) for x in cellpar(cell)]
        CellBase.isvalidcellpar(new_cellpar...) && break
        # Cannot found a valid cell parameters?
        if i > 10
            return cell
        end
        i += 1
    end
    new_lattice = Lattice(new_cellpar)
    spos = get_scaled_positions(cell)
    CellBase.set_cellmat!(cell, cellmat(new_lattice))
    positions(cell) .= cellmat(cell) * spos
    cell
end


raw"""

Generate LJ like pair-wise interactions

```math
F = \alpha(-2f(r, rc)^a + f(r, rc)^{2a})

The equilibrium position is at ``r_c/2``.
```

Support only single a element for now.
"""
function lj_like_calc(cell::Cell; α=1.0, a=6, rc=3.0)
    elem = unique(species(cell))
    @assert length(elem) == 1 "Only works for single specie Cell for now."
    cf = EDDP.CellFeature(elem, p2=[a, 2a], p3=[], q3=[], rcut2=rc)
    model = EDDP.LinearInterface([0, -2, 1.0] .* α)
    EDDP.NNCalc(cell, cf, model)
end


"""
    relax_structures(files, outdir, cf::CellFeature, ensemble::AbstractNNInterface;

Relax many structures and place the relaxed structures in the output directory
"""
function relax_structures(
    files,
    outdir,
    cf::CellFeature,
    ensemble::AbstractNNInterface;
    nmax=500,
    core_size=1.0,
    relax_cell=true,
    pressure_gpa=0.0,
    show_trace=false,
    method=TwoPointSteepestDescent(),
    kwargs...,
)
    Threads.@threads for fname in files
        # Deal with different types of inputs
        if endswith(fname, ".res")
            cell = read_res(fname)
            label = cell.metadata[:label]
        elseif endswith(fname, ".cell")
            cell = read_cell(fname)
            label = stem(fname)
        end
        calc =
            EDDP.NNCalc(cell, cf, deepcopy(ensemble); nmax, core=CoreReplusion(core_size))
        vc, _ = relax!(
            calc;
            relax_cell,
            pressure_gpa,
            show_trace,
            method,
            out_label=label,
            kwargs...,
        )
        outname = joinpath(outdir, stem(fname) * ".res")
        write_res(outname, get_cell(vc))
    end
end
