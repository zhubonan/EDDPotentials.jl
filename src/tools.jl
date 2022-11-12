#=
Various tool functions for workflow managements
=#

using CellBase: rattle!, reduce, Composition
using ProgressMeter
import CellBase: write_res
using Base.Threads
using JLD2
using Dates
using UUIDs

function relax_structures(files, en_path::AbstractString, cf; savepath="relaxed", skip_existing=true, nmax=1000, core_radius=1.0)

    ensemble = load_from_jld2(en_path, EnsembleNNInterface)

    structures = CellBase.read_cell.(files)

    isdir(savepath) || mkdir(savepath)

    p = Progress(length(structures))
    n = length(structures)

    function do_work(i)
        fname = splitpath(files[i])[end]
        # Skip and existing file
        skip_existing && isfile(joinpath(savepath, fname)) && return

        calc = NNCalc(structures[i], cf, deepcopy(ensemble); nmax=nmax, core=CoreReplusion(core_radius))
        vc = VariableCellCalc(calc)

        optimise!(vc)
        write_res(joinpath(savepath, fname), vc; label=fname, symprec=0.1)
    end

    @info "Total number of structures: $n"
    for i in 1:n
        do_work(i)
        next!(p)
    end
end



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
"""
function build_and_relax(seedfile::AbstractString, ensemble, cf; timeout=10, nmax=500, pressure_gpa=0.0,
    show_trace=false, method=TwoPointSteepestDescent(), kwargs...)
    lines = open(seedfile, "r") do seed
        cellout = read(pipeline(`timeout $(timeout) buildcell`, stdin=seed, stderr=devnull), String)
        split(cellout, "\n")
    end

    # Generate a unique label
    p = pressure_gpa / 160.21766208
    ext = diagm([p, p, p])
    cell = CellBase.read_cell(lines)

    # Broken
    calc = EDDP.NNCalc(cell, cf, ensemble; nmax)
    vc = EDDP.VariableCellCalc(calc, external_pressure=ext)
    res = EDDP.optimise!(vc; show_trace, method, kwargs...)
    vc, res
end

"""
    random_from_buildcell(seedfile; timeout=60)

Run the `buildcell` progress with a defined timeout value.
"""
function random_from_buildcell(seedfile; timeout=60)
    lines = open(seedfile, "r") do seed
        cellout = read(pipeline(`timeout $(timeout) buildcell`, stdin=seed, stderr=devnull), String)
        split(cellout, "\n")
    end
    @show lines
    CellBase.read_cell(lines)
end

"""
    build_random_structures(seedfile, outdir;n=1, show_progress=false, timeout=60)

Build multiple random structures in the target folder.
"""
function build_random_structures(seedfile, outdir;n=1, show_progress=false, timeout=60, outfmt="res")
    i = 0
    if show_progress
        prog = Progress(n)
    end
    while i< n 
        cell = random_from_buildcell(seedfile;timeout)
        label = EDDP.get_label(EDDP.stem(seedfile))
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
    run_rss(seedfile, ensemble, cf;max=1, outdir="./", kwargs...)

Perform random structure searching using the seed file.
"""
function run_rss(seedfile, ensemble, cf;show_progress=false, max=1, outdir="./", packed=false,
                niggli_reduce_output=true, max_err=10, kwargs...)
    i = 1

    isdir(outdir) || mkdir(outdir)
    if packed
        label = EDDP.get_label(EDDP.stem(seedfile))
        # Name of the packed out file
        outfile = joinpath(outdir, "$(label).packed.res")
        mode = "a"
    else
        mode = "w"
    end
    nerr = 0
    if show_progress
        pmeter = Progress(max)
    end
    while i <= max
        local vc
        try
            vc, res = build_and_relax(seedfile, ensemble, cf; kwargs...)
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
        # Reset the error counter
        nerr = 0
 
        label = EDDP.get_label(EDDP.stem(seedfile))
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
    build_and_relax_one(seedfile::AbstractString, outdir::AbstractString, ensemble, cf;timeout=10, warn=true)

Build and relax a single structure, ensure that the process *does* generate a new structure.
"""
function build_and_relax_one(seedfile::AbstractString, outdir::AbstractString, ensemble, cf; nmax=500, timeout=10, warn=true, max_attempts=999, write=true)
    not_ok = true
    n = 1
    relaxed = nothing
    while not_ok && n <= max_attempts
        try
            relaxed = build_and_relax(seedfile, outdir, ensemble, cf; timeout, write, nmax)
        catch err
            if !isa(err, InterruptException)
                if warn
                    if typeof(err) <: ProcessFailedException
                        @warn " `buildcell` failed to make the structure"
                    else
                        @warn "relaxation errored with $err"
                    end
                end
            else
                # Throw Ctrl-C interruptions
                throw(err)
            end
            n += 1
            continue
        end
        not_ok = false
    end
    relaxed
end

"""
    worker_build_and_relax_one(channel::AbstractChannel, args...; kwargs...)

Worker function that put the results into the channel
"""
function worker_build_and_relax_one(job_channel, result_channel, seed_file, outdir, ensemble, cf;
    nmax=500, timeout=10, warn=true, max_attempts=999, write=true)
    @info "Starting worker function"
    while true
        job_id = take!(job_channel)
        if job_id < 0
            break
        end
        relaxed = build_and_relax_one(seed_file, outdir, ensemble, cf; nmax, timeout, warn, max_attempts, write)
        put!(result_channel, relaxed)
    end
end

"""
    build_and_relax(num::Int, seedfile::AbstractString, outdir::AbstractString, ensemble, cf;timeout=10)

Build and relax `num` structures in parallel (threads) using passed `ModuleEnsemble` and `CellFeature`
"""
function build_and_relax(num::Int, seedfile::AbstractString, outdir::AbstractString, ensemble, cf; timeout=10, nmax=500, deduplicate=false)
    results_channel = RemoteChannel(() -> Channel(num))
    job_channel = RemoteChannel(() -> Channel(num))

    # Put the jobs
    for i = 1:num
        put!(job_channel, i)
    end

    @info "Launching workers"

    # Launch the workers
    futures = []
    for worker in workers()
        push!(futures, remotecall(worker_build_and_relax_one, worker, job_channel, results_channel, seedfile, outdir, ensemble, cf; timeout, warn=true, write=false, nmax))
    end
    sleep(0.1)
    # None of the futures should be ready
    for future in futures
        if isready(future)
            output = fetch(future)
            @error "Error detected for the worker $output"
            throw(output)
        end
    end

    @info "Start the receiving loop"
    # Receive the data and update the progress
    i = 1
    progress = Progress(num)
    # Fingerprint vectors used for deduplication
    all_fvecs = []
    try
        while i <= num
            res = take!(results_channel)
            label = res.metadata[:label]
            # Get feature vector
            # Compare feature vector
            if deduplicate
                fvec = CellBase.fingerprint(res)
                if is_unique_fvec(all_fvecs, fvec)
                    push!(all_fvecs, fvec)
                    # Unique structure - write it out
                    write_res(joinpath(outdir, "$(label).res"), res)
                    ProgressMeter.next!(progress;)
                    i += 1
                else
                    @warn "This structure has been seen before"
                    # Resubmit the job
                    put!(job_channel, i)
                end
            else
                write_res(joinpath(outdir, "$(label).res"), res)
                ProgressMeter.next!(progress;)
                i += 1
            end
        end
    catch err
        if isa(err, InterruptException)
            # interrupt workers
            Distributed.interrupt()
        else
            throw(err)
        end
    finally
        foreach(x -> put!(job_channel, -1), 1:length(workers()))
        sleep(1.0)
        close(job_channel)
        close(results_channel)
    end

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

"""
    build_and_relax(num::Int, seedfile::AbstractString, outdir::AbstractString, ensemble_file::AbstractString;timeout=10)

Build the structure and relax it
"""
function build_and_relax(num::Int, seedfile::AbstractString, outdir::AbstractString, ensemble_file::AbstractString; timeout=10, kwargs...)
    ensemble = load_ensemble_model(ensemble_file)
    featurespec = load_featurespec(ensemble_file)
    build_and_relax(num, seedfile, outdir, ensemble, featurespec; timeout, kwargs...)
end


ensure_dir(path) = isdir(path) || mkdir(path)

function get_label(seedname)
    dt = Dates.format(now(), "yy-mm-dd-HH-MM-SS")
    suffix = string(uuid4())[end-7:end]
    "$(seedname)-$(dt)-$(suffix)"
end

stem(x) = splitext(splitpath(x)[end])[1]

"""
Call `buildcell` to generate many random structure under `outdir`
"""
function build_cells(seedfile, outdir, num; save_as_res=true, build_timeout=5, ntasks=nthreads())
    asyncmap((x) -> build_one_cell(seedfile, outdir; save_as_res, build_timeout), 1:num;
        ntasks=ntasks)
end

"""
Call `buildcell` to generate many random structure under `outdir`
"""
function build_one_cell(seedfile, outdir; save_as_res=true, build_timeout=5, suppress_stderr=false, max_attemps=999)
    not_ok = true
    suppress_stderr ? stderr_dst = devnull : stderr_dst = nothing
    n = 1
    while not_ok && n <= max_attemps
        outname = save_as_res ? get_label(stem(seedfile)) * ".res" : get_label(stem(seedfile)) * ".cell"
        outpath = joinpath(outdir, outname)
        try
            if save_as_res
                pip = pipeline(pipeline(`timeout $(build_timeout) buildcell`, stdin=seedfile), `timeout $(build_timeout) cabal cell res`)
                pip = pipeline(pip, stdout=outpath, stderr=stderr_dst)
            else
                pip = pipeline(`timeout $(build_timeout) buildcell`, stdin=seedfile, stdout=outpath, stderr=stderr_dst)
            end

            run(pip)
        catch err
            if typeof(err) <: ProcessFailedException
                rm(outpath)
                n += 1
                continue
            else
                throw(err)
            end
        end
        # Success
        not_ok = false
    end
end


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
        for i in 1:nshake
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