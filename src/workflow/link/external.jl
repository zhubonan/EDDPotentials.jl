#=
External DFT calculator integrations.

This module consolidates all external tool integrations for running DFT calculations:
- Abstract interface for external calculators
- PP3 (Pairwise Potential Program)
- CRUD/ACRUD (CASTEP calculation runners)
- Mock calculator for testing
- Dispatcher for selecting calculators
=#

using Random

# Note: utility functions from utils.jl are available in the same module scope


"""
    run_pp3_many(workdir, indir, outdir, seedfile; n_parallel=1, keep=false)

Run PP3 calculations on multiple structures in parallel.

# Arguments
- `workdir`: Working directory for temporary files
- `indir`: Directory containing input .res files
- `outdir`: Directory for output structures
- `seedfile`: Seed file with PP3 pseudopotentials
- `n_parallel`: Number of parallel processes (default: 1)
- `keep`: Keep intermediate files (default: false)

# Returns
- Nothing (results written to outdir)
"""
function run_pp3_many(workdir, indir, outdir, seedfile; n_parallel=1, keep=false)
    files = glob_allow_abs(joinpath(indir, "*.res"))
    ensure_dir(workdir)

    for file in files
        working_path = joinpath(workdir, splitpath(file)[end])
        cp(file, working_path, force=true)

        try
            run_pp3(working_path, seedfile, joinpath(outdir, splitpath(file)[end]))
        catch error
            if typeof(error) <: ArgumentError
                @warn "Failed to calculate energy for $(file)"
                continue
            end
            throw(error)
        end

        # Clean up intermediate files
        if !keep
            for suffix in [".cell", ".conv", "-out.cell", ".pp", ".res"]
                fname = swapext(working_path, suffix)
                isfile(fname) && rm(fname)
            end
        end
    end
end

"""
    run_pp3(file, seedfile, outname=nothing)

Run PP3 single-point calculation on a structure.

# Arguments
- `file`: Input structure file (.res or .cell)
- `seedfile`: Seed file with PP3 pseudopotentials
- `outname`: Output file name (optional)

# Returns
- Cell object with calculated energy and pressure
"""
function run_pp3(file, seedfile, outname=nothing)
    if endswith(file, ".res")
        cell = CellBase.read_res(file)
        # Write as cell file
        CellBase.write_cell(swapext(file, ".cell"), cell)
    else
        cell = CellBase.read_cell(file)
    end

    # Copy pseudopotential file
    cp(swapext(seedfile, ".pp"), swapext(file, ".pp"), force=true)

    # Run pp3
    aname = relpath(splitext(file)[1])

    # PP3 has size limit for seed name
    if length(aname) > 70
        enthalpy, pressure = mktempdir() do tempd
            for suffix in ["cell", "pp"]
                symlink(aname * ".$(suffix)", abspath(joinpath(tempd, "seed.$(suffix)")))
            end
            seed = joinpath(tempd, "seed")
            _call_pp3(seed)
        end
    else
        enthalpy, pressure = _call_pp3(aname)
    end

    # Store results
    cell.metadata[:enthalpy] = enthalpy
    cell.metadata[:pressure] = pressure
    cell.metadata[:label] = stem(file)

    if !isnothing(outname)
        CellBase.write_res(outname, cell)
    end

    cell
end

"""
    _call_pp3(seedname)

Call the pp3 executable to calculate energy and pressure.

# Arguments
- `seedname`: Base name of input files (without extension)

# Returns
- Tuple (enthalpy, pressure)
"""
function _call_pp3(seedname)
    enthalpy = 0.0
    pressure = 0.0

    for line in eachline(pipeline(`pp3 -n $(seedname)`))
        if contains(line, "Enthalpy")
            enthalpy = parse(Float64, split(line)[end])
        end
        if contains(line, "Pressure")
            pressure = parse(Float64, split(line)[end])
        end
    end

    enthalpy, pressure
end


"""
    run_crud(workdir, seedfile_calc, indir, outdir; nparallel=1, mpinp=4)

Use `crud.pl` to calculate energies of all files in the input folder.

Files are assumed to be named like `SEED-XX-XX-XX.res` and calculation
parameters are stored under `<workdir>/SEED.cell` and `<workdir>/SEED.param`.

# Arguments
- `workdir`: Working directory for crud.pl
- `seedfile_calc`: Seed file for CASTEP calculations
- `indir`: Directory containing input .res files
- `outdir`: Directory for output structures
- `nparallel`: Number of parallel crud.pl instances
- `mpinp`: Number of MPI processes per calculation

# Returns
- Nothing (results written to outdir)
"""
function run_crud(workdir, seedfile_calc, indir, outdir; nparallel=1, mpinp=4)
    hopper_folder = joinpath(workdir, "hopper")
    gd_folder = joinpath(workdir, "good_castep")

    ensure_dir(workdir)

    # Copy seed files
    seed_calc = stem(seedfile_calc)
    cp(seed_calc * ".cell", joinpath(workdir, seed_calc * ".cell"), force=true)
    cp(seed_calc * ".param", joinpath(workdir, seed_calc * ".param"), force=true)

    ensure_dir(hopper_folder)
    ensure_dir(outdir)

    infiles = glob_allow_abs(joinpath(indir, "*.res"))

    # Copy files to hopper
    for file in infiles
        cp(file, joinpath(hopper_folder, stem(file) * ".res"), force=true)
    end

    # Run parallel crud.pl instances
    @sync begin
        for i = 1:nparallel
            @async run(Cmd(`crud.pl -mpinp $mpinp`, dir=workdir))
            sleep(0.01)
        end
    end

    # Transfer results to output folder
    for file in infiles
        fname = stem(file) * ".res"
        isfile(joinpath(gd_folder, fname)) &&
            mv(joinpath(gd_folder, fname), joinpath(outdir, fname), force=true)

        # Copy CASTEP files if they exist
        fname = stem(file) * ".castep"
        isfile(joinpath(gd_folder, fname)) &&
            mv(joinpath(gd_folder, fname), joinpath(outdir, fname), force=true)
    end
end

"""
    run_acrud(workdir, indir, outdir; batch_size=1, verbose=false,
              exec="python singlepoint.py", keep=false)

Use acrud (ASE Calculation Runner Daemon) for singlepoint calculations.

Launches many calculations in parallel using a Python/ASE script.

# Arguments
- `workdir`: Working directory
- `indir`: Directory containing input structures
- `outdir`: Directory for output structures
- `batch_size`: Number of structures per batch
- `verbose`: Print verbose output
- `exec`: Executable command for calculations
- `keep`: Keep working directory files

# Returns
- Nothing (results written to outdir)
"""
function run_acrud(
    workdir,
    indir,
    outdir;
    batch_size=1,
    verbose=false,
    exec="python singlepoint.py",
    keep=false,
)
    files = glob_allow_abs(joinpath(indir, "*.res"))
    ensure_dir(workdir)
    ensure_dir(joinpath(workdir, "hopper"))

    # Copy files to hopper directory
    for file in files
        cp(file, joinpath(workdir, "hopper", splitdir(file)[end]), force=true)
    end

    # Run acrud command
    project_path = dirname(Base.active_project())
    cmd = Cmd([
        Base.julia_cmd()...,
        "--project=$(project_path)",
        "-e",
        "using EDDPotentials;EDDPotentials.acrud(\"$workdir\";exec=\"$exec\",batch_size=$batch_size,verbose=$verbose)",
    ])
    run(cmd)

    # Copy results to output directory
    verbose && @info "Copying results to $outdir"
    for file in glob_allow_abs(joinpath(workdir, "good_castep", "*.res"))
        verbose && @info "Copying $file"
        dst = joinpath(outdir, splitdir(file)[end])
        mv(file, dst, force=true)
    end

    # Clean up if not keeping files
    if !keep
        for folder in ["hopper", "good_castep", "bad_castep"]
            run(`rm -r $workdir/$folder`)
        end
    end
end

"""
    acrud(workdir; infiles=nothing, nostop=false, exec="python singlepoint.py",
          move_good=true, verbose=false, batch_size=1)

ASE Calculation Run Daemon - monitors folder and runs calculations on new files.

Watches the `hopper` folder for new files, runs calculations, and sorts results
into `good_castep` or `bad_castep` folders based on exit status.

# Arguments
- `workdir`: Working directory (must contain hopper/ subdirectory)
- `infiles`: Initial files to process (optional)
- `nostop`: Continue running even when no files present
- `exec`: Executable command (must accept multiple input files)
- `move_good`: Move good calculation files to good_castep folder
- `verbose`: Print verbose output
- `batch_size`: Number of files to process per batch

# Returns
- Nothing (daemon runs until no files remain or stopped)
"""
function acrud(
    workdir;
    infiles=nothing,
    nostop=false,
    exec="python singlepoint.py",
    move_good=true,
    verbose=false,
    batch_size=1,
)
    hopper_folder = joinpath(workdir, "hopper")
    gd_folder = joinpath(workdir, "good_castep")
    bd_folder = joinpath(workdir, "bad_castep")
    ensure_dir(hopper_folder)
    ensure_dir(gd_folder)
    ensure_dir(bd_folder)

    # Copy initial input files
    if infiles !== nothing
        for file in infiles
            cp(file, joinpath(hopper_folder, splitdir(file)[2]), force=true)
        end
    end

    while true
        # Select files from hopper
        infiles = glob_allow_abs(joinpath(hopper_folder, "*.res"))

        # Wait or stop if no files
        if length(infiles) == 0
            !nostop && break
            sleep(5)
            continue
        end

        # Select batch
        batch_size = min(batch_size, length(infiles))
        selected_files = shuffle(infiles)[1:batch_size]
        to_run = []

        for selected in selected_files
            flag = run(`mv $selected $workdir`)
            if flag.exitcode == 0
                push!(to_run, selected)
            end
        end

        length(to_run) == 0 && continue

        for selected in to_run
            println(splitext(splitdir(selected)[2])[1])
        end

        # Build command
        fnames = [joinpath(workdir, splitdir(selected)[2]) for selected in to_run]
        verbose && @info "Processing files: $fnames"

        cmd = Cmd(String[split(exec)..., fnames...])
        verbose && @show cmd
        flag = run(Cmd(cmd))

        # Handle results
        if flag.exitcode == 0
            # Success - convert and move to good folder
            for fname in fnames
                verbose && @info "Converting and moving $fname to $gd_folder"
                extxyz2res(splitext(fname)[1] * ".extxyz", splitext(fname)[1] * ".res")

                cp(
                    splitext(fname)[1] * ".res",
                    joinpath(gd_folder, stem(fname) * ".res"),
                    force=true,
                )

                if move_good
                    debug_files = glob_allow_abs(splitext(fname)[1] * ".*")
                    for file in debug_files
                        mv(file, joinpath(gd_folder, splitdir(file)[2]), force=true)
                    end
                end
            end
        else
            # Failure - move to bad folder
            for fname in fnames
                debug_files = glob_allow_abs(splitext(fname)[1] * ".*")
                for file in debug_files
                    mv(file, joinpath(bd_folder, splitdir(file)[2]), force=true)
                end
            end
        end
    end
end

"""
    read_extxyz(extxyz_file)

Read an Extended XYZ file and return a Cell object.
"""
function read_extxyz(extxyz_file)
    data = read_frame(extxyz_file)
    arrays = Dict{Symbol,AbstractArray}()
    metadata = Dict{Symbol,Any}()

    # Parse info and arrays
    for (key, value) in data["info"]
        metadata[Symbol(key)] = value
    end

    for (key, value) in data["arrays"]
        if !(key in ["species", "pos"])
            arrays[Symbol(key)] = value
        end
    end

    Cell(
        Lattice(collect(transpose(data["cell"]))),
        Symbol.(data["arrays"]["species"]),
        data["arrays"]["pos"],
        arrays,
        metadata,
    )
end

"""
    extxyz2res(extxyz_file, res_file)

Convert an Extended XYZ file to SHELX .res format.
"""
function extxyz2res(extxyz_file, res_file)
    cell = read_extxyz(extxyz_file)

    # Map metadata keys
    mapping = [
        :energy => :enthalpy,
        :name => :label,
        :spacegroup => :symm,
        :times_found => :flag3,
    ]

    for (key1, key2) in mapping
        haskey(cell.metadata, key1) || continue
        cell.metadata[key2] = cell.metadata[key1]
    end

    # Ensure space group is properly formatted
    if :symm in keys(cell.metadata)
        value = cell.metadata[:symm]
        if !(value[1] == '(' && value[end] == ')')
            cell.metadata[:symm] = "($value)"
        end
    else
        cell.metadata[:symm] = "(P1)"
    end

    write_res(res_file, cell)
end


"""
    MockCalculator

Mock external calculator for testing without actual DFT dependencies.

Generates random energies for structures to enable testing of workflows
without requiring external programs like PP3 or CASTEP.

# Fields
- `n_structures`: Maximum number of structures to process
- `success_rate`: Probability of successful calculation (0.0-1.0)
- `delay`: Simulated computation time per structure (seconds)
"""
struct MockCalculator
    n_structures::Int
    success_rate::Float64
    delay::Float64
end

"""
    run_mock_calculations(calc::MockCalculator, indir, outdir)

Run mock calculations that simulate DFT without actually computing anything.

# Arguments
- `calc`: MockCalculator instance
- `indir`: Directory containing input .res files
- `outdir`: Directory for output structures

# Returns
- `true` on success
"""
function run_mock_calculations(calc::MockCalculator, indir, outdir)
    files = glob_allow_abs(joinpath(indir, "*.res"))
    n_process = min(calc.n_structures, length(files))

    ensure_dir(outdir)

    for i in 1:n_process
        if rand() < calc.success_rate
            # Read structure
            cell = CellBase.read_res(files[i])

            # Add mock energy (random value between -10 and 0 eV/atom)
            cell.metadata[:enthalpy] = rand() * -10.0 * length(cell.atoms)
            cell.metadata[:pressure] = 0.1  # Mock pressure

            # Write result
            outfile = joinpath(outdir, basename(files[i]))
            CellBase.write_res(outfile, cell)
        end

        # Simulate computation time
        sleep(calc.delay)
    end

    return true
end


"""
    get_external_calculator(builder::Builder)

Create appropriate external calculator based on builder configuration.

Dispatches to the correct local DFT backend based on `builder.state.dft_mode`.
Scheduler-backed local DFT modes have been superseded by the distributed link backend.

- "mock": Mock calculator for testing
- "pp3": PP3 pairwise potential calculations
- "crud": Local CRUD runner
- "acrud": ASE-based CRUD runner

# Arguments
- `builder`: Builder object with configuration

# Returns
- Function that runs calculations: `(indir, outdir) -> Bool`

# Throws
- ErrorException if dft_mode is unknown
"""
function get_external_calculator(builder::Builder)
    state = builder.state

    if state.dft_mode == "pp3"
        return (indir, outdir) -> (
            run_pp3_many(
                joinpath(state.workdir, ".pp3_work"),
                indir,
                outdir,
                joinpath(state.workdir, state.seedfile_calc);
                n_parallel=state.n_parallel,
                make_symbol_keys(state.dft_kwargs)...,
            );
            true
        )

    elseif state.dft_mode == "crud"
        return (indir, outdir) -> (
            run_crud(
                joinpath(state.workdir, "crud-work"),
                state.seedfile_calc,
                indir,
                outdir;
                make_symbol_keys(state.dft_kwargs)...,
            );
            true
        )

    elseif state.dft_mode == "acrud"
        return (indir, outdir) -> (
            run_acrud(
                joinpath(state.workdir, "acrud-work"),
                indir,
                outdir;
                make_symbol_keys(state.dft_kwargs)...,
            );
            true
        )

    elseif state.dft_mode == "mock"
        calc = MockCalculator(100, 0.95, 0.1)
        return (indir, outdir) -> run_mock_calculations(calc, indir, outdir)

    else
        throw(ErrorException("Unknown dft_mode: $(state.dft_mode)"))
    end
end
