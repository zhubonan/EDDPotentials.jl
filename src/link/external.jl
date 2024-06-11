#=
Routines for running external code
=#
using Random
using ExtXYZ

"""
    run_crud(workdir, indir, outdir;nparallel=1, mpinp=4)

Use `crud.pl` to calculate energies of all files in the input folder and store
the results to the output folder.
It is assumed that the files are named like `SEED-XX-XX-XX.res` and the parameters
for calculations are stored under `<workdir>/SEED.cell` and `<workdir>/SEED.param`. 
"""
function run_crud(workdir, seedfile_calc, indir, outdir; nparallel=1, mpinp=4)
    hopper_folder = joinpath(workdir, "hopper")
    gd_folder = joinpath(workdir, "good_castep")

    ensure_dir(workdir)
    # Copy the seed files storing the calculation settings to the working directory
    seed_calc = stem(seedfile_calc)
    cp(seed_calc * ".cell", joinpath(workdir, seed_calc * ".cell"), force=true)
    cp(seed_calc * ".param", joinpath(workdir, seed_calc * ".param"), force=true)

    ensure_dir(hopper_folder)
    ensure_dir(outdir)
    infiles = glob(joinpath(indir, "*.res"))
    # Copy files to the hopper folder
    for file in infiles
        cp(file, joinpath(hopper_folder, stem(file) * ".res"), force=true)
    end
    # Run nparallel instances of crud.pl
    @sync begin
        for i = 1:nparallel
            @async run(Cmd(`crud.pl -mpinp $mpinp`, dir=workdir))
            sleep(0.01)
        end
    end
    # Transfer files to the target folder
    for file in infiles
        fname = stem(file) * ".res"
        isfile(joinpath(gd_folder, fname)) &&
            mv(joinpath(gd_folder, fname), joinpath(outdir, fname), force=true)
        # Copy CASTEP files is there is any
        fname = stem(file) * ".castep"
        isfile(joinpath(gd_folder, fname)) &&
            mv(joinpath(gd_folder, fname), joinpath(outdir, fname), force=true)
    end
end

### Scheduler interface 

# function submit(sch::SchedulerConfig, workdir)
#     if sch.type == "SGE"
#         return submit(sch, workdir)
#     elseif sch.type == "SLURM"
#         return submit_slurm(sch, workdir)
#     end
#     throw(ErrorException("Unknown scheduler type $(sch.type)"))
# end

"""
    submit(sch::SGEScheduler, workdir)

Submit the jobs
"""
function submit(sch::SchedulerConfig, workdir)
    # Write the job script
    script_path = joinpath(workdir, "job.sh")
    open(script_path, "w") do handle
        write(handle, get_job_script_content(sch))
    end

    if sch.type == "SGE"
    # Submit the jobs
        prog = "qsub"
    elseif sch.type == "SLURM"
        prog = "sbatch"
    end


    args = String[]
    njobs = sch.njobs
    if njobs > 1
        sch.type == "SGE" && append!(args, String["-t", "1-$(njobs)"])
        sch.type == "SLURM" && append!(args, String["--array=1-$(njobs)"])
    end
    # Setup the working directory for SGE
    sch.type == "SGE" && push!(args, "-cwd")

    # Include the job script
    push!(args, "job.sh")

    # Run the submission command
    cmd = Cmd(Cmd([prog, args...]), dir=workdir)
    @info "Job submission command: $(cmd)"
    run(cmd)
end

### END of scheduler interface code


"""
Run crud.pl/acrud function through a queuing system
"""
function run_crud_queue(
    scheduler::SchedulerConfig,
    seedfile,
    workdir,
    input_dir,
    output_dir,
    ;
    perc_threshold=0.98,
    script_path="",
    mode="crud",
)

    isdir(workdir) || mkdir(workdir)

    # Copy the job script to the working directory
    if !isempty(script_path)
        cp(script_path, joinpath(workdir, "job.sh"), force=true)
    end

    # Clear existing folders
    for dir in ["hopper", "good_castep", "bad_castep"]
        path = joinpath(workdir, dir)
        isdir(path) && rm(path, recursive=true)
    end

    # Create the hopper folder
    mkdir(joinpath(workdir, "hopper"))

    # Copy the SHELX files to the folder
    nstruct = 0
    for file in glob_allow_abs(joinpath(input_dir, "*.res"))
        cp(file, joinpath(workdir, "hopper", splitpath(file)[end]), force=true)
        nstruct += 1
    end

    # Copy the seed file to the working directory this is needed for crud.pl
    if mode == "crud"
        seed = stem(seedfile)
        cp(seed * ".cell", joinpath(workdir, seed * ".cell"), force=true)
        cp(seed * ".param", joinpath(workdir, seed * ".param"), force=true)
    end

    # Copy the executable script to the working directory if it is not already there
    if mode == "acrud"
        dst = joinpath(workdir, splitdir(seedfile)[end])
        if dst != seedfile
            cp(seedfile, workdir, force=true)
        end
    end

    # Create the necessary folders
    good_castep = joinpath(workdir, "good_castep")
    bad_castep = joinpath(workdir, "bad_castep")
    nfinished = length(glob(joinpath(good_castep, "*.res"))) + length(glob(joinpath(bad_castep, "*.res")))
    perc_finished = nfinished / nstruct

    # Launch jobs
    if perc_finished < perc_threshold && scheduler.submit_jobs
        # Submit the jobs
        submit(scheduler, workdir)
    end

    # Monitor the status
    while true
        nfinished = length(glob(joinpath(good_castep, "*.res"))) + length(glob(joinpath(bad_castep, "*.res")))
        perc_finished = nfinished / nstruct
        # Check if we have enough number of structures
        perc_finished > perc_threshold && break
        sleep(60)
    end

    # Move the structures to the staged folder for training
    for file in glob_allow_abs(joinpath(good_castep, "*.res"))
        mv(file, joinpath(output_dir, splitpath(file)[end]), force=true)
    end

    # Clear the bad files 
    for file in glob_allow_abs(joinpath(bad_castep, "*.res"))
        rm(file)
    end

    # Clean up the working directory
    if scheduler.clean_workdir
        rm(workdir, recursive=true)
    end

    return true
end

"""
    arcrud(workdir;infiles=nothing, nostop=false, exec="python singlepoint.py", 
                       move_good=true, verbose=false, batch_size=1)

ASE/all calculation run daemon.

Daemon process for monitoring a folder and running calculations on new files.
Watch the `hopper` folder in the working director for new files to run calculations.
Good results are stored in the `good_castep` folder.
Bad results are stored in the `bad_castep` folder.
The executable for running calculations is `exec` and should support such syntax:
```
<exec> <input_file2> [<input_file2> <input_file3>] ...
```

and return 0 if the calculation is successful and non-zero otherwise.
The output file should has the same name as the input file but with an extension of `.extxyz`.

The extxyz should have the a property called `energy` which is the total energy of the structure.
The forces may be included but they are optional.

Supported for multiple input files is optional and only needed if `batch_size` is larger than 1.
"""
function acrud(workdir;infiles=nothing, nostop=false, exec="python singlepoint.py", 
                       move_good=true, verbose=false, batch_size=1)
    hopper_folder = joinpath(workdir, "hopper")
    gd_folder = joinpath(workdir, "good_castep")
    bd_folder = joinpath(workdir, "bad_castep")
    ensure_dir(hopper_folder)
    ensure_dir(gd_folder)
    ensure_dir(bd_folder)

    # Copy all input files to the hopper folder
    if infiles !== nothing
        for file in infiles
            cp(file, joinpath(hopper_folder, splitdir(file)[2]), force=true)
        end
    end

    while true
        # Select file
        infiles = glob(joinpath(hopper_folder, "*.res"))
        # Check if there isn't any file, wait or die
        if length(infiles) == 0
            !nostop && break
            sleep(5)
            continue
        end
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
        # Path to the file
        fnames = [joinpath(workdir, splitdir(selected)[2])  for selected in to_run]
        verbose && @info "Path of the files to be processed: $fnames"
        # Run the executable
        cmd = Cmd(String[split(exec)..., fnames...])
        verbose && @show  cmd
        flag = run(Cmd(cmd))
        # Check the exit status and act accordingly
        if flag.exitcode == 0 
            # Need to implement this function extxyz2res
            for fname in fnames
                verbose && @info "Convert and move $fname to $gd_folder"
                extxyz2res(splitext(fname)[1] * ".extxyz", splitext(fname)[1]* ".res")
                # Copy to the good castep folder
                cp(splitext(fname)[1] * ".res", joinpath(gd_folder, stem(fname) * ".res"), force=true)
                # Should one copy the good files to the good_castep folder?
                if move_good
                    debug_files = glob(splitext(fname)[1] * ".*")
                    for file in debug_files
                        mv(file, joinpath(gd_folder, splitdir(file)[2]), force=true)
                    end
                end
            end
        else
            # Calculation failed - move every thing to the bad_castep folder
            for fname in fnames
                debug_files = glob(splitext(fname)[1] * ".*")
                for file in debug_files
                    mv(file, joinpath(bd_folder, splitdir(file)[2]), force=true)
                end
            end
        end
    end
end

"""
    read_extxyz(extxyz_file)

Read an extxyz file and return a Cell object.
"""
function read_extxyz(extxyz_file)
    data = read_frame(extxyz_file)
    arrays = Dict{Symbol, AbstractArray}()
    metadata = Dict{Symbol, Any}()

    # Pass the info and arrays field
    for (key, value) in data["info"]
        metadata[Symbol(key)] = value
    end

    for (key, value) in data["arrays"]
        if !(key in ["species", "pos"])
            arrays[Symbol(key)] = value
        end
    end

    Cell(
        Lattice(collect(transpose(data["cell"]))),  # Convert to column vectors
        Symbol.(data["arrays"]["species"]),
        data["arrays"]["pos"],
        arrays,
        metadata
    )
end

"""
    extxyz2res(extxyz_file, res_file)

Convert an extxyz file to a res file.
"""
function extxyz2res(extxyz_file, res_file)
    cell = read_extxyz(extxyz_file)
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
    # Check if the space group is quoted by brackets
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