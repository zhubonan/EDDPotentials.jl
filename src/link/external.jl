#=
Routines for running external code
=#

"""
    run_crud(workdir, indir, outdir;nparallel=1, mpinp=4)

Use `crud.pl` to calculate energies of all files in the input folder and store
the results to the output folder.
It is assumed that the files are named like `SEED-XX-XX-XX.res` and the parameters
for calculations are stored under `<workdir>/SEED.cell` and `<workdir>/SEED.param`. 
"""
function run_crud(workdir, indir, outdir; nparallel=1, mpinp=4)
    hopper_folder = joinpath(workdir, "hopper")
    gd_folder = joinpath(workdir, "good_castep")
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
            @async run(`crud.pl -singlepoint -mpinp $mpinp`)
            sleep(0.01)
        end
    end
    # Transfer files to the target folder
    for file in infiles
        fname = stem(file) * ".res"
        isfile(joinpath(gd_folder, fname)) &&
            cp(joinpath(gd_folder, fname), joinpath(outdir, fname), force=true)
        # Copy CASTEP files is there is any
        fname = stem(file) * ".castep"
        isfile(joinpath(gd_folder, fname)) &&
            cp(joinpath(gd_folder, fname), joinpath(outdir, fname), force=true)
    end
end

### Scheduler interface 

function submit(sch::SchedulerConfig, workdir)
    if sch.type == "SGE"
        return submit_sge(sch, workdir)
    end
    throw(ErrorException("Unknown scheduler type $(sch.type)"))
end

"""
    submit(sch::SGEScheduler, workdir)

Submit the jobs
"""
function submit_sge(sch::SchedulerConfig, workdir)
    # Write the job script
    script_path = joinpath(workdir, "job.sh")
    open(script_path, "w") do handle
        write(handle, get_job_script_content(sch))
    end

    # Submit the jobs
    prog = "qsub"

    args = String[]
    njobs = sch.njobs
    if njobs > 1
        append!(args, String["-t", "1-$(njobs)"])
    end

    # Setup the working directory
    push!(args, "-cwd")

    # Include the job script
    push!(args, "job.sh")

    # Run the submission command
    cmd = Cmd(Cmd([prog, args...]), dir=workdir)
    @info "Job submission command: $(cmd)"
    run(cmd)
end

### END of scheduler interface code


"""
Run crud.pl through a queuing system
"""
function run_crud_queue(
    scheduler::SchedulerConfig,
    seedfile,
    workdir,
    input_dir,
    output_dir,
    ;
    perc_threshold=0.98,
)

    isdir(workdir) || mkdir(workdir)
    isdir(joinpath(workdir, "hopper")) || mkdir(joinpath(workdir, "hopper"))

    # Copy the SHELX files to the folder
    nstruct = 0
    for file in glob_allow_abs(joinpath(input_dir, "*.res"))
        cp(file, joinpath(workdir, "hopper", splitpath(file)[end]), force=true)
        nstruct += 1
    end

    # Copy the seed file to the working directory
    seed = stem(seedfile)
    cp(seed * ".cell", joinpath(workdir, seed * ".cell"), force=true)
    cp(seed * ".param", joinpath(workdir, seed * ".param"), force=true)
    good_castep = joinpath(workdir, "good_castep")

    nfinished = length(glob(joinpath(good_castep, "*.res")))
    perc_finished = nfinished / nstruct

    # Launch jobs
    if perc_finished < perc_threshold && scheduler.submit_jobs
        # Submit the jobs
        submit(scheduler, workdir)
    end

    # Monitor the status
    while true
        nfinished = length(glob(joinpath(good_castep, "*.res")))
        perc_finished = nfinished / nstruct
        # Check if we have enough number of structures
        perc_finished > perc_threshold && break
        sleep(600)
    end

    # Copy the structures to the staged folder for training
    for file in glob_allow_abs(joinpath(good_castep, "*.res"))
        cp(file, joinpath(output_dir, splitpath(file)[end]), force=true)
    end

    # Clean up the working directory
    if scheduler.clean_workdir
        rm(workdir, recursive=true)
    end

    return true
end
