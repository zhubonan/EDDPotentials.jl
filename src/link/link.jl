#=
Code for iteratively building the model
=#
import Base
using Parameters
using JSON
using YAML
using TOML
using ArgParse

const XT_NAME = "xt"
const YT_NAME = "yt"
const FEATURESPEC_NAME = "cf"

include("options.jl")



"""
The Builder stores states and options for training and using the potentials.
"""
mutable struct Builder
    cf::CellFeature
    cf_embedding::Union{Nothing,CellEmbedding}
    state::BuilderState
    rss::RssSetting
    trainer::TrainingOption
    cfopt::CellFeatureConfig
    options::BuilderOption
end


function Builder(options::BuilderOption)
    cf = CellFeature(options.cf)
    if options.cf_embedding !== nothing
        embed = CellEmbedding(cf, options.cf_embedding.n, options.cf_embedding.m)
    else
        embed = nothing
    end

    builder =
        Builder(cf, embed, options.state, options.rss, options.trainer, options.cf, options)
    _set_iteration!(builder)
    # Populate some default fields
    if builder.rss.seedfile == "null"
        builder.rss.seedfile = builder.state.seedfile
        @warn "Using seed file: $(builder.rss.seedfile)"
    end

    if builder.rss.ensemble_id == -1
        builder.rss.ensemble_id = builder.state.iteration
        @warn "Using default ensemble id: $(builder.rss.ensemble_id)"
    end

    builder_uuid(builder)
    builder
end

function get_energy_per_atom(fc::FeatureContainer)
    fc.H ./ num_atoms(fc)
end

num_atoms(fc::FeatureContainer) = size.(fc.fvecs, 2)
num_atoms(sc::StructureContainer) = length(sc.structures)

add_threads_env(cmd, threads) = addenv(cmd, "JULIA_NUM_THREADS" => threads)


"""
    save_builder(fname::AbstractString, builder)

Save a `Builder` to a file.
"""
function save_builder(fname::AbstractString, builder::Builder)
    open(fname, "w") do f
        to_toml(f, builder.options)
    end
end


"""
    Builder(str::AbstractString="link.toml")

Load the builder from a YAML file. The file contains nested key-values pairs
similar to the constructors of the types.

```yaml
state:
    seedfile : "test.cell"

trainer:
    type : "locallm"
    nmodels : 128
cf:
    elements : ["H", "O"]
    # Power of the polynomials as geometry sequence 
    p2       : [2, 10, 5] 
    geometry_sequence : true

cf_embedding:
    n : 3
```

"""
function Builder(str::AbstractString="link.toml")
    @info "Loading from file $(str)"

    if endswith(str, ".toml")
        builder_opts = from_toml(BuilderOption, str)
    else
        dict = YAML.load_file(str; dicttype=Dict{String,Any})
        # Save converged TOML if requested
        toml = splitext(str)[1] * ".toml"
        if !isfile(toml)
            @info "Saving converted TOML configuration at $toml."
            open(toml, "w") do io
                TOML.print(io, dict)
            end
        end
        builder_opts = from_dict(BuilderOption, dict)
    end

    builder = Builder(builder_opts)

    # Adjust the workdir to be that relative to the yaml file
    paths = splitpath(str)
    if length(paths) > 1
        parents = paths[1:end-1]
        @assert !startswith(builder.state.workdir, "/") ":workdir should be a relative path"
        builder.state.workdir = joinpath(parents..., builder.state.workdir)
        @info "Setting workdir to $(builder.state.workdir)"
    end

    # Store the path to the builder file
    builder.state.builder_file_path = str

    # Update the iteration number with the new path
    _set_iteration!(builder)

    builder
end

function Base.show(io::IO, m::MIME"text/plain", bu::Builder)
    println(io, "Builder:")
    println(io, "  Working directory: $(bu.state.workdir) ($(abspath(bu.state.workdir)))")
    println(io, "  Iteration: $(bu.state.iteration)")
    println(io, "  Seed file: $(bu.state.seedfile)")
    println("\nState: ")
    show(io, m, bu.state)
    println("\nTrainer: ")
    show(io, m, bu.trainer)
    println("\nRSS: ")
    show(io, m, bu.rss)
    println("\nCellFeature (one/two/tree-body): ")
    show(io, m, feature_size(bu.cf))
    println("\nEmbedding: ")
    show(io, m, bu.cf_embedding)
end

function Base.show(io::IO, bu::Builder)
    println(io, "Builder:")
    println(io, "  Working directory: $(bu.state.workdir) ($(abspath(bu.state.workdir)))")
    println(io, "  Iteration: $(bu.state.iteration)")
    println(io, "  Seed file: $(bu.state.seedfile)")
end


"""
    _set_iteration!(builder::Builder)

Fastword to the iteration by checking existing ensemble files
"""
function _set_iteration!(builder::Builder)
    # Set the iteration number
    for iter = 0:builder.state.max_iterations
        builder.state.iteration = iter
        if !has_ensemble(builder, iter)
            break
        end
    end
end

"""
    link!(builder::Builder)

Run automated iterative building cycles.
"""
function link!(builder::Builder)
    state = builder.state
    names = ["crud.pl", "buildcell", "cabal"]
    @info "Check if $names are available."
    for name in names
        if run(`which $name`).exitcode != 0
            @info "External program $name not found! Please check the runtime environment."
        end
    end
    while state.iteration <= state.max_iterations
        step!(builder)
        if should_stop(builder)
            @warn "Aborted training loop at iteration: $(state.iteration)."
            return
        end
    end
end

"""
    link()
Run `link!` from command line interface.

Example: 

```bash
julia -e "Using EDDPotentials;EDDPotentials.link()" -- --file "link.toml" 
```
"""
function link()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--file"
        help = "Name of the yaml file"
        default = "link.toml"
        arg_type = String
        "--iter"
        help = "Override the iteration number"
        arg_type = Int
        default = -1
    end
    args = parse_args(s)
    fname = args["file"]
    builder = Builder(fname)
    if args["iter"] >= 0
        builder.state.iteration = args["iter"]
    end
    @info "Loading builder from $(fname)"
    link!(builder)
end

function should_stop(bu::Builder)
    if isfile(joinpath(bu.state.workdir, "STOP"))
        @warn "STOP file detected, aborting...."
        return true
    end
    return false
end

ensure_dir(args...) = isdir(joinpath(args...)) || mkdir(joinpath(args...))

"""
Run a step step for the Builder
"""
function step!(bu::Builder)
    should_stop(bu) && return
    iter = bu.state.iteration
    if has_ensemble(bu, iter)
        bu.state.iteration += 1
        return bu
    end
    # Generate new structures
    _generate_random_structures(bu, iter)
    should_stop(bu) && return

    # Run external code for 
    if !is_training_data_ready(bu, iter)
        @info "Starting energy calculations for iteration $iter."
        flag = _run_external(bu)
        should_stop(bu) && return
        flag || throw(
            ErrorException("Cannot find external code to run for generating training data"),
        )
    end
    ns = nstructures(bu, iter)
    @info "Number of new structures in iteration $(iter): $ns"

    # Optional - run walk-forward test
    if bu.state.run_walk_forward
        walk_forward_tests(bu; print_results=true, iters=[iter - 1])
        should_stop(bu) && return
    end

    # Retrain model
    @info "Starting training for iteration $iter."
    _perform_training(bu)

    bu.state.iteration += 1
    bu
end

"""
    _generate_random_structures(bu::Builder, iter)

Generate random structure (if needed) as training data for the `Builder`.
"""
function _generate_random_structures(bu::Builder, iter)
    # Generate new structures
    outdir = _input_structure_dir(bu)
    ensure_dir(outdir)
    ndata = length(glob_allow_abs(joinpath(outdir, "*.res")))
    (; seedfile, seedfile_weights, workdir) = bu.state
    if iter == 0
        # First cycle generate from the seed without relaxation
        # Sanity check - are we definitely overfitting?
        if nfeatures(bu.cf) > bu.state.n_initial
            @warn "The number of features $(nfeatures(bu.cf)) is larger than the initial training size!"
        end
        # Generate random structures
        nstruct = bu.state.n_initial - ndata
        if nstruct > 0
            @info "Genearating $(nstruct) initial training structures."
            build_random_structures(
                joinpath.(workdir, seedfile),
                outdir;
                seedfile_weights,
                n=nstruct,
            )
        end
    else
        # Subsequent cycles - generate from the seed and perform relaxation
        # Read ensemble file
        nstruct = bu.state.per_generation - ndata
        if nstruct > 0
            # Launch external processes
            if bu.state.rss_external
                _launch_rss_external(bu, iter, nstruct)
            else
                _launch_rss_internal(bu, iter, nstruct)
            end


            @info "Shaking generated structures."
            outdir = joinpath(bu.state.workdir, "gen$(iter)")
            shake_res(
                collect(glob_allow_abs(joinpath(outdir, "*.res"))),
                bu.state.shake_per_minima,
                bu.state.shake_amp,
                bu.state.shake_cell_amp,
            )
        end
    end
end

function _launch_rss_external(bu::Builder, iter::Int, nstruct::Int)
    @info "Generating $(nstruct) training structures for iteration $iter."

    # Distribute the workloads
    nstruct_per_proc = div(nstruct, bu.state.rss_nprocs) + 1
    nstruct_per_proc_last = nstruct % nstruct_per_proc
    nstructs = fill(nstruct_per_proc, bu.state.rss_nprocs)
    nstructs[end] = nstruct_per_proc_last

    # Launch external processes
    state = bu.state
    project_path = dirname(Base.active_project())
    outdir = joinpath(state.workdir, "gen$(iter)")
    cmds = [
        add_threads_env(
            Cmd([
                Base.julia_cmd()...,
                "--project=$(project_path)",
                "-e",
                "using EDDPotentials;EDDPotentials._run_rss_link()",
                "--",
                "--file",
                "$(state.builder_file_path)",
                "--iteration",
                "$(state.iteration)",
                "--num",
                "$(num)",
                "--pressure",
                "$(state.rss_pressure_gpa)",
                "--outdir",
                "$(outdir)",
            ]),
            state.rss_num_threads,
        ) for num in nstructs
    ]
    # Apply random pressure range
    if !isnothing(state.rss_pressure_gpa_range) && !isempty(state.rss_pressure_gpa_range)
        # Add pressure ranges
        a, b = state.rss_pressure_gpa_range
        for i in eachindex(cmds)
            cmds[i] = add_threads_env(
                Cmd([cmds[i]..., "--pressure-range", "$(a),$(b)"]),
                state.rss_num_threads,
            )
        end
    end

    @info "Subprocess launch command: $(Cmd([cmds[1]...])) for $(state.rss_nprocs) process"
    # Launch tasks
    tasks = Task[]
    for i = 1:state.rss_nprocs
        this_task = @async run(
            pipeline(cmds[i], stdout="rss-process-$i-stdout", stderr="rss-process-$i"),
        )
        push!(tasks, this_task)
    end

    @info "Search process launched, waiting..."
    last_nstruct = length(glob(joinpath(outdir, "*.res")))
    while !all(istaskdone.(tasks))
        nfound = length(glob(joinpath(outdir, "*.res")))
        # Print progress if the number of models have changed
        if nfound != last_nstruct
            @info "Number of relaxed structures: $(nfound)/$(state.per_generation)"
            last_nstruct = nfound
        end
        sleep(5)
    end
    @assert all(fetch(x).exitcode == 0 for x in tasks) "There are tasks with non-zero exit code!"
end

function _launch_rss_internal(bu::Builder, iter::Int, nstruct::Int)
    @info "Generating $(nstruct) training structures for iteration $iter."
    ensemble = load_ensemble(bu, iter - 1)
    state = bu.state
    outdir = joinpath(state.workdir, "gen$(iter)")
    ensure_dir(outdir)

    (; seedfile, seedfile_weights, ensemble_std_min, ensemble_std_max) = state
    _run_rss(
        joinpath.(Ref(state.workdir), seedfile),
        ensemble,
        bu.cf;
        show_progress=true,
        max=nstruct,
        core_size=state.core_size,
        outdir,
        ensemble_std_max=ensemble_std_max,
        ensemble_std_min=ensemble_std_min,
        packed=false,
        niggli_reduce_output=state.rss_niggli_reduce,
        max_err=10,
        pressure_gpa=state.rss_pressure_gpa,
        pressure_gpa_range=state.rss_pressure_gpa_range,
        seedfile_weights,
        relax_option=bu.state.relax,
        elemental_energies=_make_symbol_keys(state.elemental_energies),
    )
end



"Directory for input structures"
_input_structure_dir(bu::Builder) = joinpath(bu.state.workdir, "gen$(bu.state.iteration)")
"Directory for output structures after external calculations"
_output_structure_dir(bu::Builder) =
    joinpath(bu.state.workdir, "gen$(bu.state.iteration)-dft")


include("external.jl")
"""
Run external code for generating training data
"""
function _run_external(bu::Builder)
    ensure_dir(_output_structure_dir(bu))
    if bu.state.dft_mode == "disp-castep"
        if bu.state.project_prefix_override == ""
            project_prefix = "eddp.jl/$(builder_short_uuid(bu))"
        else
            project_prefix = bu.state.project_prefix_override
        end
        run_disp_castep(
            _input_structure_dir(bu),
            _output_structure_dir(bu),
            joinpath(bu.state.workdir, bu.state.seedfile_calc);
            project_prefix,
            threshold=bu.state.per_generation_threshold,
            _make_symbol_keys(bu.state.dft_kwargs)...,
        )
        return true
    elseif bu.state.dft_mode == "pp3"
        run_pp3_many(
            joinpath(bu.state.workdir, ".pp3_work"),
            _input_structure_dir(bu),
            _output_structure_dir(bu),
            joinpath(bu.state.workdir, bu.state.seedfile_calc);
            n_parallel=bu.state.n_parallel,
            _make_symbol_keys(bu.state.dft_kwargs)...,
        )
        return true
    elseif bu.state.dft_mode == "crud-queue"
        run_crud_queue(
            bu.options.scheduler,
            joinpath(bu.state.workdir, bu.state.seedfile_calc),
            joinpath(bu.state.workdir, "crud-work"),
            _input_structure_dir(bu),
            _output_structure_dir(bu);
            perc_threshold=bu.state.per_generation_threshold,
        )
        return true
    elseif bu.state.dft_mode == "acrud"
        run_acrud(
            joinpath(bu.state.workdir, "acrud-work"),
            _input_structure_dir(bu),
            _output_structure_dir(bu);
            _make_symbol_keys(bu.state.dft_kwargs)...,
        )
        return true
    elseif bu.state.dft_mode == "crud"
        run_crud(
            joinpath(bu.state.workdir, "crud-work"),
            bu.state.seedfile_calc,
            _input_structure_dir(bu),
            _output_structure_dir(bu);
            _make_symbol_keys(bu.state.dft_kwargs)...,
        )
        return true
    end
    return false
end
function _perform_training(bu::Builder)

    # Write the dataset to the disk
    @info "Preparing dataset..."
    write_dataset(bu)

    if bu.trainer.external
        _perform_training_external(bu)
    else
        run_trainer(bu, bu.trainer)
    end

    nm = num_existing_models(bu)
    @info "Number of trained models: $nm"

    # Create ensemble
    nm = num_existing_models(bu)
    if nm >= bu.trainer.nmodels * 0.9
        ensemble = create_ensemble(bu; save_and_clean=true)
    else
        throw(
            ErrorException(
                "Only $nm models are found in the training directory, need $(bu.trainer.nmodels)",
            ),
        )
    end
    return ensemble
end

"""
    _perform_training_external(bu::Builder)

Carry out training and save the ensemble as a JLD2 archive.
"""
function _perform_training_external(bu::Builder)

    tra = bu.trainer
    # Call multiple sub processes
    project_path = dirname(Base.active_project())
    builder_file = bu.state.builder_file_path
    @assert builder_file != ""
    cmd = Cmd(
        Cmd([
            Base.julia_cmd()...,
            "--project=$(project_path)",
            "-e",
            "using EDDPotentials;EDDPotentials.run_trainer()",
            "$(builder_file)",
            "--iteration",
            "$(bu.state.iteration)",
        ]),
        env=("OMP_NUM_THREADS" => "1",),
    )

    # Call multiple trainer processes
    @info "Subprocess launch command: $cmd"
    tasks = Task[]
    for i = 1:tra.num_workers
        # Run with ids
        _cmd = add_threads_env(Cmd([cmd..., "--id", "$i"]), tra.num_threads_per_worker)
        this_task = @async begin
            run(pipeline(_cmd, stdout="lm-process-$i-stdout", stderr="lm-process-$i-stderr"))
        end
        push!(tasks, this_task)
    end

    @info "Training processes launched, waiting..."

    last_nm = num_existing_models(bu)
    while !all(istaskdone.(tasks))
        nm = num_existing_models(bu)
        # Print progress if the number of models have changed
        if nm != last_nm
            @info "Number of trained models: $nm"
            last_nm = nm
        end
        sleep(30)
    end
end


function builder_uuid(workdir='.')
    fname = joinpath(workdir, ".eddp_builder")
    if isfile(fname)
        uuid = open(fname) do fh
            chomp(readline(fh))
        end
    else
        uuid = string(uuid4())
        open(fname, "w") do fh
            write(fh, uuid)
            write(fh, "\n")
        end
    end
    uuid
end

builder_uuid(bu::Builder) = builder_uuid(bu.state.workdir)
builder_short_uuid(x) = builder_uuid(x)[1:8]

"""
    _disp_get_completed_jobs(project_name)

Get the number of completed jobs as well as the total number of jobs under a certain project.
NOTE: this requires `disp` to be available in the commandline.
"""
function _disp_get_completed_jobs(project_name)
    cmd = `disp db summary --singlepoint --project $project_name --json`
    json_string = readchomp(pipeline(cmd))
    data = parse_disp_output(json_string)
    ncomp = get(data, "COMPLETED", 0)
    nall = get(data, "ALL", -1)
    ncomp, nall
end


"""
Run relaxation through DISP
"""
function run_disp_castep(
    indir,
    outdir,
    seedfile;
    categories,
    priority=90,
    project_prefix="eddp.jl",
    monitor_only=false,
    watch_every=60,
    threshold=0.98,
    disp_extra_args=String[],
    kwargs...,
)

    file_pattern = joinpath(indir, "*.res")
    seed = splitext(seedfile)[1]
    # Setup the inputs
    project_name = joinpath(project_prefix, abspath(indir)[2:end])
    seed_stem = splitext(basename(seedfile))[1]
    cmd = `disp deploy singlepoint --seed $seed_stem --base-cell $seed.cell --param $seed.param --cell $file_pattern --project $project_name --priority $priority`

    # Define the categories
    for category in categories
        push!(cmd.exec, "--category")
        push!(cmd.exec, category)
    end

    # Add extra argument for DISP
    append!(cmd.exec, disp_extra_args)

    if !monitor_only
        # Check if jobs have been submitted already
        ncomp, nall = _disp_get_completed_jobs(project_name)
        if nall == -1
            @info "Command to be run $(cmd)"
            run(cmd)
        else
            @info "There are already $(ncomp) out of $(nall) jobs completed - monitoring the progress next."
        end
    else
        @info "Not launching jobs - only watching for completion"
    end

    # Start to monitor the progress
    @info "Start watching for progress"
    sleep(1)
    last_ncomp = 0
    while true
        ncomp, nall = _disp_get_completed_jobs(project_name)
        if ncomp / nall > threshold
            @info " $(ncomp)/$(nall) calculation completed - moving on ..."
            break
        elseif ncomp != last_ncomp
            # Only update when there is any change
            @info "Completed calculations: $(ncomp)/$(nall) - waiting ..."
        end
        last_ncomp = ncomp
        sleep(watch_every)
    end
    # Pulling calculations down
    isdir(outdir) || mkdir(outdir)
    cmd = Cmd(`disp db retrieve-project --project $project_name`, dir=outdir)
    run(cmd)
    @info "Calculation results pulled into $outdir."
end


"""
    run_pp3_many(workdir, indir, outdir, seedfile; n_parallel=1, keep=false)

Use PP3 for singlepoint calculation.
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
                @warn "Failed to calculate energy for $(file)!"
                continue
            end
            throw(error)
        end
        if !keep
            for suffix in [".cell", ".conv", "-out.cell", ".pp", ".res"]
                fname = swapext(working_path, suffix)
                isfile(fname) && rm(fname)
            end
        end
    end
end

"""
    run_acrud(workdir, indir, outdir, seedfile; n_parallel=1, keep=false)

Use acruid for singlepoint calculation - launch many calculations in parallel.
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
    # Copy files to the hopper directory
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
    # Copy the results to the output directory
    verbose && @info "Copying results to $outdir"
    for file in glob_allow_abs(joinpath(workdir, "good_castep", "*.res"))
        verbose && @info "$file to "
        dst = joinpath(outdir, splitdir(file)[end])
        mv(file, dst, force=true)
    end
    # If not keeping the files, remove the directory in the relavent folders
    if !keep
        for folder in ["hopper", "good_castep", "bad_castep"]
            run(`rm -r $workdir/$folder`)
        end
    end
end




"""
    run_pp3(file, seedfile, outpath)

Use `pp3` for single point calculations.
"""
function run_pp3(file, seedfile, outname=nothing)
    if endswith(file, ".res")
        cell = CellBase.read_res(file)
        # Write as cell file
        CellBase.write_cell(swapext(file, ".cell"), cell)
    else
        cell = CellBase.read_cell(file)
    end
    # Copy the seed file
    cp(swapext(seedfile, ".pp"), swapext(file, ".pp"), force=true)
    # Run pp3 relax
    aname = relpath(splitext(file)[1])

    # pp3 has size limit for the seed name....
    if length(aname) > 70
        tempd = mktempdir()
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

    # Write res
    cell.metadata[:enthalpy] = enthalpy
    cell.metadata[:pressure] = pressure
    cell.metadata[:label] = stem(file)
    if !isnothing(outname)
        CellBase.write_res(outname, cell)
    end
    cell
end

"""
    _call_pp3(aname)

Call pp3 to calculate the energy/pressure of a seed file.
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
    parse_disp_output(json_string)

Parse the output of `disp db summary`
"""
function parse_disp_output(json_string)
    data = Dict{String,Int}()
    if contains(json_string, "No data")
        return data
    end
    tmp = JSON.parse(json_string)
    for (x, y) in tmp
        # How to unpack this way due to nested multi-index
        if contains(x, "RES")
            data["RES"] = first(values(first(values(y))))
        elseif contains(x, "ALL")
            data["ALL"] = first(values(first(values(y))))
        elseif contains(x, "COMPLETED")
            data["COMPLETED"] = first(values(first(values(y))))
        end
    end
    data
end


"""
Summarise the status of the project
"""
function summarise(builder::Builder)
    opts = builder.state
    subpath(x) = joinpath(opts.workdir, x)
    println("Workding directory: $(opts.workdir)")
    println("Seed file: $(opts.seedfile)")

    function print_res_count(path)
        nfiles = length(glob_allow_abs(joinpath(path, "*.res")))
        println("  $(path): $(nfiles) structures")
        nfiles
    end

    total_dft = 0
    for i = 0:opts.max_iterations
        println("iteration $i")

        path = subpath("gen$i")
        print_res_count(path)

        path = subpath("gen$(i)-dft")
        total_dft += print_res_count(path)
    end
    println("Total training structures: $(total_dft)")
end


raw"""
    walk_forward_tests(builder::Builder)

Perform walk forward tests - test if the data of generation ``N`` can be predicted by the model 
from generation ``N-1``, and compare it with the results using model from generation ``N`` itself. 
"""
function walk_forward_tests(
    bu::Builder;
    print_results=false,
    iters=0:bu.state.iteration-1,
    fc_show_progress=false,
    check_training_data=false,
)
    trs = []
    for iter in iters
        if check_training_data
            is_training_data_ready(bu, iter + 1) || continue
        else
            if nstructures_calculated(bu, iter + 1) < 1
                continue
            end
        end
        has_ensemble(bu, iter) || break
        @info "Loading features of generation $(iter+1) to test for generation $(iter)..."
        fc = load_features(bu, iter + 1; show_progress=fc_show_progress)
        ensemble = load_ensemble(bu, iter)
        tr = EDDPotentials.TrainingResults(ensemble, fc)
        push!(trs, tr)
        if print_results
            println(
                "Trained model using iteration 0-$iter applied for iteration $(iter+1):",
            )
            println(tr)
        end
    end
    trs
end

function _latest_ensemble_iteration(bu::Builder)
    gen = -1
    for i = 0:bu.state.max_iterations
        if has_ensemble(bu, i)
            gen = i
        end
    end
    if gen < 0
        throw(ErrorException("No valid ensemble found!"))
    end
    gen
end

function has_ensemble(bu::Builder, iteration=bu.state.iteration)
    isfile(joinpath(bu.state.workdir, "$(bu.trainer.prefix)ensemble-gen$(iteration).jld2"))
end

function load_ensemble(bu::Builder, iteration=_latest_ensemble_iteration(bu))
    EDDPotentials.load_from_jld2(
        ensemble_name(bu, iteration),
        EDDPotentials.EnsembleNNInterface,
    )
end

ensemble_name(bu::Builder, iteration=bu.state.iteration) =
    joinpath(bu.state.workdir, "$(bu.trainer.prefix)ensemble-gen$(iteration).jld2")

function is_training_data_ready(bu::Builder, iteration=bu.state.iteration)
    isdir(joinpath(bu.state.workdir, "gen$(iteration)-dft")) || return false
    ndft = nstructures_calculated(bu, iteration)
    if iteration == 0
        nexpected = bu.state.n_initial
    else
        nexpected = bu.state.per_generation * (bu.state.shake_per_minima + 1)
    end
    if ndft / nexpected > bu.state.per_generation_threshold
        return true
    end
    return false
end

nstructures(bu::Builder, iteration) =
    length(glob_allow_abs(joinpath(bu.state.workdir, "gen$(iteration)/*.res")))
nstructures_calculated(bu::Builder, iteration) =
    length(glob_allow_abs(joinpath(bu.state.workdir, "gen$(iteration)-dft/*.res")))

function load_structures(bu::Builder, iteration::Vararg{Int})
    dirs = [joinpath(bu.state.workdir, "gen$(iter)-dft/*.res") for iter in iteration]
    sc = EDDPotentials.StructureContainer(dirs, threshold=bu.trainer.energy_threshold)
    return sc
end
load_structures(bu::Builder) = load_structures(bu, 0:bu.state.iteration)
load_structures(bu::Builder, iteration) = load_structures(bu, iteration...)

"""
    load_features(bu::Builder, iteration...)

Loading features for specific iterations.   
"""
function load_features(bu::Builder, iteration::Vararg{Int}; show_progress=true, kwargs...)
    sc = load_structures(bu, iteration...;)
    elemental_energies = _make_symbol_keys(bu.state.elemental_energies)
    return EDDPotentials.FeatureContainer(
        sc,
        bu.cf;
        nmax=bu.trainer.nmax,
        show_progress,
        elemental_energies,
        kwargs...,
    )
end

load_features(bu::Builder; kwargs...) = load_features(bu, 0:bu.state.iteration; kwargs...)
load_features(bu::Builder, iteration; kwargs...) =
    load_features(bu, iteration...; kwargs...)

"""
    run_rss(builder::Builder)

Run random structures search using trained ensembel model. The output files are in the 
`search` subfolder by default.
"""
function run_rss(builder::Builder)
    rs = builder.rss
    ensemble = load_ensemble(builder, rs.ensemble_id)
    searchdir = joinpath(builder.state.workdir, rs.subfolder_name)

    # Change the output name
    if abs(rs.pressure_gpa) > 0.01
        searchdir = searchdir * "-$(rs.pressure_gpa)gpa"
    end

    ensure_dir(searchdir)
    (; seedfile, seedfile_weights) = rs
    _run_rss(
        joinpath.(Ref(builder.state.workdir), seedfile),
        ensemble,
        builder.cf;
        show_progress=rs.show_progress,
        max=rs.max,
        outdir=searchdir,
        ensemble_std_max=rs.ensemble_std_max,
        ensemble_std_min=rs.ensemble_std_min,
        packed=rs.packed,
        niggli_reduce_output=rs.niggli_reduce_output,
        max_err=rs.max_err,
        pressure_gpa=rs.pressure_gpa,
        seedfile_weights,
        relax_option=rs.relax,
        elemental_energies=_make_symbol_keys(builder.state.elemental_energies),
    )
end

"""
    run_rss(str::AbstractString)

Run random structure searching for a configuration file for the builder.
"""
function run_rss(str::AbstractString="link.toml")
    builder = Builder(str)
    run_rss(builder)
end

# Map for trainer names
TRAINER_NAME = Dict(TrainingOption => "locallm")


for func in [:get_energy, :get_forces, :get_pressure, :get_energy_std, :get_enthalpy]
    @eval begin
        function $func(
            cell::Cell,
            builder::Builder,
            gen::Int=_latest_ensemble_iteration(builder);
            kwargs...,
        )
            ensemble = load_ensemble(builder, gen)
            $func(VariableCellCalc(NNCalc(cell, builder.cf, ensemble)); kwargs...)
        end
        @doc """
            $($func)(cell::Cell, builder::Builder, gen::Int=_latest_ensemble_iteration(builder);kwargs...)

        Convenient method for calling $(EDDPotentials.$func) using a Builder object.
        """ $func(cell::Cell, builder::Builder, gen::Int)
    end
end

"""
    _run_rss_link()

Run the random search step as part of the `link!` iterative building protocol.
This function is intented to be called as a separated Julia process.


!!! note 

    This function is meanted to be called with commandline arguments.
"""
function _run_rss_link()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--iteration"
        help = "Iteration number the random search step is for"
        required = true
        arg_type = Int
        "--file"
        help = "Path to the configuration file"
        arg_type = String
        default = "link.toml"
        "--num"
        help = "Number of structures to generate"
        required = true
        arg_type = Int
        "--outdir"
        arg_type = String
        required = true
        help = "Output directory where the structures should be placed"
        "--pressure"
        help = "Pressure under which the structure will be relaxed (GPa)."
        "--pressure-range"
        help = "Range of the pressure"
    end

    args = parse_args(s)
    fname = args["file"]
    bu = Builder(fname)
    bu.state.iteration = args["iteration"]
    # Load the ensemble file of the "last" iteration
    ensemble = load_ensemble(bu, bu.state.iteration - 1)

    if args["pressure"] !== nothing
        pressure_gpa = parse(Float64, args["pressure"])
    else
        pressure_gpa = 0.001
    end

    if args["pressure-range"] !== nothing
        pressure_gpa_range = map(x -> parse(Float64, x), split(args["pressure-range"], ","))
    else
        pressure_gpa_range = nothing
    end
    (; seedfile, seedfile_weights) = bu.state
    _run_rss(
        joinpath.(bu.state.workdir, seedfile),
        ensemble,
        bu.cf;
        core_size=bu.state.core_size,
        ensemble_std_max=bu.state.ensemble_std_max,
        ensemble_std_min=bu.state.ensemble_std_min,
        max=args["num"],
        outdir=args["outdir"],
        pressure_gpa,
        pressure_gpa_range,
        seedfile_weights,
        niggli_reduce_output=bu.state.rss_niggli_reduce,
        relax_option=bu.state.relax,
        elemental_energies=_make_symbol_keys(bu.state.elemental_energies),
    )
end


"""
    _make_symbol_keys(dict::Dict)

Swap the String keys of a dictonary into Symbol.
"""
function _make_symbol_keys(dict::Dict{T,K}) where{T, K}
    Dict{Symbol,K}(
        Symbol(key) => _make_symbol_keys(value) for (key, value) in pairs(dict)
    )
end

_make_symbol_keys(x) = x

function _make_string_keys(dict::Dict)
    Dict{String,Any}(
        string(key) => _make_string_keys(value) for (key, value) in pairs(dict)
    )
end

_make_string_keys(x) = x
