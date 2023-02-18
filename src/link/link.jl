#=
Code for iteratively building the model
=#
import Base
using Parameters
using JSON
using YAML
using ArgParse

const XT_NAME = "xt"
const YT_NAME = "yt"
const FEATURESPEC_NAME = "cf"


@with_kw mutable struct BuilderState
    iteration::Int = 0
    workdir::String = "."
    seedfile::String
    seedfile_calc::String = seedfile
    max_iterations::Int = 5
    per_generation::Int = 100
    per_generation_threshold::Float64 = 0.98
    shake_per_minima::Int = 10
    build_timeout::Float64 = 1.0
    shake_amp::Float64 = 0.02
    shake_cell_amp::Float64 = 0.02
    n_parallel::Int = 1
    mpinp::Int = 2
    n_initial::Int = 1000
    dft_mode::String = "castep"
    dft_kwargs::Dict{Symbol,Any} = Dict{Symbol,Any}()
    rss_pressure_gpa::Float64 = 0.1
    rss_pressure_gpa_range::Vector{Float64} = Float64[]
    rss_niggli_reduce::Bool = true
    core_size::Float64 = 1.0
    ensemble_std_min::Float64 = 0.0
    ensemble_std_max::Float64 = -1.0
    "Run walk-forward test before re-training"
    run_walk_forward::Bool = true
    "Override the project_prefix"
    project_prefix_override::String = ""
    builder_file_path::String = ""
end

abstract type AbstractTrainer end

@with_kw mutable struct LocalLMTrainer <: AbstractTrainer
    energy_threshold::Float64 = 10.0
    nmax::Int = 3000
    nmodels::Int = 256
    user_test_for_ensemble::Bool = true
    max_iter::Int = 300
    "number of hidden nodes in each layer"
    n_nodes::Vector{Int} = [8]
    earlystop::Int = 30
    show_progress::Bool = true
    "Store the data used for training in the archive"
    store_training_data::Bool = true
    rmse_threshold::Float64 = 0.5
    training_mode::String = "manual_backprop"
    training_kwargs::Dict{Symbol,Any} = Dict{Symbol,Any}()
    train_split::Vector{Float64} = [0.8, 0.1, 0.1]
    use_test_for_ensemble::Bool = true
    save_each_model::Bool = true
    p::Float64 = 1.25
    keep_best::Bool = true
    tb_logger_dir::Any = nothing
    log_file::Any = nothing
    prefix::String = ""
    max_train::Int = 999
    "Number of workers to be launcher in parallel"
    num_workers::Int = 1
end

@with_kw mutable struct RssSetting
    packed::Bool = true
    seedfile::String = "null"
    ensemble_id::Int = -1
    max::Int = 1000
    subfolder_name::String = "search"
    show_progress::Bool = true
    ensemble_std_max::Float64 = 0.2
    ensemble_std_min::Float64 = -1.0
    eng_threshold::Float64 = -1.0
    niggli_reduce_output::Bool = true
    max_err::Int = 10
    pressure_gpa::Float64 = 0.01
end


struct Builder{M<:AbstractTrainer}
    state::BuilderState
    cf::CellFeature
    trainer::M
    cf_embedding::Any
    rss::RssSetting
end

function Builder(
    state::BuilderState,
    cf::CellFeature,
    trainer;
    cf_embedding=nothing,
    rss=RssSetting(),
)
    builder = Builder{typeof(trainer)}(state, cf, trainer, cf_embedding, rss)
    _set_iteration!(builder)
    if rss.seedfile == "null"
        rss.ensemble_id = builder.state.iteration
        rss.seedfile = splitext(builder.state.seedfile)[1]
        @warn "Using default ensemble id: $(rss.ensemble_id)"
        @warn "Using seed file: $(rss.seedfile)"
    end
    if rss.ensemble_id < 0
        rss.ensemble_id = builder.state.iteration
        @warn "Using default ensemble id: $(rss.ensemble_id)"
    end

    builder_uuid(builder)
    builder
end

"""
    _todict(builder)

Construct a dictionary for a builder - used for serialization through YAML
"""
function _todict(state::T) where {T<:Union{BuilderState,LocalLMTrainer,RssSetting}}
    out = Dict{Symbol,Any}()
    for name in fieldnames(T)
        val = getproperty(state, name)
        # Convert NamedTuple into dictionary
        if isa(val, NamedTuple)
            out[name] = Dict(pairs(val)...)
        else
            out[name] = val
        end
    end
    out
end


function _todict(cf::CellFeature)
    out = Dict{Symbol,Any}()
    out[:elements] = map(string, cf.elements)
    if cf.constructor_kwargs == nothing
        throw(
            ErrorException(
                "CellFeature with out recorded constructed keyword arguments cannot be converted!",
            ),
        )
    end
    for (key, value) in pairs(cf.constructor_kwargs)
        out[key] = value
    end
    return out
end

"""
    _fromdict(T::DataType, dict::Dict)

Load from a diction of field names.
"""
function _fromdict(
    T::Type{N},
    dict::Dict,
) where {N<:Union{BuilderState,LocalLMTrainer,RssSetting}}
    new_dict = deepcopy(dict)
    # Manual type conversion
    for key in keys(dict)
        # Reconstruct NamedTuple from dictionary
        if fieldtype(T, key) <: NamedTuple
            new_dict[key] = NamedTuple(dict[key])
        end
    end
    T(; dict...)
end

function _fromdict(::Type{<:CellFeature}, dict)
    _dict = deepcopy(dict)
    elements = pop!(_dict, :elements)
    CellFeature(elements; _dict...)
end

function _todict(builder::Builder)
    outdict = Dict{Symbol,Any}()
    outdict[:rss] = _todict(builder.rss)
    outdict[:trainer] = _todict(builder.trainer)
    outdict[:cf] = _todict(builder.cf)
    outdict[:trainer][:type] = TRAINER_NAME[typeof(builder.trainer)]

    outdict[:state] = _todict(builder.state)
    if builder.cf_embedding !== nothing
        outdict[:cf_embedding] =
            Dict{Symbol,Any}(:n => builder.cf_embedding.n, :m => builder.cf_embedding.m)
    end
    outdict
end

function _fromdict(::Type{Builder}, dict)

    # Adjust the workdir to be that relative to the yaml file
    statedict = dict[:state]

    state = _fromdict(BuilderState, statedict)

    # Setup cell Feature
    cf_dict = dict[:cf]
    cf = _fromdict(CellFeature, cf_dict)

    # Setup trainer
    trainer_dict = deepcopy(get(dict, :trainer, Dict{Symbol,Any}()))
    trainer_name = pop!(trainer_dict, :type)
    if trainer_name == "locallm"
        trainer = _fromdict(LocalLMTrainer, trainer_dict)
    else
        throw(ErrorException("trainer type $(trainer) is not known"))
    end

    # Setup embedding
    if :cf_embedding in keys(dict)
        n = dict[:cf_embedding][:n]
        m = get(dict[:cf_embedding], :m, n)
        embedding = CellEmbedding(cf, n, m)
    else
        embedding = nothing
    end

    if :rss in keys(dict)
        rss = _fromdict(RssSetting, dict[:rss])
        Builder(state, cf, trainer; cf_embedding=embedding, rss=rss)
    else
        Builder(state, cf, trainer; cf_embedding=embedding)
    end
end

"""
    save_builder(fname::AbstractString, builder)

Save a `Builder` to a file.
"""
function save_builder(fname::AbstractString, builder::Builder)
    YAML.write_file(fname, _todict(builder))
end


"""
    Builder(str::AbstractString="link.yaml")

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
function Builder(str::AbstractString="link.yaml")
    @info "Loading from file $(str)"

    loaded = YAML.load_file(str; dicttype=Dict{Symbol,Any})
    builder = _fromdict(Builder, loaded)

    # Adjust the workdir to be that relative to the yaml file
    paths = splitpath(str)
    statedict = loaded[:state]
    if length(paths) > 1
        parents = paths[1:end-1]
        @assert !startswith(statedict[:workdir], "/") ":workdir should be a relative path"
        builder.state.workdir = joinpath(parents..., statedict[:workdir])
        @info "Setting workdir to $(statedict[:workdir])"
    end

    # Store the path to the builder file
    builder.state.builder_file_path = str

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
    println("\nCellFeature: ")
    show(io, m, bu.cf)
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
julia -e "Using EDDP;EDDP.link()" -- --file "link.yaml" 
```
"""
function link()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--file"
        help = "Name of the yaml file"
        default = "link.yaml"
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
    if iter == 0
        # First cycle generate from the seed without relaxation
        # Sanity check - are we definitely overfitting?
        if nfeatures(bu.cf) > bu.state.n_initial
            @warn "The number of features $(nfeature(bu.cf)) is larger than the initial training size!"
        end
        # Generate random structures
        nstruct = bu.state.n_initial - ndata
        if nstruct > 0
            @info "Genearating $(nstruct) initial training structures."
            build_random_structures(bu.state.seedfile, outdir; n=nstruct)
        end
    else
        # Subsequent cycles - generate from the seed and perform relaxation
        # Read ensemble file
        ensemble = load_ensemble(bu, iter - 1)
        nstruct = bu.state.per_generation - ndata
        if nstruct > 0
            @info "Generating $(nstruct) training structures for iteration $iter."
            # Generate data sets
            if length(bu.state.rss_pressure_gpa_range) > 0
                a, b = bu.state.rss_pressure_gpa_range
                pressure = rand() * (b - a) + a
            else
                pressure = bu.state.rss_pressure_gpa
            end
            _run_rss(
                bu.state.seedfile,
                ensemble,
                bu.cf;
                core_size=bu.state.core_size,
                ensemble_std_max=bu.state.ensemble_std_max,
                ensemble_std_min=bu.state.ensemble_std_min,
                max=nstruct,
                outdir=outdir,
                pressure_gpa=pressure,
                niggli_reduce_output=bu.state.rss_niggli_reduce,
            )
            # Shake the generate structures
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


"Directory for input structures"
_input_structure_dir(bu::Builder) = joinpath(bu.state.workdir, "gen$(bu.state.iteration)")
"Directory for output structures after external calculations"
_output_structure_dir(bu::Builder) =
    joinpath(bu.state.workdir, "gen$(bu.state.iteration)-dft")


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
            bu.state.seedfile_calc;
            project_prefix,
            threshold=bu.state.per_generation_threshold,
            bu.state.dft_kwargs...,
        )
        return true
    elseif bu.state.dft_mode == "pp3"
        run_pp3_many(
            joinpath(bu.state.workdir, ".pp3_work"),
            _input_structure_dir(bu),
            _output_structure_dir(bu),
            bu.state.seedfile_calc;
            n_parallel=bu.state.n_parallel,
            bu.state.dft_kwargs...,
        )
        return true
    elseif bu.state.dft_mode == "castep"
        run_crud(
            bu.state.workdir,
            _input_structure_dir(bu),
            _output_structure_dir(bu);
            mpinp=bu.state.mpinp,
            bu.state.n_parallel,
            bu.state.dft_kwargs...,
        )
        return true
    end
    return false
end

"""
Carry out training and save the ensemble as a JLD2 archive.
"""
function _perform_training(bu::Builder{M}) where {M<:LocalLMTrainer}

    tra = bu.trainer
    # Write the dataset to the disk
    @info "Training with LocalLMTrainer"
    @info "Preparing dataset..."
    write_dataset(bu)

    # Call multiple sub processes
    project_path = dirname(Base.active_project())
    builder_file = bu.state.builder_file_path
    @assert builder_file != ""
    cmd = Cmd([
        Base.julia_cmd()...,
        "--project=$(project_path)",
        "-e",
        "using EDDP;EDDP.run_trainer()",
        "$(builder_file)",
        "--iteration",
        "$(bu.state.iteration)",
    ])

    # Call multiple trainer processes
    @info "Subprocess launch command: $cmd"
    tasks = Task[]
    for i = 1:tra.num_workers
        # Run with ids
        _cmd = Cmd([cmd..., "--id", "$i"])
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

    nm = num_existing_models(bu)
    @info "Number of trained models: $nm"

    # Create ensemble
    nm = num_existing_models(bu)
    if nm >= tra.nmodels * 0.9
        ensemble = create_ensemble(bu; save_and_clean=true)
    else
        throw(
            ErrorException(
                "Only $nm models are found in the training directory, need $(tra.nmodels)",
            ),
        )
    end

    return ensemble
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
NOTE: this requires `disp` to be avaliable in the commandline.
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
    while true
        ncomp, nall = _disp_get_completed_jobs(project_name)
        if ncomp / nall > threshold
            @info " $(ncomp)/$(nall) calculation completed - moving on ..."
            break
        else
            @info "Completed calculations: $(ncomp)/$(nall) - waiting ..."
        end
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

Use PP3 for singlepoint calculation - launch many calculations in parallel.
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
                rm(swapext(working_path, suffix))
            end
        end
    end
end


"""
    run_pp3(file, seedfile, outpath)

Use `pp3` for single point calculations.
"""
function run_pp3(file, seedfile, outpath)
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
    # Read enthalpy
    enthalpy = 0.0
    pressure = 0.0
    for line in eachline(pipeline(`pp3 -n $(splitext(file)[1])`))
        if contains(line, "Enthalpy")
            enthalpy = parse(Float64, split(line)[end])
        end
        if contains(line, "Pressure")
            pressure = parse(Float64, split(line)[end])
        end
    end
    # Write res
    cell.metadata[:enthalpy] = enthalpy
    cell.metadata[:pressure] = pressure
    cell.metadata[:label] = stem(file)
    CellBase.write_res(outpath, cell)
    cell
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
        tr = EDDP.TrainingResults(ensemble, fc)
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
    EDDP.load_from_jld2(ensemble_name(bu, iteration), EDDP.EnsembleNNInterface)
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
    sc = EDDP.StructureContainer(dirs, threshold=bu.trainer.energy_threshold)
    return sc
end
load_structures(bu::Builder) = load_structures(bu, 0:bu.state.iteration)
load_structures(bu::Builder, iteration) = load_structures(bu, iteration...)

"""
    load_features(bu::Builder, iteration...)

Loading features for specific iterations.   
"""
function load_features(bu::Builder, iteration::Vararg{Int}; show_progress=true)
    sc = load_structures(bu, iteration...;)
    return EDDP.FeatureContainer(sc, bu.cf; nmax=bu.trainer.nmax, show_progress)
end

load_features(bu::Builder; kwargs...) = load_features(bu, 0:bu.state.iteration; kwargs...)
load_features(bu::Builder, iteration; kwargs...) =
    load_features(bu, iteration...; kwargs...)

"""
    run_rss(builder::Builder;kwargs...)

Run random structures search using trained ensembel model. The output files are in the 
`search` subfolder by default.
"""
function run_rss(builder::Builder; kwargs...)
    rs = builder.rss
    ensemble = load_ensemble(builder, rs.ensemble_id)
    searchdir = joinpath(builder.state.workdir, rs.subfolder_name)
    ensure_dir(searchdir)
    _run_rss(
        rs.seedfile,
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
        kwargs...,
    )
end

"""
    run_rss(str::AbstractString)

Run random structure searching for a configuration file for the builder.
"""
function run_rss(str::AbstractString="link.yaml"; kwargs...)
    builder = Builder(str)
    run_rss(builder; kwargs...)
end

# Map for trainer names
TRAINER_NAME = Dict(LocalLMTrainer => "locallm")


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

        Convenient method for calling $(EDDP.$func) using a Builder object.
        """ $func(cell::Cell, builder::Builder, gen::Int)
    end
end
