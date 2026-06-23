#=
Builder and workflow orchestration for iterative potential construction.

This module contains:
- Builder struct and configuration management
- Workflow orchestration (link!, step!)
- RSS (Random Structure Search) generation
- Training orchestration
=#

using Parameters
using JSON
using YAML
using TOML
using ArgParse
using Random
using ..Potential: CellEmbedding, EnsembleNNInterface, load_model_artifact

# Note: Utils functions will be available from parent module after include
# using ensure_dir, input_structure_dir, output_structure_dir
# using builder_uuid, builder_short_uuid, make_symbol_keys


"""
    Builder

The Builder is the main object that stores states and options for training and using the potentials.
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


"""
    Builder(options::BuilderOption)

Create a new `Builder` object with the given options.
"""
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


"""
    builder_uuid(bu::Builder)

Convenience wrapper to get UUID for a Builder object.
"""
builder_uuid(bu::Builder) = builder_uuid(bu.state.workdir)

"""
    builder_short_uuid(bu::Builder)

Convenience wrapper to get short UUID for a Builder object.
"""
builder_short_uuid(bu::Builder) = builder_short_uuid(bu.state.workdir)

function diagnose_formula_sampling(
    builder::Builder;
    seedfile=nothing,
    nsamples::Int=3,
    io::IO=stdout,
)
    diagnose_formula_sampling(
        builder.state,
        builder.cfopt.elements;
        seedfile,
        nsamples,
        io,
    )
end


"""
    Builder(str::AbstractString="link.toml")

Load the builder from a TOML/YAML file. The file contains nested key-values pairs
similar to the constructors of the types.

```yaml
state:
    seedfile : "test.cell"

trainer:
    train_method : "lm"
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
        # Save converted TOML if requested
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
    builder.state.builder_file_path = abspath(str)

    # Update the iteration number with the new path
    _set_iteration!(builder)

    builder
end


"""
    save_builder(fname::AbstractString, builder)

Save a `Builder` to a file.
"""
function save_builder(fname::AbstractString, builder::Builder)
    open(fname, "w") do f
        to_toml(f, builder.options)
    end
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

Fast-forward to the correct iteration by checking existing ensemble files.
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
    has_ensemble(bu::Builder, iteration)

Check if an ensemble file exists for the given iteration.
"""
function has_ensemble(bu::Builder, iteration=bu.state.iteration)
    isfile(_ensemble_existing_path(bu, iteration))
end


"""
    load_ensemble(bu::Builder, iteration)

Load the ensemble model for a given iteration.
"""
function load_ensemble(bu::Builder, iteration=_latest_ensemble_iteration(bu))
    artifact = load_model_artifact(_ensemble_existing_path(bu, iteration))
    artifact.model isa EnsembleNNInterface ||
        throw(ArgumentError("Artifact `$(_ensemble_existing_path(bu, iteration))` is not an ensemble model."))
    artifact.model
end


"""
    ensemble_name(bu::Builder, iteration)

Get the file path for the ensemble at a given iteration.
"""
ensemble_name(bu::Builder, iteration=bu.state.iteration) =
    joinpath(bu.state.workdir, "$(bu.trainer.prefix)ensemble-gen$(iteration).json")

function _ensemble_existing_path(bu::Builder, iteration=bu.state.iteration)
    json_path = ensemble_name(bu, iteration)
    isfile(json_path) && return json_path
    legacy_path = joinpath(bu.state.workdir, "$(bu.trainer.prefix)ensemble-gen$(iteration).jld2")
    isfile(legacy_path) && return legacy_path
    json_path
end


"""
    _latest_ensemble_iteration(bu::Builder)

Find the most recent iteration with a trained ensemble.
"""
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


"""
    link!(builder::Builder)

Run automated iterative building cycles.
"""
abstract type AbstractLinkBackend end
struct LocalLinkBackend <: AbstractLinkBackend end
struct DistributedLinkBackend <: AbstractLinkBackend end

struct LinkPhaseResult
    status::Symbol
    phase::Symbol
    iteration::Int
    message::String
    metrics::Dict{Symbol,Any}
end

LinkPhaseResult(status::Symbol, phase::Symbol, iteration::Int; message="", metrics=Dict{Symbol,Any}()) =
    LinkPhaseResult(status, phase, iteration, String(message), Dict{Symbol,Any}(metrics))

auto_link_backend(bu::Builder) =
    bu.options.distributed.enabled ? DistributedLinkBackend() : LocalLinkBackend()

function link!(builder::Builder)
    run_link_workflow!(builder, auto_link_backend(builder))
end

function run_link_workflow!(builder::Builder, backend::AbstractLinkBackend=auto_link_backend(builder))
    validate_builder_for_link!(builder, backend)
    state = builder.state

    while state.iteration <= state.max_iterations
        result = run_link_iteration!(builder, backend, state.iteration)
        if result.status == :stopped
            @warn "Aborted training loop at iteration: $(state.iteration)."
            shutdown_link_backend!(backend, builder)
            return builder
        end
        if result.status == :pending
            @info "Paused training loop at iteration $(state.iteration); phase $(result.phase) is pending."
            return builder
        end
    end
    shutdown_link_backend!(backend, builder)
    return builder
end

shutdown_link_backend!(::AbstractLinkBackend, ::Builder) = nothing

function run_link_iteration!(bu::Builder, backend::AbstractLinkBackend, iter::Int)
    should_stop(bu) && return LinkPhaseResult(:stopped, :preflight, iter)

    if has_ensemble(bu, iter)
        bu.state.iteration += 1
        return LinkPhaseResult(:ok, :advance, iter; message="ensemble already exists")
    end

    if !is_generation_data_ready(bu, iter)
        result = _generate_random_structures(bu, iter)
        result == :stopped && return LinkPhaseResult(:stopped, :generate, iter)
        should_stop(bu) && return LinkPhaseResult(:stopped, :generate, iter)
    end
    validate_generated_structures!(bu, iter)

    if !is_training_data_ready(bu, iter)
        @info "Starting energy calculations for iteration $iter."
        result = run_dft_phase!(backend, bu, iter)
        result.status == :stopped && return result
        result.status == :ok ||
            throw(ErrorException("DFT phase failed for iteration $iter: $(result.message)"))
    end
    validate_dft_outputs!(bu, iter)

    ns = nstructures(bu, iter)
    @info "Number of new structures in iteration $(iter): $ns"

    if bu.state.run_walk_forward && iter > 0
        walk_forward_tests(bu; print_results=true, iters=[iter - 1])
        should_stop(bu) && return LinkPhaseResult(:stopped, :walk_forward, iter)
    end

    if !is_training_dataset_ready(bu, iter)
        @info "Preparing dataset..."
        write_dataset(bu)
    else
        @info "Skipping dataset preparation for iteration $iter; dataset artifact already exists."
    end
    should_stop(bu) && return LinkPhaseResult(:stopped, :prepare_dataset, iter)

    if !_models_ready_for_ensemble(bu)
        @info "Starting training for iteration $iter."
        result = run_training_phase!(backend, bu, iter)
        result.status == :stopped && return result
        result.status == :pending && return result
        result.status == :ok ||
            throw(ErrorException("Training phase failed for iteration $iter: $(result.message)"))
    else
        @info "Skipping training for iteration $iter; enough model artifacts already exist."
    end

    create_iteration_ensemble!(bu, iter)
    bu.state.iteration += 1
    return LinkPhaseResult(:ok, :advance, iter)
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
    return link!(builder)
end


"""
    _check_dependencies()

Check if external programs are available in the system.
"""
function _check_dependencies(names)
    unique_names = unique(String.(names))
    isempty(unique_names) && return true
    @info "Checking required external programs: $unique_names"
    missing = [
        name for name in unique_names if
        !(isfile(name) || Sys.which(name) !== nothing)
    ]
    isempty(missing) || throw(
        ErrorException(
            "Missing required external program(s): $(join(missing, ", ")). Please check the runtime environment.",
        ),
    )
    return true
end

_scheduler_command(scheduler) = scheduler.type == "SLURM" ? "sbatch" :
                                scheduler.type == "SGE" ? "qsub" : ""

function _dft_required_commands(bu::Builder)
    mode = lowercase(strip(bu.state.dft_mode))
    if mode == "pp3"
        return ["pp3"]
    elseif mode in ("crud", "crud-queue")
        return ["crud.pl"]
    elseif mode == "acrud"
        ap_cmd = get(bu.state.dft_kwargs, "ap_cmd", get(bu.state.dft_kwargs, :ap_cmd, nothing))
        if ap_cmd !== nothing && !isempty(strip(String(ap_cmd)))
            return [String(ap_cmd)]
        end
        local_ap = joinpath(dirname(Base.active_project()), "airsspy-repo", ".venv", "bin", "ap")
        isfile(local_ap) && return [local_ap]
        return ["ap"]
    elseif mode == "disp-castep"
        return ["disp"]
    elseif mode == "mock"
        return String[]
    end
    return String[]
end

function _required_commands(::LocalLinkBackend, bu::Builder)
    commands = String["buildcell"]
    append!(commands, _dft_required_commands(bu))
    if bu.trainer.external && lowercase(strip(bu.trainer.external_mode)) in ("submit", "command")
        scheduler = bu.options.scheduler
        scheduler === nothing && throw(
            ArgumentError("A top-level `[scheduler]` config is required when trainer.external_mode = \"submit\"."),
        )
        scheduler.submit_jobs && push!(commands, _scheduler_command(scheduler))
    end
    filter!(!isempty, commands)
    return commands
end

function _required_commands(::DistributedLinkBackend, bu::Builder)
    commands = String["buildcell"]
    append!(commands, _dft_required_commands(bu))
    scheduler = bu.options.scheduler
    if scheduler !== nothing && bu.options.distributed.manage_worker_jobs && scheduler.submit_jobs
        push!(commands, _scheduler_command(scheduler))
    end
    filter!(!isempty, commands)
    return commands
end

function _require_builder_file_path!(bu::Builder)
    path = strip(bu.state.builder_file_path)
    isempty(path) && throw(
        ArgumentError(
            "builder.state.builder_file_path must be set to an existing configuration file before launching subprocesses or distributed link workers. Build from `Builder(\"link.toml\")` or save the BuilderOption first.",
        ),
    )
    abspath_path = abspath(path)
    isfile(abspath_path) || throw(
        ArgumentError("builder.state.builder_file_path does not exist: `$abspath_path`."),
    )
    bu.state.builder_file_path = abspath_path
    return abspath_path
end

function _uses_subprocesses(bu::Builder, ::LocalLinkBackend)
    return bu.state.rss_external ||
           bu.trainer.external ||
           lowercase(strip(bu.trainer.external_mode)) in ("submit", "command")
end

_uses_subprocesses(bu::Builder, ::DistributedLinkBackend) = true

function validate_builder_for_link!(bu::Builder, backend::AbstractLinkBackend)
    bu.state.rss_nprocs >= 1 ||
        throw(ArgumentError("state.rss_nprocs must be >= 1."))
    0.0 < bu.state.per_generation_threshold <= 1.0 ||
        throw(ArgumentError("state.per_generation_threshold must be in the interval (0, 1]."))
    if backend isa DistributedLinkBackend
        _distributed_log_level(bu.options.distributed.log_level)
        bu.options.distributed.log_interval_seconds > 0.0 ||
            throw(ArgumentError("options.distributed.log_interval_seconds must be > 0."))
    end
    _check_dependencies(_required_commands(backend, bu))
    _uses_subprocesses(bu, backend) && _require_builder_file_path!(bu)
    return true
end

function run_dft_phase!(::LocalLinkBackend, bu::Builder, iter::Int)
    is_training_data_ready(bu, iter) && return LinkPhaseResult(:ok, :dft, iter; message="DFT outputs already ready")
    flag = _run_external(bu)
    should_stop(bu) && return LinkPhaseResult(:stopped, :dft, iter)
    flag || throw(ErrorException("Cannot find external code to run for generating training data"))
    return LinkPhaseResult(:ok, :dft, iter)
end

function run_training_phase!(::LocalLinkBackend, bu::Builder, iter::Int)
    _models_ready_for_ensemble(bu) &&
        return LinkPhaseResult(:ok, :train_models, iter; message="models already ready")
    result = _perform_training_models(bu)
    result == :stopped && return LinkPhaseResult(:stopped, :train_models, iter)
    result == :pending && return LinkPhaseResult(:pending, :train_models, iter)
    return LinkPhaseResult(:ok, :train_models, iter)
end

function create_iteration_ensemble!(bu::Builder, iter::Int=bu.state.iteration)
    nm = num_existing_models(bu)
    required = _required_models_for_ensemble(bu)
    @info "Number of trained models: $nm"
    if nm >= required
        return create_ensemble(
            bu;
            clean_individual_models=bu.trainer.clean_models,
            save_ensemble_model=true,
        )
    end
    throw(
        ErrorException(
            "Only $nm models are found in the training directory, need $required.",
        ),
    )
end

_required_models_for_ensemble(tra::TrainingOption) =
    tra.external ? _external_required_models(tra) : tra.nmodels

_required_models_for_ensemble(bu::Builder) =
    (bu.trainer.external || bu.options.distributed.enabled) ?
    _external_required_models(bu.trainer) :
    bu.trainer.nmodels

_models_ready_for_ensemble(bu::Builder) =
    num_existing_models(bu) >= _required_models_for_ensemble(bu)


"""
    should_stop(bu::Builder)

Check if a STOP file exists to abort the workflow.
"""
function should_stop(bu::Builder)
    if isfile(joinpath(bu.state.workdir, "STOP"))
        @warn "STOP file detected, aborting...."
        return true
    end
    return false
end


"""
    step!(bu::Builder)

Run a single step for the Builder.
"""
function step!(bu::Builder)
    validate_builder_for_link!(bu, auto_link_backend(bu))
    run_link_iteration!(bu, auto_link_backend(bu), bu.state.iteration)
    return bu
end


"""Directory for input structures"""
_input_structure_dir(bu::Builder) = input_structure_dir(bu.state.workdir, bu.state.iteration)

"""Directory for output structures after external calculations"""
_output_structure_dir(bu::Builder) = output_structure_dir(bu.state.workdir, bu.state.iteration)

function _resolve_known_structure_dir(state::BuilderState)
    d = strip(state.known_structure_dir)
    isempty(d) && return nothing
    isabspath(d) ? d : joinpath(state.workdir, d)
end

"""
    _import_known_structures!(state, outdir) -> Vector{String}

Import known structures for generation 0 as `.res` files.
"""
function _import_known_structures!(state::BuilderState, outdir::AbstractString)
    kdir = _resolve_known_structure_dir(state)
    isnothing(kdir) && return String[]
    isdir(kdir) || return String[]

    copied = String[]
    files = vcat(
        glob_allow_abs(joinpath(kdir, "*.res")),
        glob_allow_abs(joinpath(kdir, "*.cell")),
    )
    for f in sort(files)
        outname = joinpath(outdir, "$(stem(f)).res")
        isfile(outname) && continue
        try
            if endswith(f, ".res")
                cp(f, outname; force=false)
            else
                cell = CellBase.read_cell(f)
                cell.metadata[:label] = stem(f)
                write_res(outname, cell)
            end
            push!(copied, outname)
        catch err
            @warn "Failed to import known structure `$f` with error: $err"
        end
    end
    copied
end


"""
    _run_external(bu::Builder)

Run external DFT calculator for generating training data.
"""
function _run_external(bu::Builder)
    ensure_dir(_output_structure_dir(bu))
    calculator = get_external_calculator(bu)
    return calculator(_input_structure_dir(bu), _output_structure_dir(bu))
end


"""
    _generate_random_structures(bu::Builder, iter)

Generate random structures (if needed) as training data for the `Builder`.
"""
function required_generated_structures(bu::Builder, iter::Int=bu.state.iteration)
    iter == 0 ? bu.state.n_initial : bu.state.per_generation
end

function is_generation_data_ready(bu::Builder, iter::Int=bu.state.iteration)
    generated_input_count(bu, iter) >= required_generated_structures(bu, iter)
end

function validate_generated_structures!(bu::Builder, iter::Int=bu.state.iteration)
    required = required_generated_structures(bu, iter)
    actual = generated_input_count(bu, iter)
    actual >= required && return true
    throw(
        ErrorException(
            "Too few generated structures for iteration $iter: found $actual in `$(input_structure_dir(bu.state.workdir, iter))`, need at least $required.",
        ),
    )
end

function _generate_random_structures(bu::Builder, iter)
    # Generate new structures
    outdir = input_structure_dir(bu.state.workdir, iter)
    ensure_dir(outdir)
    ndata = length(glob_allow_abs(joinpath(outdir, "*.res")))
    required = required_generated_structures(bu, iter)
    if ndata >= required
        @info "Skipping structure generation for iteration $iter; found $ndata existing structures."
        return :ok
    end
    (; seedfile, seedfile_weights, workdir, buildcell_nprocs, build_timeout) = bu.state
    formula_ctx = _formula_sampling_context(bu.state, bu.cfopt.elements)
    seed_text_transform = _make_seed_text_transform(formula_ctx)

    if iter == 0
        known_files = _import_known_structures!(bu.state, outdir)
        num_known = 0
        if !isempty(known_files) && bu.state.shake_per_minima > 0
            @info "Shaking imported known structures."
            shake_res(
                known_files,
                bu.state.shake_per_minima,
                bu.state.shake_amp,
                bu.state.shake_cell_amp,
            )
            num_known = length(known_files) * (bu.state.shake_per_minima + 1)
        end

        ndata = length(glob_allow_abs(joinpath(outdir, "*.res")))
        if ndata >= required
            @info "Skipping initial structure generation for iteration $iter; found $ndata existing structures."
            return :ok
        end

        # First cycle generate from the seed without relaxation
        # Sanity check - are we definitely overfitting?
        if nfeatures(bu.cf) > bu.state.n_initial
            @warn "The number of features $(nfeatures(bu.cf)) is larger than the initial training size!"
        end
        # Generate random structures
        nstruct = required - ndata
        if nstruct > 0
            @info "Generating $(nstruct) initial training structures with $(buildcell_nprocs) buildcell process(es)."
            build_random_structures(
                joinpath.(workdir, seedfile),
                outdir;
                seedfile_weights,
                n=nstruct,
                nprocs=buildcell_nprocs,
                timeout=build_timeout,
                seed_text_transform,
            )
        end
    else
        # Subsequent cycles - generate from the seed and perform relaxation
        nstruct = required - ndata
        if nstruct > 0
            # Launch external processes or internal RSS
            result = if bu.state.rss_external
                _launch_rss_external(bu, iter, nstruct)
            else
                _launch_rss_internal(bu, iter, nstruct)
            end
            result == :stopped && return :stopped

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
    return :ok
end


"""
    _launch_rss_external(bu::Builder, iter::Int, nstruct::Int)

Launch external Julia processes for RSS generation.
"""
function _launch_rss_external(bu::Builder, iter::Int, nstruct::Int)
    @info "Generating $(nstruct) training structures for iteration $iter."
    _require_builder_file_path!(bu)

    # Distribute the workloads
    bu.state.rss_nprocs >= 1 ||
        throw(ArgumentError("state.rss_nprocs must be >= 1."))
    nstruct_per_proc = max(1, ceil(Int, nstruct / bu.state.rss_nprocs))
    nstructs = fill(nstruct_per_proc, bu.state.rss_nprocs)

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
                "using EDDPotentials;EDDPotentials.Workflow._run_rss_link()",
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
                "--relax-mode",
                "$(state.relax_mode)",
            ]);
            julia_threads=state.rss_num_threads,
        ) for num in nstructs
    ]

    # Apply random pressure range
    if !isnothing(state.rss_pressure_gpa_range) && !isempty(state.rss_pressure_gpa_range)
        # Add pressure ranges
        a, b = state.rss_pressure_gpa_range
        for i in eachindex(cmds)
            cmds[i] = add_threads_env(
                Cmd([cmds[i]..., "--pressure-range", "$(a),$(b)"]);
                julia_threads=state.rss_num_threads,
            )
        end
    end

    @info "Subprocess launch command: $(Cmd([cmds[1]...])) for $(state.rss_nprocs) process"

    # Launch processes
    tasks = Task[]
    processes = []
    for i = 1:state.rss_nprocs
        proc = run(
            pipeline(cmds[i], stdout="rss-process-$i-stdout", stderr="rss-process-$i");
            wait=false,
        )
        push!(processes, proc)
        this_task = @async wait(proc)
        push!(tasks, this_task)
    end

    @info "Search process launched, waiting..."
    last_nstruct = length(glob(joinpath(outdir, "*.res")))
    while !all(istaskdone.(tasks))
        if should_stop(bu)
            for proc in processes
                try
                    process_running(proc) && kill(proc)
                catch err
                    @warn "Failed to stop RSS process `$proc`: $err"
                end
            end
            return :stopped
        end
        nfound = length(glob(joinpath(outdir, "*.res")))
        # Print progress if the number of models have changed
        if nfound != last_nstruct
            @info "Number of relaxed structures: $(nfound)/$(state.per_generation)"
            last_nstruct = nfound
        end
        sleep(5)
    end
    foreach(fetch, tasks)
    @assert all(success, processes) "There are tasks with non-zero exit code!"
    return :ok
end


"""
    _populate_hyper_option(state::BuilderState; verbose::Union{Nothing,Bool}=nothing) -> RelaxOption

Create a RelaxOption from the BuilderState. `state.relax` is the single source
of hyperspace relaxation parameters.
"""
function _populate_hyper_option(state; verbose::Union{Nothing,Bool}=nothing)
    opt = deepcopy(state.relax)
    if verbose !== nothing
        opt.verbose = verbose
    end
    if state.relax_mode == "hyperspace" && opt.hyper_ndim <= 0
        opt.hyper_ndim = 3
    end
    opt
end

_normalized_relax_mode(state, opt) =
    state.relax_mode == "hyperspace" || opt.hyper_ndim > 0 ? "hyperspace" : state.relax_mode


"""
    _launch_rss_internal(bu::Builder, iter::Int, nstruct::Int)

Run RSS generation internally within the current process.
"""
function _launch_rss_internal(bu::Builder, iter::Int, nstruct::Int)
    @info "Generating $(nstruct) training structures for iteration $iter ($(bu.state.relax_mode) mode)."
    ensemble = load_ensemble(bu, iter - 1)
    state = bu.state
    outdir = joinpath(state.workdir, "gen$(iter)")
    ensure_dir(outdir)

    (; seedfile, seedfile_weights, ensemble_std_min, ensemble_std_max) = state
    _relax_option = _populate_hyper_option(state)
    relax_mode = _normalized_relax_mode(state, _relax_option)
    formula_ctx = _formula_sampling_context(state, bu.cfopt.elements)
    seed_text_transform = _make_seed_text_transform(formula_ctx)
    hot_kwargs = (; n_rounds=state.hot_n_rounds, md_temperature=state.hot_md_temperature,
        md_steps=state.hot_md_steps, md_timestep=state.hot_md_timestep,
        relax_cell=state.hot_relax_cell)

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
        relax_option=_relax_option,
        elemental_energies=make_symbol_keys(state.elemental_energies),
        relax_mode,
        hot_kwargs,
        seed_text_transform,
        prune_option=state.rss_prune,
        prune_random_composition=formula_ctx !== nothing,
        require_converged=state.rss_require_converged,
    )
    return :ok
end


"""
    run_rss(builder::Builder)

Run random structure search using trained ensemble model. The output files are in the
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
    state = builder.state
    _relax_option = deepcopy(rs.relax)
    _relax_option.verbose = state.rss_nprocs > 0
    _relax_mode = _normalized_relax_mode(state, _relax_option)
    formula_ctx = _formula_sampling_context(state, builder.cfopt.elements; seedfile=seedfile)
    seed_text_transform = _make_seed_text_transform(formula_ctx)
    hot_kwargs = (; n_rounds=state.hot_n_rounds, md_temperature=state.hot_md_temperature,
        md_steps=state.hot_md_steps, md_timestep=state.hot_md_timestep,
        relax_cell=state.hot_relax_cell)

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
        relax_option=_relax_option,
        elemental_energies=make_symbol_keys(state.elemental_energies),
        relax_mode=_relax_mode,
        hot_kwargs,
        seed_text_transform,
        prune_option=rs.prune,
        prune_random_composition=formula_ctx !== nothing,
        require_converged=rs.require_converged,
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


"""
    _run_rss_link()

Run the random search step as part of the `link!` iterative building protocol.
This function is intended to be called as a separated Julia process.

!!! note
    This function is meant to be called with commandline arguments.
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
        "--relax-mode"
        help = "Relaxation mode: standard, hyperspace, or hot"
        default = "standard"
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

    relax_mode = args["relax-mode"]
    state = bu.state
    _relax_option = deepcopy(state.relax)
    relax_mode = _normalized_relax_mode(state, _relax_option)
    formula_ctx = _formula_sampling_context(state, bu.cfopt.elements)
    seed_text_transform = _make_seed_text_transform(formula_ctx)
    hot_kwargs = (; n_rounds=state.hot_n_rounds, md_temperature=state.hot_md_temperature,
        md_steps=state.hot_md_steps, md_timestep=state.hot_md_timestep,
        relax_cell=state.hot_relax_cell)

    (; seedfile, seedfile_weights) = state
    _run_rss(
        joinpath.(Ref(state.workdir), seedfile),
        ensemble,
        bu.cf;
        core_size=state.core_size,
        ensemble_std_max=state.ensemble_std_max,
        ensemble_std_min=state.ensemble_std_min,
        max=args["num"],
        outdir=args["outdir"],
        pressure_gpa,
        pressure_gpa_range,
        seedfile_weights,
        niggli_reduce_output=state.rss_niggli_reduce,
        relax_option=_relax_option,
        elemental_energies=make_symbol_keys(state.elemental_energies),
        relax_mode,
        hot_kwargs,
        seed_text_transform,
        prune_option=state.rss_prune,
        prune_random_composition=formula_ctx !== nothing,
        require_converged=state.rss_require_converged,
    )
end


"""
    _perform_training(bu::Builder)

Carry out training and save the ensemble as a JLD2 archive.
"""
function _perform_training(bu::Builder)
    # Write the dataset to the disk
    @info "Preparing dataset..."
    write_dataset(bu)
    result = _perform_training_models(bu)
    result == :pending && throw(
        ErrorException(
            "Training jobs were submitted without waiting for model artifacts; rerun link! after the jobs finish to create the ensemble.",
        ),
    )
    result == :stopped && return nothing
    return create_iteration_ensemble!(bu)
end

function _perform_training_models(bu::Builder)
    if bu.trainer.external
        result = _perform_training_external(bu)
        result isa Symbol && return result
    else
        run_trainer(bu, bu.trainer)
    end
    return :ok
end


"""
    _perform_training_external(bu::Builder)

Launch external training workers.
"""
function _perform_training_external(bu::Builder)
    tra = bu.trainer
    mode = lowercase(strip(tra.external_mode))

    if mode == "local"
        _perform_training_external_local(bu)
    elseif mode == "submit" || mode == "command"
        return _perform_training_external_submit(bu)
    else
        throw(ArgumentError("Unknown trainer.external_mode `$mode`; expected `local` or `submit`."))
    end
    return :ok
end

function _training_julia_code(tra::TrainingOption)
    if tra.use_cuda == true
        return "using EDDPotentials;using CUDA;EDDPotentials.USE_CUDA[]=true;EDDPotentials.run_trainer()"
    end
    return "using EDDPotentials;EDDPotentials.run_trainer()"
end

function _training_base_cmd(bu::Builder)
    tra = bu.trainer
    project_path = dirname(Base.active_project())
    builder_file = _require_builder_file_path!(bu)

    Cmd([
        Base.julia_cmd()...,
        "--project=$(project_path)",
        "-e",
        _training_julia_code(tra),
        "$(builder_file)",
        "--iteration",
        "$(bu.state.iteration)",
    ])
end

function _training_worker_cmd(bu::Builder, worker_id::Int)
    tra = bu.trainer
    cmd = add_threads_env(
        Cmd([_training_base_cmd(bu)..., "--id", "$worker_id"]);
        julia_threads=tra.num_threads_per_worker,
    )
    if tra.cuda_visible_devices != ""
        devices = split(tra.cuda_visible_devices, ",")
        cmd = addenv(cmd, "EDDP_CUDA_DEVICE" => devices[(worker_id - 1) % length(devices) + 1])
    end
    cmd
end

function _shell_escape(s::AbstractString)
    "'" * replace(s, "'" => "'\\''") * "'"
end

_command_string(cmd::Cmd) = join(_shell_escape.(string.(collect(cmd))), " ")

const TRAIN_CMD_PLACEHOLDER = "{{EDDP_TRAIN_CMD}}"

function _external_required_models(tra::TrainingOption)
    max(1, ceil(Int, tra.nmodels * tra.external_completion_ratio))
end

function _wait_for_training_models(
    bu::Builder;
    tasks=Task[],
    processes=[],
    poll_after_tasks::Bool=true,
    cancel_callback=nothing,
)
    tra = bu.trainer
    required = _external_required_models(tra)
    started = time()
    last_nm = num_existing_models(bu)
    last_nm > 0 && @info "Number of trained models: $last_nm"

    while true
        nm = num_existing_models(bu)
        if nm >= required
            if cancel_callback !== nothing
                cancel_callback()
            end
            return :ok
        end

        if should_stop(bu)
            @warn "STOP file detected while waiting for training models."
            if cancel_callback !== nothing
                cancel_callback()
            end
            return :stopped
        end

        if !isempty(tasks) && all(istaskdone.(tasks))
            for task in tasks
                fetch(task)
            end
            for proc in processes
                if !success(proc)
                    throw(ErrorException("Training worker process failed: $proc"))
                end
            end
            if !poll_after_tasks
                throw(
                    ErrorException(
                        "Training workers finished with $nm models, need $required.",
                    ),
                )
            end
        end

        if tra.external_timeout > 0 && time() - started > tra.external_timeout
            throw(
                ErrorException(
                    "Timed out waiting for external training models: found $nm, need $required.",
                ),
            )
        end

        if nm != last_nm
            @info "Number of trained models: $nm/$required"
            last_nm = nm
        end
        sleep(tra.external_poll_interval)
    end
end

function _cancel_training_processes(processes)
    for proc in processes
        try
            if process_running(proc)
                @info "Stopping surplus training worker: $proc"
                kill(proc)
            end
        catch err
            @warn "Failed to stop surplus training worker `$proc`: $err"
        end
    end
end

function _perform_training_external_local(bu::Builder)
    tra = bu.trainer
    cmd = _training_base_cmd(bu)

    # Call multiple trainer processes
    @info "Subprocess launch command: $cmd"
    tasks = Task[]
    processes = []
    for i = 1:tra.num_workers
        # Run with ids
        _cmd = _training_worker_cmd(bu, i)
        proc = run(
            pipeline(_cmd, stdout="lm-process-$i-stdout", stderr="lm-process-$i-stderr");
            wait=false,
        )
        push!(processes, proc)
        this_task = @async wait(proc)
        push!(tasks, this_task)
    end

    @info "Training processes launched, waiting..."
    _wait_for_training_models(
        bu;
        tasks,
        processes,
        poll_after_tasks=false,
        cancel_callback=() -> _cancel_training_processes(processes),
    )
end

function _training_submit_payload(bu::Builder)
    worker_id_line = "WORKER_ID=\"\${SLURM_ARRAY_TASK_ID:-\${SGE_TASK_ID:-1}}\""
    train_cmd = _command_string(_training_base_cmd(bu)) * " --id \"\$WORKER_ID\""
    worker_id_line * "\n" * train_cmd
end

function _render_training_job_script(bu::Builder, scheduler::SchedulerConfig, workdir)
    mkpath(workdir)
    template = get_job_script_content(scheduler)
    if !contains(template, TRAIN_CMD_PLACEHOLDER)
        throw(
            ArgumentError(
                "Training scheduler script `$(scheduler.script)` must contain the placeholder `$TRAIN_CMD_PLACEHOLDER`.",
            ),
        )
    end
    rendered = replace(template, TRAIN_CMD_PLACEHOLDER => _training_submit_payload(bu))
    script_path = joinpath(workdir, "job.sh")
    open(script_path, "w") do handle
        write(handle, rendered)
    end
    script_path
end

function _training_submit_cmd(sch::SchedulerConfig, workdir)
    if sch.type == "SGE"
        prog = "qsub"
    elseif sch.type == "SLURM"
        prog = "sbatch"
    else
        throw(ArgumentError("Unknown scheduler.type `$(sch.type)`; expected `SGE` or `SLURM`."))
    end

    args = String[]
    if sch.njobs > 1
        sch.type == "SGE" && append!(args, String["-t", "1-$(sch.njobs)"])
        sch.type == "SLURM" && append!(args, String["--array=1-$(sch.njobs)"])
    end
    sch.type == "SGE" && push!(args, "-cwd")
    push!(args, "job.sh")
    Cmd(Cmd([prog, args...]), dir=workdir)
end

function _parse_scheduler_job_id(sch::SchedulerConfig, output::AbstractString)
    if sch.type == "SLURM"
        match_result = match(r"Submitted batch job\s+(\d+)", output)
        return match_result === nothing ? "" : match_result.captures[1]
    elseif sch.type == "SGE"
        match_result = match(r"Your job(?:-array)?\s+(\d+)", output)
        return match_result === nothing ? "" : match_result.captures[1]
    end
    return ""
end

function _cancel_scheduler_job(sch::SchedulerConfig, job_id::AbstractString, workdir)
    isempty(job_id) && return
    if sch.type == "SLURM"
        cmd = Cmd(Cmd(["scancel", job_id]), dir=workdir)
    elseif sch.type == "SGE"
        cmd = Cmd(Cmd(["qdel", job_id]), dir=workdir)
    else
        return
    end
    try
        @info "Cancelling surplus training scheduler job: $cmd"
        run(cmd)
    catch err
        @warn "Failed to cancel scheduler job `$job_id`: $err"
    end
end

function _perform_training_external_submit(bu::Builder)
    tra = bu.trainer
    scheduler = bu.options.scheduler
    isnothing(scheduler) &&
        throw(ArgumentError("A top-level `[scheduler]` config is required when trainer.external_mode = \"submit\"."))

    workdir = joinpath(bu.state.workdir, "training-work", "gen$(bu.state.iteration)")
    script_path = _render_training_job_script(bu, scheduler, workdir)
    @info "Rendered training job script: $script_path"

    job_id = ""
    if scheduler.submit_jobs
        cmd = _training_submit_cmd(scheduler, workdir)
        @info "Training job submission command: $cmd"
        submit_output = readchomp(cmd)
        !isempty(submit_output) && @info "Training job submission output: $submit_output"
        job_id = _parse_scheduler_job_id(scheduler, submit_output)
        if isempty(job_id)
            @warn "Could not parse scheduler job id from submission output; surplus cluster jobs cannot be cancelled automatically."
        end
    else
        @info "scheduler.submit_jobs=false; monitoring existing training artifacts without submitting jobs."
    end

    if tra.external_wait
        @info "Cluster training jobs submitted, waiting for model artifacts..."
        return _wait_for_training_models(
            bu;
            cancel_callback=() -> _cancel_scheduler_job(scheduler, job_id, workdir),
        )
    else
        @info "Cluster training jobs submitted without waiting for model artifacts."
        return :pending
    end
end
