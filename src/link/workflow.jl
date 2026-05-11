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
using .EDDPotentialsCore: CellEmbedding, EnsembleNNInterface, load_model_artifact

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


"""
    Builder(str::AbstractString="link.toml")

Load the builder from a TOML/YAML file. The file contains nested key-values pairs
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
    builder.state.builder_file_path = str

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
function link!(builder::Builder)
    state = builder.state
    _check_dependencies()

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


"""
    _check_dependencies()

Check if external programs are available in the system.
"""
function _check_dependencies()
    names = ["crud.pl", "buildcell", "cabal"]
    @info "Check if $names are available."
    for name in names
        if run(`which $name`).exitcode != 0
            @info "External program $name not found! Please check the runtime environment."
        end
    end
end


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
    should_stop(bu) && return
    iter = bu.state.iteration

    if has_ensemble(bu, iter)
        bu.state.iteration += 1
        return bu
    end

    # Generate new structures
    _generate_random_structures(bu, iter)
    should_stop(bu) && return

    # Run external code for DFT calculations
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
function _generate_random_structures(bu::Builder, iter)
    # Generate new structures
    outdir = _input_structure_dir(bu)
    ensure_dir(outdir)
    ndata = length(glob_allow_abs(joinpath(outdir, "*.res")))
    (; seedfile, seedfile_weights, workdir) = bu.state
    formula_ctx = _formula_sampling_context(bu.state, bu.cfopt.elements)
    seed_text_transform = _make_seed_text_transform(formula_ctx)

    if iter == 0
        known_files = _import_known_structures!(bu.state, outdir)
        if !isempty(known_files) && bu.state.shake_per_minima > 0
            @info "Shaking imported known structures."
            shake_res(
                known_files,
                bu.state.shake_per_minima,
                bu.state.shake_amp,
                bu.state.shake_cell_amp,
            )
        end

        ndata = length(glob_allow_abs(joinpath(outdir, "*.res")))

        # First cycle generate from the seed without relaxation
        # Sanity check - are we definitely overfitting?
        if nfeatures(bu.cf) > bu.state.n_initial
            @warn "The number of features $(nfeatures(bu.cf)) is larger than the initial training size!"
        end
        # Generate random structures
        nstruct = bu.state.n_initial - ndata
        if nstruct > 0
            @info "Generating $(nstruct) initial training structures."
            build_random_structures(
                joinpath.(workdir, seedfile),
                outdir;
                seedfile_weights,
                n=nstruct,
                seed_text_transform,
            )
        end
    else
        # Subsequent cycles - generate from the seed and perform relaxation
        nstruct = bu.state.per_generation - ndata
        if nstruct > 0
            # Launch external processes or internal RSS
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


"""
    _launch_rss_external(bu::Builder, iter::Int, nstruct::Int)

Launch external Julia processes for RSS generation.
"""
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


"""
    _populate_hyper_option(state::BuilderState; verbose::Bool=true) -> RelaxOption

Create a RelaxOption from the BuilderState, copying hyper_* fields into the
RelaxOption when `relax_mode == "hyperspace"`. Uses `state.relax` as the base.
"""
function _populate_hyper_option(state; verbose::Bool=true)
    opt = deepcopy(state.relax)
    opt.verbose = verbose
    if state.relax_mode == "hyperspace"
        opt.hyper_ndim = state.hyper_ndim
        opt.hyper_alpha_start = state.hyper_alpha_start
        opt.hyper_schedule_every = state.hyper_schedule_every
        opt.hyper_target_iterations = state.hyper_target_iterations
        opt.hyper_dynamic_alpha = state.hyper_dynamic_alpha
        opt.hyper_force_ratio = state.hyper_force_ratio
        opt.hyper_max_alpha = state.hyper_max_alpha
        opt.hyper_ramp_alpha = state.hyper_ramp_alpha
        opt.hyper_alpha_ramp_factor = state.hyper_alpha_ramp_factor
        opt.hyper_phase1_alpha = state.hyper_phase1_alpha
        opt.hyper_phase1_fraction = state.hyper_phase1_fraction
    end
    opt
end


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

    (; seedfile, seedfile_weights, ensemble_std_min, ensemble_std_max, relax_mode) = state
    _relax_option = _populate_hyper_option(state)
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
    )
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
    _relax_mode = _relax_option.hyper_ndim > 0 ? "hyperspace" : state.relax_mode
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
    rs = bu.rss
    _relax_option = deepcopy(rs.relax)
    _relax_option.verbose = true
    if _relax_option.hyper_ndim > 0
        relax_mode = "hyperspace"
    end
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
        ensemble = create_ensemble(
            bu;
            clean_individual_models=bu.trainer.clean_models,
            save_ensemble_model=true,
        )
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

Launch external Julia processes for parallel training.
"""
function _perform_training_external(bu::Builder)
    tra = bu.trainer

    # Call multiple sub processes
    project_path = dirname(Base.active_project())
    builder_file = bu.state.builder_file_path
    @assert builder_file != ""

    julia_cmd = "using EDDPotentials;EDDPotentials.run_trainer()"
    if tra.use_cuda == true
        julia_cmd = "using EDDPotentials;using CUDA;EDDPotentials.USE_CUDA[]=true;EDDPotentials.run_trainer()"
    end

    cmd = Cmd([
        Base.julia_cmd()...,
        "--project=$(project_path)",
        "-e",
        julia_cmd,
        "$(builder_file)",
        "--iteration",
        "$(bu.state.iteration)",
    ])

    # Call multiple trainer processes
    @info "Subprocess launch command: $cmd"
    tasks = Task[]
    for i = 1:tra.num_workers
        # Run with ids
        _cmd = add_threads_env(Cmd([cmd..., "--id", "$i"]); julia_threads=tra.num_threads_per_worker)
        if tra.cuda_visible_devices != ""
            devices = split(tra.cuda_visible_devices, ",")
            _cmd = addenv(_cmd, "EDDP_CUDA_DEVICE" => devices[(i-1)%length(devices)+1])
        end
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
