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
    if bu.options.hpc.enabled
        hpc_run_dft_phase!(bu)
        return true
    end
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

        # First cycle generate from the seed without relaxation
        # Sanity check - are we definitely overfitting?
        if nfeatures(bu.cf) > bu.state.n_initial
            @warn "The number of features $(nfeatures(bu.cf)) is larger than the initial training size!"
        end
        # Generate random structures
        nstruct = bu.state.n_initial - ndata + num_known
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
        prune_option=state.rss_prune,
        prune_random_composition=formula_ctx !== nothing,
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
        prune_option=rs.prune,
        prune_random_composition=formula_ctx !== nothing,
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
        prune_option=state.rss_prune,
        prune_random_composition=formula_ctx !== nothing,
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

Launch external training workers.
"""
function _perform_training_external(bu::Builder)
    tra = bu.trainer
    mode = lowercase(strip(tra.external_mode))

    if bu.options.hpc.enabled
        hpc_run_training_phase!(bu; create_ensemble_model=false)
    elseif mode == "local"
        _perform_training_external_local(bu)
    elseif mode == "submit" || mode == "command"
        _perform_training_external_submit(bu)
    elseif mode == "hpc"
        hpc_run_training_phase!(bu; create_ensemble_model=false)
    else
        throw(ArgumentError("Unknown trainer.external_mode `$mode`; expected `local`, `submit`, or `hpc`."))
    end
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
    builder_file = bu.state.builder_file_path
    @assert builder_file != ""

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
            return
        end

        if should_stop(bu)
            @warn "STOP file detected while waiting for training models."
            if cancel_callback !== nothing
                cancel_callback()
            end
            return
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
        _wait_for_training_models(
            bu;
            cancel_callback=() -> _cancel_scheduler_job(scheduler, job_id, workdir),
        )
    else
        @info "Cluster training jobs submitted without waiting for model artifacts."
    end
end
