module EDDPotentialsCli

using Comonicon
using EDDPotentials: link!, Builder, BuilderOption, to_toml, from_dict
using EDDPotentials: run_rss as run_rss_eddp
using EDDPotentials: load_model_artifact, EDDPCalculator, MetaCalculator, CellRelaxationModifier
using EDDPotentials: Relax, RelaxOption, multirelax!
using EDDPotentials: update_metadata!, eddp_energy, eddp_forces, eddp_stress
using EDDPotentials: eddp_pressure_gpa, eddp_enthalpy
using EDDPotentials: get_cell, stem, ensure_dir, external_stress_matrix
using EDDPotentials: _load_structures_from_path, write_res
using EDDPotentials: eVAngToGPa
using CellBase: volume, remove_dimensions
using ExtXYZ
using YAML
using TOML

# write your code here
"""
    link(fname::String="link.toml"; iter::Int=-1)

Perform iterative building of the EDDPotentials potential

# Intro

This function initialise a `Builder` object and run iterative training process.
Note: It can be useful to set `JULIA_NUM_THREADS` to the number of physical cores.
This will significantly accelerate the feature generation process at the starts of each iteration.


# Args

- `fname`: Configuration file name (default: `link.toml`).

# Options

- `--iter`: Iteration number, default is to determine automatically by inspecting the working directory.

"""
Comonicon.@cast function link(fname::AbstractString="link.toml"; iter::Int=-1)
    builder = Builder(fname)
    if iter >= 0
        builder.state.iteration = iter
    end
    link!(builder)
end

"""
    run_rss

Perform random structure searching.

# Intro

This function initialise a `Builder` object based on the configuration file passed and run
random structure search using the specified ensemble ID which defaults to the latest generation
of the trained model (`ensemble-gen<id>.jld2`) in the working directory.

# Arguments

- `fname`: Configuration file name (default: `link.toml`).
"""
Comonicon.@cast function run_rss(fname::AbstractString="link.toml";)
    builder = Builder(fname)
    run_rss_eddp(builder)
end

"""
    yaml2toml

Convenient converter to generate a TOML file from a YAML file.

# Intro

This tool can be used to convert a YAML format configuration file to TOML format.
The former format is now deprecated.

"""
Comonicon.@cast function yaml2toml(fname::AbstractString="link.yaml")
    dictin = YAML.load_file(fname)
    open(splitext(fname)[1] * ".toml", "w") do f
        TOML.print(f, dictin)
    end
end

"""
    genconfig

Generate default configuration file

# Intro

Generate default configuration file for building EDDPotentials potential as well as running searching
using an existing project.

# Flags

- `--all-items, -a`: Populate all fields.

"""
Comonicon.@cast function genconfig(seedfile, elements...; all_items::Bool=false)
    builder_opts = _get_builder_opt_template(seedfile, elements...)
    to_toml(stdout, builder_opts, include_defaults=all_items)
end

"""
    _get_builder_opt_template(seedfile, elements...)

Returns a template `BuilderOption` object
"""
function _get_builder_opt_template(seedfile, elements...)
    template = Dict{String,Any}(
        "cf" => Dict{String,Any}(
            "elements" => collect(elements),
            "geometry_sequence" => true,
            "rcut2" => 6.0,
            "p2" => [2, 10, 5],
            "p3" => [2, 10, 5],
        ),
        "state" => Dict{String,Any}(
            "seedfile" => seedfile,
            "seedfile_calc" => seedfile,
            "per_generation" => 100,
            "n_initial" => 1000,
            "shake_per_minima" => 10,
            "dft_mode" => "disp-castep",
        ),
        "trainer" => Dict{String,Any}("type" => "locallm", "log_file" => "train-log"),
        "rss" => Dict{String,Any}("packed" => true),
    )
    from_dict(BuilderOption, template)
end

# ==============================================================================
# relax command helpers
# ==============================================================================

_is_extxyz(path) = lowercase(splitext(path)[2]) in (".xyz", ".extxyz")

"""
    _make_output_path(input_path, outdir; suffix="_relaxed", index=nothing)

Determine the output file path based on input path and output directory.
"""
function _make_output_path(input_path, outdir; suffix="_relaxed", index=nothing)
    dir = isempty(outdir) ? dirname(input_path) : outdir
    base = basename(input_path)
    name, ext = splitext(base)
    if index !== nothing
        return joinpath(dir, "$(name)$(suffix)_$(index)$(ext)")
    else
        return joinpath(dir, "$(name)$(suffix)$(ext)")
    end
end

"""
    _relax_single(cell, model, feature_spec, elemental_energies; kwargs...)

Relax a single structure and return `(relaxed_cell, result, calc)`.
"""
function _relax_single(
    cell,
    model,
    feature_spec,
    elemental_energies;
    relax_cell=true,
    method="tpsd",
    max_steps=1000,
    f_tol=1e-2,
    s_tol=1e-1,
    pressure=0.0,
)
    cell = deepcopy(cell)
    calc = EDDPCalculator(cell, feature_spec, model; elemental_energies)

    if relax_cell
        smat = external_stress_matrix([pressure])
        calc = MetaCalculator(calc, [CellRelaxationModifier(get_cell(calc); external_pressure=smat)])
    end

    options = RelaxOption(;
        method,
        force_threshold=f_tol,
        stress_threshold=s_tol,
        relax_cell,
        iterations=max_steps,
    )
    re = Relax(calc, options)
    result = multirelax!(re; max_iter=max_steps)

    relaxed_cell = get_cell(calc)
    return (relaxed_cell, result, calc)
end

"""
    _attach_relax_metadata!(cell, calc, result; source_file, original_label)

Attach relaxation metadata to the cell for output writing.
"""
function _attach_relax_metadata!(cell, calc, result; source_file="", original_label="")
    # update_metadata! sets :enthalpy, :volume, :pressure, :label, :symm
    update_metadata!(calc, original_label)

    cell.metadata[:energy] = eddp_energy(calc)
    cell.metadata[:forces] = eddp_forces(calc)
    cell.metadata[:max_force] = result.fmax
    cell.metadata[:max_stress] = result.smax
    cell.metadata[:converged] = result.converged
    cell.metadata[:iterations] = result.iterations
    cell.metadata[:original_label] = original_label
    cell.metadata[:source_file] = source_file

    # For SHELX output: store extra metadata as REM key: value lines
    cell.metadata[:info] = Dict(
        "converged" => result.converged,
        "iterations" => result.iterations,
        "max_force" => result.fmax,
        "max_stress" => result.smax,
        "source_file" => source_file,
        "original_label" => original_label,
    )

    return cell
end

"""
    _write_relaxed_extxyz(output_path, cells)

Write relaxed cells to an extxyz file.
"""
function _write_relaxed_extxyz(output_path, cells)
    for (i, cell) in enumerate(cells)
        xyzdict = ExtXYZ.write_dict(remove_dimensions(cell))
        # Merge cell metadata into extxyz info dict
        for (k, v) in cell.metadata
            k === :info && continue  # skip the REM info dict
            xyzdict["info"][string(k)] = v
        end
        ExtXYZ.write_frame(output_path, xyzdict; append=(i > 1))
    end
end

# ==============================================================================
# relax command
# ==============================================================================

"""
    relax

Relax crystal structures using a trained EDDP model.

# Intro

Loads a trained EDDP model and relaxes one or more crystal structures.
Supports SHELX (.res) and extxyz (.xyz/.extxyz) input files.
ExtXYZ files may contain multiple structures; each is relaxed independently.
The output file arrangement mirrors the input: single-structure files produce
single-structure outputs, and multi-structure extxyz files produce a single
multi-structure extxyz output.

# Args

- `model_path`: Path to the model artifact file (.jld2 or .json).
- `structures`: One or more structure files to relax (.res, .xyz, .extxyz).

# Options

- `--outdir <dir>`: Output directory (default: same as input with `_relaxed` suffix).
- `--method <method>`: Optimization method: tpsd, lbfgs, bfgs, cg (default: tpsd).
- `--max-steps <n>`: Maximum number of optimization steps (default: 1000).
- `--f-tol <f>`: Force convergence tolerance in eV/Å (default: 0.01).
- `--s-tol <s>`: Stress convergence tolerance in GPa (default: 0.1).
- `--pressure <p>`: External pressure in GPa (default: 0.0).

# Flags

- `--no-cell-relax`: Disable cell relaxation (positions only).

"""
Comonicon.@cast function relax(
    model_path::AbstractString,
    structures::AbstractString...;
    outdir::AbstractString="",
    method::AbstractString="tpsd",
    max_steps::Int=1000,
    f_tol::Float64=1e-2,
    s_tol::Float64=1e-1,
    pressure::Float64=0.0,
    no_cell_relax::Bool=false,
)
    # Load model artifact (once, reuse for all structures)
    artifact = load_model_artifact(model_path)
    model = artifact.model
    feature_spec = artifact.feature_spec
    elemental_energies = artifact.elemental_energies

    feature_spec !== nothing ||
        error("Model artifact does not contain a feature spec. Cannot construct calculator.")

    relax_cell = !no_cell_relax

    # Create output directory if specified
    if !isempty(outdir)
        ensure_dir(outdir)
    end

    # Process each input file
    for input_path in structures
        is_extxyz_input = _is_extxyz(input_path)
        cells = _load_structures_from_path(input_path)
        n_cells = length(cells)

        @info "Loaded $n_cells structure(s) from $input_path"

        # Relax all structures from this input file
        relaxed_results = []

        for (idx, cell) in enumerate(cells)
            original_label = get(cell.metadata, :label, "$(stem(input_path))-$idx")

            try
                rcell, result, calc = _relax_single(
                    cell,
                    model,
                    feature_spec,
                    elemental_energies;
                    relax_cell,
                    method,
                    max_steps,
                    f_tol,
                    s_tol,
                    pressure,
                )

                _attach_relax_metadata!(
                    rcell,
                    calc,
                    result;
                    source_file=abspath(input_path),
                    original_label,
                )

                push!(relaxed_results, (rcell, result, calc))
                @info "  Structure $idx/$n_cells: converged=$(result.converged) iter=$(result.iterations) fmax=$(round(result.fmax; sigdigits=3)) smax=$(round(result.smax; sigdigits=3))"
            catch e
                @error "  Structure $idx/$n_cells failed: $e"
                rethrow()
            end
        end

        # Write output (mirrors input arrangement)
        if is_extxyz_input
            # extxyz: one output file per input file (mirrors input arrangement)
            output_path = _make_output_path(input_path, outdir)
            _write_relaxed_extxyz(output_path, [rcell for (rcell, _, _) in relaxed_results])
            @info "Wrote $(length(relaxed_results)) relaxed structure(s) to $output_path"
        else
            # SHELX .res: one file per structure
            for (i, (rcell, result, calc)) in enumerate(relaxed_results)
                if n_cells == 1
                    output_path = _make_output_path(input_path, outdir)
                else
                    output_path = _make_output_path(input_path, outdir; index=i)
                end
                write_res(output_path, calc; label=rcell.metadata[:label])
                @info "  Wrote $output_path"
            end
        end
    end
end

"""
Command line interface for EDDPotentials.jl
"""
Comonicon.@main

end # EDDPotentialsCli.jl
