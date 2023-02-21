module EDDPCli

using Comonicon
using EDDP:
    link!,
    Builder,
    _run_rss,
    load_ensemble,
    ensure_dir,
    BuilderState,
    CellFeature,
    TrainingOption,
    RssSetting
using EDDP:
    run_rss as run_rss_eddp, _make_string_keys, _make_symbol_keys, _fromdict, _todict
using YAML
using TOML

# write your code here
"""
    link(fname::String="link.toml"; iter::Int=-1)

Perform iterative building of the EDDP potential

# Intro

This function initialise a `Builder` object and run iterative training process.
Note: It can be useful to set `JULIA_NUM_THREADS` to the number of physical cores.
This will significantly accelerate the feature generation process at the starts of each iteration.


# Args

- `fname`: Name of to be used the configuration file (default: link.toml).

# Options

- `--iter`: Iteration number, default is to determine automatically by inspecting the working directory.

"""
@cast function link(fname::AbstractString="link.toml"; iter::Int=-1)
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
of the train model (`ensebmle-gen<id>.jld2`) in the working directory. 

# Arguments

- `fname`: Name of to be used the configuration file (default: link.toml).
"""
@cast function run_rss(fname::AbstractString="link.toml";)
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
@cast function yaml2toml(fname::AbstractString="link.yaml")
    dictin = YAML.load_file(fname)
    open(splitext(fname)[1] * ".toml", "w") do f
        TOML.print(f, dictin)
    end
end

"""
    genconfig

Generate default configuration file

# Intro

Generate default configuration file for building EDDP potential as well as running searching 
using an existing project.

# Flags

- `--all-items, -a`: Populate all fields.

"""
@cast function genconfig(seedfile, elements...; all_items::Bool=false)
    out = Dict{String,Any}(
        "cf" => Dict{String,Any}(
            "elements" => collect(elements),
            "geometry_sequence" => true,
            "rcut2" => 6.0,
            "p2" => [2, 10, 5],
            "p3" => [2, 10, 5],
        ),
        "state" => Dict{String,Any}(
            "seedfile" => seedfile,
            "per_generation" => 100,
            "n_initial" => 1000,
            "shake_per_minima" => 10,
            "dft_mode" => "disp-castep",
        ),
        "trainer" => Dict{String,Any}("type" => "locallm", "log_file" => "train-log"),
        "rss" => Dict{String,Any}("packed" => true),
    )
    if all_items
        builder = _fromdict(Builder, _make_symbol_keys(out))
        TOML.print(_make_string_keys(_todict(builder)); sorted=true)
    else
        TOML.print(out; sorted=true)
    end
end

"""
Command line interface for EDDP.jl
"""
@main

end # EDDPCli.jl
