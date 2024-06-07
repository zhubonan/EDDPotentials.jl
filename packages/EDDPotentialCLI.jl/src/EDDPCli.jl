module EDDPotentialCli

using Comonicon
using EDDPotential: link!, Builder, BuilderOption, to_toml, from_dict
using EDDPotential: run_rss as run_rss_eddp
using YAML
using TOML

# write your code here
"""
    link(fname::String="link.toml"; iter::Int=-1)

Perform iterative building of the EDDPotential potential

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

Generate default configuration file for building EDDPotential potential as well as running searching 
using an existing project.

# Flags

- `--all-items, -a`: Populate all fields.

"""
@cast function genconfig(seedfile, elements...; all_items::Bool=false)
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

"""
Command line interface for EDDPotential.jl
"""
@main

end # EDDPotentialCli.jl
