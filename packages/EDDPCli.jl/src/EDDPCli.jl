module EDDPCli

using Comonicon
using EDDP: link!, Builder, _run_rss, load_ensemble, ensure_dir
using EDDP: run_rss as run_rss_eddp
using YAML
using TOML

# write your code here
"""
    link(fname::String="link.toml"; iter::Int=-1)

Perform iterative building of the EDDP potential

# Args

- `fname`: Name of to be used the configuration file (default: link.toml).

# Options

- `--iter`: Iteration number, default is to determine automatically by inspecting the working directory.

"""
@cast function link(fname::AbstractString="link.toml"; iter::Int=-1)
    builder = Builder(fname.content)
    if iter >= 0
        builder.state.iteration = iter
    end
    link!(builder)
end

"""
    run_rss

Perform random structure searching

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
"""
@cast function yaml2toml(fname::AbstractString="link.yaml")
    dictin = YAML.load_file(fname)
    open(splitext(fname)[1] * ".toml", "w") do f
        TOML.print(f, dictin)
    end
end


"""
Command line interface for EDDP.jl
"""
@main

end # EDDPCli.jl
