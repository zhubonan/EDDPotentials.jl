#=
Build a development sysimage that contains EDDPotentials' dependencies, while
leaving EDDPotentials and CellBase editable outside the image.
=#
module DevSysimage

using Pkg
using TOML

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_SYSIMAGE_PATH = joinpath(PROJECT_ROOT, "eddpdev.so")

const EXCLUDED_PACKAGES = Set(["EDDPotentials", "CellBase"])

"""
    usage()

Return the command-line help text for the development sysimage builder.
"""
function usage()
    return """
    Build a development sysimage for EDDPotentials dependencies.

    Usage:
      julia scripts/dev_sysimg.jl [--output PATH]

    Options:
      --output PATH    Sysimage output path. Default: $(DEFAULT_SYSIMAGE_PATH)
      --help           Show this message.
    """
end

"""
    parse_args(args)

Parse command-line arguments for the sysimage builder.

Recognized options are `--output PATH`, `--output=PATH`, `--help`, and `-h`.
Returns `nothing` when help was printed.
"""
function parse_args(args)
    output = DEFAULT_SYSIMAGE_PATH

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help" || arg == "-h"
            println(usage())
            return nothing
        elseif arg == "--output"
            i == length(args) && error("--output requires a path")
            i += 1
            output = args[i]
        elseif startswith(arg, "--output=")
            output = split(arg, "="; limit=2)[2]
        else
            error("unknown argument: $arg\n\n$(usage())")
        end
        i += 1
    end

    return (; output=abspath(expanduser(output)))
end

"""
    expanduser(path)

Expand a leading `~` in a filesystem path, leaving all other paths unchanged.
"""
function expanduser(path::AbstractString)
    if path == "~"
        return homedir()
    elseif startswith(path, "~/")
        return joinpath(homedir(), path[3:end])
    else
        return String(path)
    end
end

"""
    needs_rebuild(sysimage_path, manifest_path)

Return `true` when the sysimage is missing or older than `Manifest.toml`.
"""
function needs_rebuild(sysimage_path::AbstractString, manifest_path::AbstractString)
    isfile(sysimage_path) || return true
    return stat(manifest_path).mtime > stat(sysimage_path).mtime
end

"""
    absolutize_manifest_paths!(manifest, project)

Rewrite relative `path` entries in a parsed `Manifest.toml` so they remain valid
when the manifest is copied into the temporary sysimage build environment.
"""
function absolutize_manifest_paths!(manifest::AbstractDict, project::AbstractString)
    for entries in values(get(manifest, "deps", Dict{String,Any}()))
        for entry in (entries isa AbstractVector ? entries : Any[entries])
            if haskey(entry, "path") && !isabspath(entry["path"])
                entry["path"] = normpath(joinpath(project, entry["path"]))
            end
        end
    end
    return manifest
end

"""
    setup_build_project!(build_project, project)

Copy `project`'s `Project.toml` and `Manifest.toml` into `build_project`.

The copied project is converted into a plain environment by removing package
identity fields (`name`, `uuid`, `authors`, `version`). This prevents
`EDDPotentials` itself from being treated as the package under development while
still preserving the dependency versions and local path dependencies recorded in
the manifest.
"""
function setup_build_project!(build_project::AbstractString, project::AbstractString)
    project_file = joinpath(project, "Project.toml")
    manifest_file = joinpath(project, "Manifest.toml")
    isfile(manifest_file) || error("project path does not contain Manifest.toml: $project")

    project_toml = TOML.parsefile(project_file)
    for key in ("name", "uuid", "authors", "version")
        delete!(project_toml, key)
    end
    manifest = TOML.parsefile(manifest_file)
    absolutize_manifest_paths!(manifest, project)

    open(joinpath(build_project, "Project.toml"), "w") do io
        TOML.print(io, project_toml)
    end
    open(joinpath(build_project, "Manifest.toml"), "w") do io
        TOML.print(io, manifest)
    end

    Pkg.activate(build_project)
    Pkg.instantiate()
    return build_project
end

"""
    import_packagecompiler()

Install and import `PackageCompiler` in a temporary environment, keeping it as a
build-time dependency rather than adding it to this package's project.
"""
function import_packagecompiler()
    mktempdir(; prefix="eddpdev-packagecompiler-") do compiler_project
        Pkg.activate(compiler_project)
        Pkg.add("PackageCompiler")
        @eval import PackageCompiler
    end
    return PackageCompiler
end

"""
    create_dev_sysimage(packagecompiler, root_packages; build_project, sysimage_path, excluded_packages)

Create the development sysimage from direct dependencies in `root_packages`,
excluding actively edited packages such as `EDDPotentials` and `CellBase`.
Transitive dependencies are resolved by PackageCompiler from `build_project`'s
manifest-backed environment.
"""
function create_dev_sysimage(
    packagecompiler::Module,
    root_packages::Vector{String};
    build_project::AbstractString,
    sysimage_path::AbstractString,
    excluded_packages::Set{String}=EXCLUDED_PACKAGES,
)
    pc = packagecompiler
    Pkg.activate(build_project)
    compile_packages = sort(unique(filter(!in(excluded_packages), root_packages)))

    @info "Root packages" root_packages
    @info "Excluded packages" excluded_packages
    @info "Packages included in sysimage" compile_packages

    mkpath(dirname(sysimage_path))
    return Base.invokelatest(
        pc.create_sysimage,
        compile_packages;
        sysimage_path,
        project=build_project,
        incremental=true,
    )
end

"""
    build(; output=DEFAULT_SYSIMAGE_PATH, project=PROJECT_ROOT)

Build the repository-local development sysimage and print the Julia command that
uses it. The default output is `eddpdev.so` at the repository root. If the
sysimage exists and is newer than or as new as `Manifest.toml`, the build is
skipped.
"""
function build(; output::AbstractString=DEFAULT_SYSIMAGE_PATH, project::AbstractString=PROJECT_ROOT)
    project = abspath(expanduser(project))
    output = abspath(expanduser(output))
    manifest_file = joinpath(project, "Manifest.toml")

    isfile(joinpath(project, "Project.toml")) ||
        error("project path does not contain Project.toml: $project")
    isfile(manifest_file) || error("project path does not contain Manifest.toml: $project")

    if !needs_rebuild(output, manifest_file)
        @info "System image is up to date" output manifest_file
        println()
        println("Use it with:")
        println("  julia --project=$project -J $output")
        return output
    end

    mktempdir(; prefix="eddpdev-sysimg-") do build_project
        setup_build_project!(build_project, project)
        root_packages = collect(keys(Pkg.project().dependencies))
        pc = import_packagecompiler()
        create_dev_sysimage(
            pc,
            root_packages;
            build_project,
            sysimage_path=output,
            excluded_packages=EXCLUDED_PACKAGES,
        )
    end

    @info "System image saved" output
    println()
    println("Use it with:")
    println("  julia --project=$project -J $output")
    return output
end

"""
    main(args=ARGS)

Command-line entry point for `julia scripts/dev_sysimg.jl`.
"""
function main(args=ARGS)
    parsed = parse_args(args)
    parsed === nothing && return nothing
    return build(; output=parsed.output)
end

end

using .DevSysimage: build

if abspath(PROGRAM_FILE) == @__FILE__
    DevSysimage.main(ARGS)
end
