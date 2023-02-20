#=
Compile system image for accelerated loading of EDDP related stuff
=#
module SysImageBuild

using PackageCompiler
using PackageCompiler:
    get_compiler_cmd,
    create_sysimg_object_file,
    create_pkg_context,
    check_packages_in_project
using PackageCompiler:
    create_fresh_base_sysimage, gather_stdlibs_project, ensurecompiled, tempname
using PackageCompiler:
    compile_c_init_julia, active_project, NATIVE_CPU_TARGET, create_sysimg_from_object_file
using Pkg

function create_sysimage(
    packages::Union{Nothing,Symbol,Vector{String},Vector{Symbol}}=nothing;
    excluded_packages=String[],
    sysimage_path::String,
    project::String=dirname(active_project()),
    precompile_execution_file::Union{String,Vector{String}}=String[],
    precompile_statements_file::Union{String,Vector{String}}=String[],
    incremental::Bool=true,
    filter_stdlibs::Bool=false,
    cpu_target::String=NATIVE_CPU_TARGET,
    script::Union{Nothing,String}=nothing,
    sysimage_build_args::Cmd=``,
    include_transitive_dependencies::Bool=true,
    # Internal args
    base_sysimage::Union{Nothing,String}=nothing,
    julia_init_c_file=nothing,
    version=nothing,
    soname=nothing,
    compat_level::String="major",
    extra_precompiles::String="",
)
    # We call this at the very beginning to make sure that the user has a compiler available. Therefore, if no compiler 
    # is found, we throw an error immediately, instead of making the user wait a while before the error is thrown.
    get_compiler_cmd()

    if filter_stdlibs && incremental
        error("must use `incremental=false` to use `filter_stdlibs=true`")
    end

    ctx = create_pkg_context(project)

    if packages === nothing
        packages = collect(keys(ctx.env.project.deps))
        if ctx.env.pkg !== nothing
            push!(packages, ctx.env.pkg.name)
        end
    end

    packages = string.(vcat(packages))
    excluded_packages = string.(vcat(excluded_packages))
    precompile_execution_file = vcat(precompile_execution_file)
    precompile_statements_file = vcat(precompile_statements_file)

    check_packages_in_project(ctx, packages)

    # Instantiate the project

    @debug "instantiating project at $(repr(project))"
    Pkg.instantiate(ctx, verbose=true, allow_autoprecomp=false)

    if !incremental
        if base_sysimage !== nothing
            error("cannot specify `base_sysimage`  when `incremental=false`")
        end
        sysimage_stdlibs =
            filter_stdlibs ? gather_stdlibs_project(ctx) : stdlibs_in_sysimage()
        base_sysimage = create_fresh_base_sysimage(sysimage_stdlibs; cpu_target)
    else
        base_sysimage = something(base_sysimage, unsafe_string(Base.JLOptions().image_file))
    end

    ensurecompiled(project, packages, base_sysimage)

    packages_sysimg = Set{Base.PkgId}()

    if include_transitive_dependencies
        # We are not sure that packages actually load their dependencies on `using`
        # but we still want them to end up in the sysimage. Therefore, explicitly
        # collect their dependencies, recursively.

        frontier = Set{Base.PkgId}()
        deps = ctx.env.project.deps
        for pkg in packages
            # Add all dependencies of the package
            if ctx.env.pkg !== nothing && pkg == ctx.env.pkg.name
                push!(frontier, Base.PkgId(ctx.env.pkg.uuid, pkg))
            else
                uuid = ctx.env.project.deps[pkg]
                push!(frontier, Base.PkgId(uuid, pkg))
            end
        end
        copy!(packages_sysimg, frontier)
        new_frontier = Set{Base.PkgId}()
        while !(isempty(frontier))
            for pkgid in frontier
                deps = if ctx.env.pkg !== nothing && pkgid.uuid == ctx.env.pkg.uuid
                    ctx.env.project.deps
                else
                    ctx.env.manifest[pkgid.uuid].deps
                end
                pkgid_deps = [Base.PkgId(uuid, name) for (name, uuid) in deps]
                for pkgid_dep in pkgid_deps
                    if !(pkgid_dep in packages_sysimg) #
                        push!(packages_sysimg, pkgid_dep)
                        push!(new_frontier, pkgid_dep)
                    end
                end
            end
            copy!(frontier, new_frontier)
            empty!(new_frontier)
        end
    end

    # Exclude certain packages 
    packages = filter(x -> !(x in excluded_packages), packages)
    packages_sysimg = filter(x -> !(x.name in excluded_packages), packages_sysimg)

    @info "Included packages: $packages"
    @info "Included packages_sysimg: $packages_sysimg"

    # Create the sysimage
    object_file = tempname() * ".o"

    create_sysimg_object_file(
        object_file,
        packages,
        packages_sysimg;
        project,
        base_sysimage,
        precompile_execution_file,
        precompile_statements_file,
        cpu_target,
        script,
        sysimage_build_args,
        extra_precompiles,
        incremental,
    )
    object_files = [object_file]
    if julia_init_c_file !== nothing
        push!(
            object_files,
            compile_c_init_julia(julia_init_c_file, basename(sysimage_path)),
        )
    end
    create_sysimg_from_object_file(
        object_files,
        sysimage_path;
        compat_level,
        version,
        soname,
    )

    rm(object_file; force=true)

    if Sys.isapple()
        cd(dirname(abspath(sysimage_path))) do
            sysimage_file = basename(sysimage_path)
            cmd = `install_name_tool -id @rpath/$(sysimage_file) $sysimage_file`
            @debug "running $cmd"
            run(cmd)
        end
    end

    return nothing
end

const DEV_PKGS = ["../../CellBase/", "../", "../packages/EDDPTools.jl/"]

# Additional packages to be included in the sysimage
const PKGS = [
    "Revise",
    "Optim",
    "DataFrames",
    "AtomsBase",
    "Requires",
    "PyCall",
    "Plots",
    "PlotlyJS",
    "DirectQhull",
    "Documenter",
    "Flux",
    "ChainRulesTestUtils",
    "Glob",
    "JLD2",
    "PackageCompiler",
    "Zygote",
    "NNlib",
]

const EXCLUDED = ["EDDP", "CellBase", "EDDPTools"]

function setup_temp_project()
    dir = dirname(@__FILE__)
    target_project = joinpath(dirname(@__FILE__), "devenv/")

    # Clean the build project
    isfile(joinpath(target_project, "Project.toml")) &&
        rm(joinpath(target_project, "Project.toml"))
    isfile(joinpath(target_project, "Manifest.toml")) &&
        rm(joinpath(target_project, "Manifest.toml"))

    Pkg.activate(joinpath(dirname(@__FILE__), "devenv/"))

    # Add development packages
    dev_pkgs = [Pkg.PackageSpec(path=joinpath(dir, x)) for x in DEV_PKGS]
    Pkg.develop(dev_pkgs)
    Pkg.add(PKGS)
    Pkg.instantiate()
    target_project
end

"""
    build(sysimage_path=joinpath(DEPOT_PATH[1], "eddpdev.so");target_env="eddp", kwargs...)

Build the sysimage and deploy the environment to the target folder
"""
function build(
    sysimage_path=joinpath(DEPOT_PATH[1], "eddpdev.so");
    target_env="eddp",
    kwargs...,
)
    temp_project_path = setup_temp_project()
    create_sysimage(; excluded_packages=EXCLUDED, sysimage_path, kwargs...)
    @info "System image saved to $sysimage_path"
    # Copy the 
    if target_env !== nothing
        envpath = joinpath(Pkg.envdir(), target_env)
        isdir(envpath) || mkdir(envpath)
        # Deploy the environment
        for name in ["Project.toml", "Manifest.toml"]
            cp(joinpath(temp_project_path, name), joinpath(envpath, name))
        end
    end
end

end

using .SysImageBuild: build
export build