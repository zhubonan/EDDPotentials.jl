using Test

@testset "Source Topology" begin
    root = normpath(joinpath(@__DIR__, ".."))
    srcdir = joinpath(root, "src")
    entry = joinpath(srcdir, "EDDPotentials.jl")
    experimental = joinpath(srcdir, "experimental")

    expected_paths = [
        "data/preprocessing.jl",
        "training/lmsolve.jl",
        "training/lmsolve_batched.jl",
        "training/sgd_training.jl",
        "training/training.jl",
        "simulation/Simulation.jl",
        "simulation/md.jl",
        "simulation/opt.jl",
        "simulation/relax.jl",
        "simulation/optimizer/abstract.jl",
        "simulation/optimizer/constraints.jl",
        "simulation/optimizer/fire.jl",
        "simulation/optimizer/tpsd.jl",
        "simulation/optimizer/utils.jl",
        "search/buildcell.jl",
        "search/hot.jl",
        "search/minsep.jl",
        "search/model_structures.jl",
        "search/rss.jl",
        "search/transforms.jl",
        "analysis/evaluation.jl",
        "analysis/lazy_res.jl",
        "analysis/plotting.jl",
        "analysis/plotting/recipes.jl",
        "analysis/records.jl",
        "workflow/hyper.jl",
        "namespaces.jl",
    ]

    retired_paths = [
        "preprocessing.jl",
        "lmsolve.jl",
        "lmsolve_batched.jl",
        "sgd_training.jl",
        "training.jl",
        "md.jl",
        "opt.jl",
        "functional/functional.jl",
        "functional/buildcell.jl",
        "functional/relax.jl",
        "functional/rss.jl",
        "functional/transforms.jl",
        "hot.jl",
        "hyper.jl",
        "lazy_res.jl",
        "minsep.jl",
        "model_structures.jl",
        "records.jl",
        "plotting.jl",
        "plotting/recipes.jl",
        "evaluation.jl",
        "optimizer/abstract.jl",
        "optimizer/constraints.jl",
        "optimizer/fire.jl",
        "optimizer/tpsd.jl",
        "optimizer/utils.jl",
    ]

    function include_targets(path)
        text = read(path, String)
        [m.captures[1] for m in eachmatch(r"^\s*include\(\"([^\"]+)\"\)"m, text)]
    end

    seen = Set([entry])
    stack = [entry]
    while !isempty(stack)
        file = pop!(stack)
        base = dirname(file)
        for rel in include_targets(file)
            child = normpath(joinpath(base, rel))
            if isfile(child) && !(child in seen)
                push!(seen, child)
                push!(stack, child)
            end
        end
    end

    all_sources = String[]
    for (dir, _, files) in walkdir(srcdir)
        for name in files
            endswith(name, ".jl") || continue
            push!(all_sources, joinpath(dir, name))
        end
    end

    orphans = [
        f for f in all_sources
        if !(f in seen) && !startswith(f, experimental)
    ]

    @test isempty(orphans)
    @test all(rel -> isfile(joinpath(srcdir, rel)), expected_paths)
    @test all(rel -> !isfile(joinpath(srcdir, rel)), retired_paths)
end
