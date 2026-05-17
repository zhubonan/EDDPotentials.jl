using Test

@testset "Source Topology" begin
    root = normpath(joinpath(@__DIR__, ".."))
    srcdir = joinpath(root, "src")
    entry = joinpath(srcdir, "EDDPotentials.jl")
    experimental = joinpath(srcdir, "experimental")

    expected_paths = [
        "potential/Potential.jl",
        "potential/constants.jl",
        "potential/nnls/nnls.jl",
        "potential/utils.jl",
        "potential/utilities.jl",
        "potential/feature/feature.jl",
        "potential/feature/cell.jl",
        "potential/feature/embedded.jl",
        "potential/feature/multi_embedded.jl",
        "potential/feature/spline_table.jl",
        "potential/embedding.jl",
        "potential/embedding_rules.jl",
        "potential/gradient/gradient.jl",
        "potential/gradient/repulsive_core.jl",
        "potential/gradient/workspace.jl",
        "potential/nn/interface.jl",
        "potential/nn/manual_backprop.jl",
        "potential/nn/linear.jl",
        "potential/nn/ensemble.jl",
        "potential/nn/flux.jl",
        "potential/nn/json_serialization.jl",
        "potential/nntools.jl",
        "potential/calculator.jl",
        "potential/calculator/eddpcalculator.jl",
        "potential/calculator/modifiers/modifiers.jl",
        "potential/calculator/modifiers/abstract.jl",
        "potential/calculator/modifiers/dimension_penalty.jl",
        "potential/calculator/modifiers/volume_constraint.jl",
        "potential/calculator/modifiers/additive_calc.jl",
        "potential/calculator/metacalculator.jl",
        "potential/pairpot.jl",
        "training/Training.jl",
        "training/data.jl",
        "training/lmsolve.jl",
        "training/lmsolve_batched.jl",
        "training/context.jl",
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
        "search/Search.jl",
        "search/buildcell.jl",
        "search/hot.jl",
        "search/hyper.jl",
        "search/minsep.jl",
        "search/model_structures.jl",
        "search/rss.jl",
        "search/transforms.jl",
        "workflow/Workflow.jl",
        "workflow/link/analysis.jl",
        "workflow/link/data.jl",
        "workflow/link/distributed.jl",
        "workflow/link/external.jl",
        "workflow/link/formula_sampling.jl",
        "workflow/link/hyper_builder.jl",
        "workflow/link/link.jl",
        "workflow/link/options.jl",
        "workflow/link/trainer.jl",
        "workflow/link/utils.jl",
        "workflow/link/workflow.jl",
        "analysis/Analysis.jl",
        "analysis/evaluation.jl",
        "analysis/lazy_res.jl",
        "analysis/plotting.jl",
        "analysis/plotting_recipes.jl",
        "analysis/records.jl",
    ]

    retired_paths = [
        "Potential.jl",
        "Model.jl",
        "Tools.jl",
        "Training.jl",
        "Search.jl",
        "Workflow.jl",
        "Analysis.jl",
        "tools.jl",
        "data/Data.jl",
        "data/preprocessing.jl",
        "training/Data.jl",
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
        "analysis/plotting/recipes.jl",
        "evaluation.jl",
        "optimizer/abstract.jl",
        "optimizer/constraints.jl",
        "optimizer/fire.jl",
        "optimizer/tpsd.jl",
        "optimizer/utils.jl",
        "core/EDDPotentialsCore.jl",
        "potential/Model.jl",
        "potential/utils/utils.jl",
        "training/data/preprocessing.jl",
        "link/analysis.jl",
        "link/client_server.jl",
        "link/distributed.jl",
        "link/data.jl",
        "link/external.jl",
        "link/formula_sampling.jl",
        "link/link.jl",
        "link/options.jl",
        "link/trainer.jl",
        "link/utils.jl",
        "link/workflow.jl",
        "workflow/link/client_server.jl",
        "workflow/hyper.jl",
        "workflow/hyper_builder.jl",
        "workflow/Search.jl",
        "workflow/Simulation.jl",
        "workflow/search/buildcell.jl",
        "workflow/search/hot.jl",
        "workflow/search/hyper.jl",
        "workflow/search/minsep.jl",
        "workflow/search/model_structures.jl",
        "workflow/search/rss.jl",
        "workflow/search/transforms.jl",
        "workflow/simulation/md.jl",
        "workflow/simulation/opt.jl",
        "workflow/simulation/relax.jl",
        "workflow/simulation/optimizer/abstract.jl",
        "workflow/simulation/optimizer/constraints.jl",
        "workflow/simulation/optimizer/fire.jl",
        "workflow/simulation/optimizer/tpsd.jl",
        "workflow/simulation/optimizer/utils.jl",
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
