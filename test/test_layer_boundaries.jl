using Test

@testset "Layer Boundaries" begin
    root = normpath(joinpath(@__DIR__, ".."))
    srcdir = joinpath(root, "src")

    function include_targets(path)
        text = read(path, String)
        [m.captures[1] for m in eachmatch(r"include\(\"([^\"]+)\"\)", text)]
    end

    core_targets = include_targets(joinpath(srcdir, "core", "EDDPotentialsCore.jl"))
    top_targets = include_targets(joinpath(srcdir, "EDDPotentials.jl"))
    top_text = read(joinpath(srcdir, "EDDPotentials.jl"), String)

    @test core_targets == [
        "constants.jl",
        "nnls/nnls.jl",
        "utils/utils.jl",
        "feature/feature.jl",
        "embedding.jl",
        "embedding_rules.jl",
        "gradient/gradient.jl",
        "nn/interface.jl",
        "nntools.jl",
        "calculator.jl",
        "calculator/modifiers/modifiers.jl",
        "calculator/metacalculator.jl",
        "pairpot.jl",
    ]

    @test top_targets == [
        "core/EDDPotentialsCore.jl",
        "lmsolve.jl",
        "preprocessing.jl",
        "eddpf90.jl",
        "training.jl",
        "tools.jl",
        "functional/functional.jl",
        "optimizer/abstract.jl",
        "optimizer/utils.jl",
        "optimizer/tpsd.jl",
        "optimizer/fire.jl",
        "opt.jl",
        "records.jl",
        "quickbuild.jl",
        "link/link.jl",
        "hyper.jl",
        "minsep.jl",
        "plotting/recipes.jl",
        "lazy_res.jl",
        "model_structures.jl",
        "plotting.jl",
    ]
    @test occursin(r"using\s+\.EDDPotentialsCore", top_text)
end
