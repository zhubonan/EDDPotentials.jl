using Test
using EDDPotentials

@testset "Layer Boundaries" begin
    root = normpath(joinpath(@__DIR__, ".."))
    srcdir = joinpath(root, "src")

    function include_targets(path)
        text = read(path, String)
        [m.captures[1] for m in eachmatch(r"^\s*include\(\"([^\"]+)\"\)"m, text)]
    end

    core_targets = include_targets(joinpath(srcdir, "core", "EDDPotentialsCore.jl"))
    top_targets = include_targets(joinpath(srcdir, "EDDPotentials.jl"))
    top_text = read(joinpath(srcdir, "EDDPotentials.jl"), String)

    @test "core/EDDPotentialsCore.jl" in top_targets
    @test "data/preprocessing.jl" in top_targets
    @test "simulation/Simulation.jl" in top_targets
    @test "training/training.jl" in top_targets
    @test "search/rss.jl" in top_targets
    @test "link/link.jl" in top_targets
    @test !("eddpf90.jl" in top_targets)

    @test "feature/feature.jl" in core_targets
    @test "gradient/gradient.jl" in core_targets
    @test "nn/interface.jl" in core_targets
    @test "calculator/eddpcalculator.jl" in core_targets
    @test all(!startswith(target, "../") for target in core_targets)
    @test all(!startswith(target, "link/") for target in core_targets)
    @test all(!startswith(target, "functional/") for target in core_targets)

    @test occursin(r"using\s+\.EDDPotentialsCore", top_text)

    for name in [
        "Potential",
        "Model",
        "Tools",
        "Training",
        "Search",
        "Analysis",
        "Workflow",
    ]
        @test occursin("module $name", top_text)
    end
    @test occursin("include(\"training/training.jl\")", top_text)
    @test occursin("include(\"search/rss.jl\")", top_text)
    @test occursin("include(\"link/link.jl\")", top_text)
    @test isdefined(EDDPotentials, :Simulation)
end
