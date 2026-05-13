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
    @test "namespaces.jl" in top_targets
    @test !("eddpf90.jl" in top_targets)

    @test "feature/feature.jl" in core_targets
    @test "gradient/gradient.jl" in core_targets
    @test "nn/interface.jl" in core_targets
    @test "calculator/eddpcalculator.jl" in core_targets
    @test all(!startswith(target, "../") for target in core_targets)
    @test all(!startswith(target, "link/") for target in core_targets)
    @test all(!startswith(target, "functional/") for target in core_targets)

    @test occursin(r"using\s+\.EDDPotentialsCore", top_text)

    namespace_text = read(joinpath(srcdir, "namespaces.jl"), String)
    for name in [
        "Potential",
        "Model",
        "Training",
        "Search",
        "Analysis",
        "Workflow",
        "Tools",
    ]
        @test occursin("module $name", namespace_text)
    end
    @test isdefined(EDDPotentials, :Simulation)
end
