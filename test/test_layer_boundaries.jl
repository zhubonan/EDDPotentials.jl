using Test
using EDDPotentials

@testset "Layer Boundaries" begin
    root = normpath(joinpath(@__DIR__, ".."))
    srcdir = joinpath(root, "src")
    top_dirs = sort(filter(name -> isdir(joinpath(srcdir, name)), readdir(srcdir)))
    @test setdiff(top_dirs, ["experimental"]) ==
          ["analysis", "potential", "search", "simulation", "training", "workflow"]

    function include_targets(path)
        text = read(path, String)
        [m.captures[1] for m in eachmatch(r"^\s*include\(\"([^\"]+)\"\)"m, text)]
    end

    potential_targets = include_targets(joinpath(srcdir, "potential", "Potential.jl"))
    top_targets = include_targets(joinpath(srcdir, "EDDPotentials.jl"))
    top_text = read(joinpath(srcdir, "EDDPotentials.jl"), String)

    @test "potential/Potential.jl" in top_targets
    @test "simulation/Simulation.jl" in top_targets
    @test "training/Training.jl" in top_targets
    @test "search/Search.jl" in top_targets
    @test "workflow/Workflow.jl" in top_targets
    @test "analysis/Analysis.jl" in top_targets
    @test !("Model.jl" in top_targets)
    @test !("Tools.jl" in top_targets)
    @test !("eddpf90.jl" in top_targets)

    @test "feature/feature.jl" in potential_targets
    @test "gradient/gradient.jl" in potential_targets
    @test "nn/interface.jl" in potential_targets
    @test "calculator/eddpcalculator.jl" in potential_targets
    @test "utilities.jl" in potential_targets
    @test all(!startswith(target, "../") for target in potential_targets)
    @test all(!startswith(target, "link/") for target in potential_targets)
    @test all(!startswith(target, "functional/") for target in potential_targets)

    @test occursin(r"using\s+\.Potential", top_text)

    module_files = Dict(
        "Potential" => joinpath(srcdir, "potential", "Potential.jl"),
        "Training" => joinpath(srcdir, "training", "Training.jl"),
        "Simulation" => joinpath(srcdir, "simulation", "Simulation.jl"),
        "Search" => joinpath(srcdir, "search", "Search.jl"),
        "Analysis" => joinpath(srcdir, "analysis", "Analysis.jl"),
        "Workflow" => joinpath(srcdir, "workflow", "Workflow.jl"),
    )
    for (name, path) in module_files
        module_text = read(path, String)
        @test occursin("module $name", module_text)
    end
    @test occursin("include(\"data.jl\")", read(joinpath(srcdir, "training", "Training.jl"), String))
    @test occursin("include(\"training.jl\")", read(joinpath(srcdir, "training", "Training.jl"), String))
    @test occursin("include(\"rss.jl\")", read(joinpath(srcdir, "search", "Search.jl"), String))
    @test occursin("include(\"optimizer/abstract.jl\")", read(joinpath(srcdir, "simulation", "Simulation.jl"), String))
    @test occursin("include(\"link/link.jl\")", read(joinpath(srcdir, "workflow", "Workflow.jl"), String))
    @test isfile(joinpath(srcdir, "analysis", "plotting_recipes.jl"))
    @test !isdir(joinpath(srcdir, "analysis", "plotting"))
    @test isfile(joinpath(srcdir, "potential", "utils.jl"))
    @test !isdir(joinpath(srcdir, "potential", "utils"))
    @test isdefined(EDDPotentials, :Simulation)
    @test EDDPotentials.to === EDDPotentials.Potential.to
end
