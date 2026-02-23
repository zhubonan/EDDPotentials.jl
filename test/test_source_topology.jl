using Test

@testset "Source Topology" begin
    root = normpath(joinpath(@__DIR__, ".."))
    srcdir = joinpath(root, "src")
    entry = joinpath(srcdir, "EDDPotentials.jl")
    experimental = joinpath(srcdir, "experimental")

    function include_targets(path)
        text = read(path, String)
        [m.captures[1] for m in eachmatch(r"include\(\"([^\"]+)\"\)", text)]
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
end
