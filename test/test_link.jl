using EDDPotentials
using Test
using Logging


function prepare_folder()
    target = mktempdir()
    sourcedir = joinpath(@__DIR__, "pp3-example")
    for fname in readdir(sourcedir; join=false)
        path = joinpath(target, fname)
        cp(joinpath(sourcedir, fname), path)
    end
    return target
end


function prepare_folder(func)
    mktempdir() do target
        sourcedir = joinpath(@__DIR__, "pp3-example")
        for fname in readdir(sourcedir; join=false)
            path = joinpath(target, fname)
            cp(joinpath(sourcedir, fname), path)
        end
        func(target)
    end
end

function check_airss(; verbose=false)
    try
        buildcell = run(`buildcell`, devnull, devnull; wait=false)
        pp3 = run(`pp3`, devnull, devnull; wait=false)
    catch error
        if verbose
            @show error
            @show ENV["PATH"]
        end
        return false
    end
    return true
end

if check_airss(; verbose=true)
    logger = SimpleLogger(IOBuffer())
    with_logger(logger) do
        @testset "Link" begin
            prepare_folder() do target
                builder = Builder(joinpath(target, "link-test.toml"))
                @test builder.state.iteration == 0
                EDDPotentials.step!(builder)
                @test builder.state.iteration == 1
                EDDPotentials.link!(builder)
                @test builder.state.iteration == 3

                # Run walk forward tests
                EDDPotentials.walk_forward_tests(builder)

                # Load ensemble
                load_ensemble(builder, 0)
                ensemble = load_ensemble(builder, 1)
                @test length(ensemble.models) > 0

                # Loading features
                fc = load_features(builder)
                @test isa(fc, FeatureContainer)
                fc = load_features(builder, 0)
                @test isa(fc, FeatureContainer)

                # Perform rss
                builder.rss.ensemble_id = 1
                builder.rss.max = 1
                run_rss(builder)
                @test isdir(joinpath(builder.state.workdir, "search"))
            end
        end
    end
else
    @info "Skipping `link!` test as AIRSS is not available in the test environment."
end
