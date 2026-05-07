using EDDPotentials
using Test
using Logging


function _env_dict(cmd::Base.Cmd)
    d = Dict{String,String}()
    for pair in cmd.env
        k, v = split(pair, "="; limit=2)
        d[k] = v
    end
    return d
end

@testset "add_threads_env" begin
    # Default: both threads set to 1
    cmd = EDDPotentials.add_threads_env(`echo hello`)
    env = _env_dict(cmd)
    @test env["JULIA_NUM_THREADS"] == "1"
    @test env["OMP_NUM_THREADS"] == "1"

    # Custom julia threads
    cmd = EDDPotentials.add_threads_env(`echo hello`; julia_threads=4)
    env = _env_dict(cmd)
    @test env["JULIA_NUM_THREADS"] == "4"
    @test env["OMP_NUM_THREADS"] == "1"

    # Custom omp threads
    cmd = EDDPotentials.add_threads_env(`echo hello`; omp_threads=2)
    env = _env_dict(cmd)
    @test env["JULIA_NUM_THREADS"] == "1"
    @test env["OMP_NUM_THREADS"] == "2"

    # Both custom
    cmd = EDDPotentials.add_threads_env(`echo hello`; julia_threads=4, omp_threads=2)
    env = _env_dict(cmd)
    @test env["JULIA_NUM_THREADS"] == "4"
    @test env["OMP_NUM_THREADS"] == "2"

    # Original command is preserved
    @test cmd.exec == Cmd(`echo hello`).exec

    # Existing env vars on the cmd are preserved
    base = setenv(`echo hello`, "MY_VAR" => "myval")
    cmd = EDDPotentials.add_threads_env(base; julia_threads=2)
    env = _env_dict(cmd)
    @test env["JULIA_NUM_THREADS"] == "2"
    @test env["OMP_NUM_THREADS"] == "1"
    @test env["MY_VAR"] == "myval"
end


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
                @test isa(fc, FeatureStore)
                fc = load_features(builder, 0)
                @test isa(fc, FeatureStore)

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
