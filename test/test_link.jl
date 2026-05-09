using EDDPotentials
using CellBase
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

function with_fake_buildcell(f; sleep_seconds=0.0)
    mktempdir() do bindir
        timeout_log = joinpath(bindir, "timeout.log")
        start_log = joinpath(bindir, "buildcell-starts.log")
        build_output = joinpath(bindir, "buildcell-output.cell")
        sample_cell = CellBase.read_res(joinpath(@__DIR__, "data", "training", "8B-22-09-16-17-27-38-08314cef.res"))
        CellBase.write_cell(build_output, sample_cell)

        timeout_script = joinpath(bindir, "timeout")
        open(timeout_script, "w") do io
            write(
                io,
                "#!/bin/sh\n" *
                "set -eu\n" *
                "printf '%s\\n' \"\$1\" >> \"\$FAKE_TIMEOUT_LOG\"\n" *
                "shift\n" *
                "exec \"\$@\"\n",
            )
        end
        chmod(timeout_script, 0o755)

        buildcell_script = joinpath(bindir, "buildcell")
        open(buildcell_script, "w") do io
            write(
                io,
                "#!/bin/sh\n" *
                "set -eu\n" *
                "date +%s.%N >> \"\$FAKE_BUILDCELL_START_LOG\"\n" *
                "sleep \"\${FAKE_BUILDCELL_SLEEP:-0}\"\n" *
                "cat \"\$FAKE_BUILDCELL_OUTPUT\"\n",
            )
        end
        chmod(buildcell_script, 0o755)

        withenv(
            "PATH" => string(bindir, ":", ENV["PATH"]),
            "FAKE_TIMEOUT_LOG" => timeout_log,
            "FAKE_BUILDCELL_START_LOG" => start_log,
            "FAKE_BUILDCELL_SLEEP" => string(sleep_seconds),
            "FAKE_BUILDCELL_OUTPUT" => build_output,
        ) do
            f((; timeout_log, start_log))
        end
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

@testset "initial buildcell generation" begin
    prepare_folder() do target
        with_fake_buildcell(; sleep_seconds=0.3) do logs
            builder = Builder(joinpath(target, "link-test.toml"))
            builder.state.n_initial = 4
            builder.state.buildcell_nprocs = 2
            builder.state.build_timeout = 7.5

            EDDPotentials._generate_random_structures(builder, 0)

            outdir = joinpath(builder.state.workdir, "gen0")
            resfiles = filter(x -> endswith(x, ".res"), readdir(outdir; join=true))
            @test length(resfiles) == 4

            timeout_args = readlines(logs.timeout_log)
            @test timeout_args == fill("7.5", 4)

            starts = sort(parse.(Float64, readlines(logs.start_log)))
            @test length(starts) == 4
            @test minimum(diff(starts)) < 0.2
        end
    end
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
