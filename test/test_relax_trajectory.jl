using Test
using EDDPotentials
using CellBase
using LinearAlgebra

isdefined(Main, :_h2_cell) || include("utils.jl")
const relax_traj_lj_like_calc = EDDPotentials.lj_like_calc

@testset "Relaxation Trajectory Monitoring" begin
    @testset "trajectory options" begin
        opts = RelaxOption()
        @test opts.trajectory_format == "csv_extxyz"
        @test opts.trajectory_every == 1
        @test_throws ArgumentError EDDPotentials.Simulation._normalize_trajectory_format("yaml")
    end

    @testset "native optimizer frames" begin
        for relax_fn in (opt_tpsd, opt_fire)
            cell = _h2_cell(10.0, 1.0)
            calc = relax_traj_lj_like_calc(cell; α=0.5, a=2, rc=8.0)
            opt = relax_fn(calc;
                itermax=3,
                convergence="force",
                f_tol=0.0,
                relax_cell=false,
                keep_trajectory=true,
                trajectory_every=1,
                maxstep=0.05)

            @test opt.iteration == 3
            @test length(opt.trajectory) == 3
            frame = opt.trajectory[end]
            @test frame.iteration == opt.iteration
            @test isfinite(frame.energy)
            @test isfinite(frame.enthalpy)
            @test isfinite(frame.fmax)
            @test isfinite(frame.smax_gpa)
            @test isfinite(frame.step_energy_de)
            @test isfinite(frame.step_energy_derun)
            @test frame.step_norm >= 0.0
            @test frame.convergence == "force"
            @test frame.is_final
        end
    end

    @testset "relax trajectory file export and metadata" begin
        mktempdir() do dir
            cell = _h2_cell(10.0, 1.0)
            calc = relax_traj_lj_like_calc(cell; α=0.5, a=2, rc=8.0)
            stem = joinpath(dir, "direct")

            result = relax!(calc;
                multi=false,
                method="tpsd",
                iterations=3,
                convergence="force",
                force_threshold=0.0,
                relax_cell=false,
                maxstep=0.05,
                trajectory_file=stem,
                trajectory_format="csv_extxyz",
                trajectory_every=1)

            csv = "$(stem).relax.csv"
            xyz = "$(stem).relax.extxyz"
            @test isfile(csv)
            @test isfile(xyz)
            csv_text = read(csv, String)
            @test occursin("iteration,energy,enthalpy,fmax,smax_gpa", csv_text)
            @test occursin(",1\n", csv_text)
            xyz_text = read(xyz, String)
            @test occursin("Properties=species:S:1:pos:R:3:force:R:3", xyz_text)
            @test occursin("final=T", xyz_text)

            meta = get_cell(calc).metadata
            @test meta[:relax_iterations] == result.iterations
            @test meta[:relax_converged] == result.converged
            @test meta[:relax_stop_reason] == "iteration_limit"
            @test meta[:relax_fmax] ≈ result.fmax
            @test haskey(meta, :relax_step_energy)
            @test haskey(meta, :relax_derun)
        end
    end

    @testset "RSS writes per-candidate trajectories" begin
        mktempdir() do dir
            buildcell = joinpath(dir, "buildcell")
            write(buildcell, """
#!/bin/sh
cat >/dev/null
cat <<'CELL'
%BLOCK LATTICE_CART
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
H 0.0 0.0 0.0
H 0.0 0.0 0.2
%ENDBLOCK POSITIONS_FRAC
CELL
""")
            chmod(buildcell, 0o755)

            seed = joinpath(dir, "H.cell")
            write(seed, "VARVOL=10\n")
            outdir = joinpath(dir, "rss")
            cf = CellFeature([:H]; p2=[2.0], p3=Float64[], q3=Float64[])
            model = LinearInterface(fill(0.1, nfeatures(cf)))
            opts = RelaxOption(;
                iterations=2,
                convergence="force",
                force_threshold=0.0,
                relax_cell=false,
                maxstep=0.05,
                trajectory_file="enabled",
                trajectory_format="csv_extxyz",
                trajectory_every=1)

            old_path = get(ENV, "PATH", "")
            ENV["PATH"] = "$(dir):$(old_path)"
            try
                run_rss(seed, model, cf;
                    max=1,
                    outdir,
                    max_err=1,
                    pressure_gpa=0.0,
                    relax_option=opts,
                    core_size=0.0,
                    niggli_reduce_output=false,
                    show_progress=false)
            finally
                ENV["PATH"] = old_path
            end

            csvs = filter(endswith(".relax.csv"), readdir(outdir; join=true))
            xyzs = filter(endswith(".relax.extxyz"), readdir(outdir; join=true))
            ress = filter(endswith(".res"), readdir(outdir; join=true))
            @test length(csvs) == 1
            @test length(xyzs) == 1
            @test length(ress) == 1
            res_text = read(ress[1], String)
            @test occursin("REM relax_iterations: 2", res_text)
            @test occursin("REM relax_stop_reason: iteration_limit", res_text)
            @test occursin("REM relax_fmax:", res_text)
        end
    end

    @testset "RSS can reject non-converged candidates" begin
        mktempdir() do dir
            buildcell = joinpath(dir, "buildcell")
            write(buildcell, """
#!/bin/sh
cat >/dev/null
cat <<'CELL'
%BLOCK LATTICE_CART
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
H 0.0 0.0 0.0
H 0.0 0.0 0.2
%ENDBLOCK POSITIONS_FRAC
CELL
""")
            chmod(buildcell, 0o755)

            seed = joinpath(dir, "H.cell")
            write(seed, "VARVOL=10\n")
            outdir = joinpath(dir, "rss")
            cf = CellFeature([:H]; p2=[2.0], p3=Float64[], q3=Float64[])
            model = LinearInterface(fill(0.1, nfeatures(cf)))
            opts = RelaxOption(;
                iterations=2,
                convergence="force",
                force_threshold=0.0,
                relax_cell=false,
                maxstep=0.05)

            old_path = get(ENV, "PATH", "")
            ENV["PATH"] = "$(dir):$(old_path)"
            try
                @test_throws ErrorException run_rss(seed, model, cf;
                        max=1,
                        outdir,
                        max_err=1,
                        pressure_gpa=0.0,
                        relax_option=opts,
                        core_size=0.0,
                        niggli_reduce_output=false,
                        show_progress=false,
                        require_converged=true)
            finally
                ENV["PATH"] = old_path
            end

            ress = filter(endswith(".res"), readdir(outdir; join=true))
            @test isempty(ress)
        end
    end
end
