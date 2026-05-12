using EDDPotentials
using CellBase
using Test

function with_fake_buildcell_formula_diag(f)
    mktempdir() do bindir
        sample_cell = Cell(Lattice(6, 6, 6), [:Al, :O], [0.0 1.5; 0.0 1.5; 0.0 1.5])
        build_output = joinpath(bindir, "buildcell-output.cell")
        CellBase.write_cell(build_output, sample_cell)

        timeout_script = joinpath(bindir, "timeout")
        open(timeout_script, "w") do io
            write(io, "#!/bin/sh\nset -eu\nshift\nexec \"\$@\"\n")
        end
        chmod(timeout_script, 0o755)

        buildcell_script = joinpath(bindir, "buildcell")
        open(buildcell_script, "w") do io
            write(io, "#!/bin/sh\nset -eu\ncat \"\$FAKE_BUILDCELL_OUTPUT\"\n")
        end
        chmod(buildcell_script, 0o755)

        withenv(
            "PATH" => string(bindir, ":", ENV["PATH"]),
            "FAKE_BUILDCELL_OUTPUT" => build_output,
        ) do
            f()
        end
    end
end

@testset "Formula Sampling" begin
    @test EDDPotentials.BuilderState(
        seedfile="seed.cell",
        seedfile_calc="calc.cell",
    ).known_structure_dir == "known_structures"

    @testset "Seed Directive Injection" begin
        seed = """
        #SPECIES=Si,O
        #NATOM=4-16
        #FORMULA=SiO
        #VARVOL=999
        #SLACK=0.25
        """
        out = EDDPotentials._inject_formula_directive(seed, "SiO2")
        @test occursin(r"(?m)^#FORMULA=SiO2$", out)
        @test !occursin(r"(?m)^\\s*#SPECIES\\s*=", out)
        @test !occursin(r"(?m)^\\s*#NATOM\\s*=", out)
        @test !occursin(r"(?m)^\\s*#FORMULA\\s*=SiO$", out)

        out_varvol = EDDPotentials._inject_formula_directive(seed, "Al2O3"; varvol=20.0)
        @test occursin(r"(?m)^#FORMULA=Al2O3$", out_varvol)
        @test occursin(r"(?m)^#VARVOL=20$", out_varvol)
        @test !occursin(r"(?m)^\\s*#VARVOL\\s*=999$", out_varvol)
    end

    @testset "Oxidation-State Filtered Formula Pool" begin
        mktempdir() do temp
            seedfile = joinpath(temp, "LTO-SEARCH.cell")
            write(
                seedfile,
                """
                #SPECIES=Li,Ti,O
                #NFORM=1-2
                #NATOM=4-24
                """,
            )

            state = EDDPotentials.BuilderState(
                seedfile="LTO-SEARCH.cell",
                seedfile_calc="LTO.cell",
                workdir=temp,
                formula_sampling_enabled=true,
                formula_max_reduced_formula_coeff=3,
                formula_require_charge_neutral=true,
                formula_oxidation_states=Dict(
                    "Li" => [1],
                    "Ti" => [4],
                    "O" => [-2],
                ),
            )

            forms = EDDPotentials._build_formula_pool(state, [:Li, :Ti, :O])
            @test !isempty(forms)

            lto = reduce_composition(Composition("Li2TiO3"))
            @test any(forms) do f
                reduce_composition(Composition(f)) == lto
            end

            ox = Dict(:Li => [1], :Ti => [4], :O => [-2])
            @test all(forms) do f
                EDDPotentials._has_neutral_oxidation_state(Composition(f), ox)
            end
        end
    end

    @testset "Known Structure Import" begin
        mktempdir() do temp
            kdir = joinpath(temp, "known_structures")
            outdir = joinpath(temp, "gen0")
            mkpath(kdir)
            mkpath(outdir)

            cell = Cell(Lattice(6, 6, 6), [:Li, :O], [[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]])
            write_res(joinpath(kdir, "known-a.res"), cell)

            state = EDDPotentials.BuilderState(
                seedfile="dummy.cell",
                seedfile_calc="dummy.cell",
                workdir=temp,
            )
            imported = EDDPotentials._import_known_structures!(state, outdir)
            @test length(imported) == 1
            @test isfile(joinpath(outdir, "known-a.res"))
        end
    end

    @testset "Formula VARVOL From Reduced Formula" begin
        mktempdir() do temp
            write(joinpath(temp, "seed.cell"), "#NFORM=1\n")
            state = EDDPotentials.BuilderState(
                seedfile="seed.cell",
                seedfile_calc="dummy.cell",
                workdir=temp,
                formula_sampling_enabled=true,
                formula_list=["Al2O3", "Al2O3Sc"],
                formula_target_atom_volumes=Dict(
                    "Al" => 10.0,
                    "O" => 10.0,
                    "Sc" => 10.0,
                ),
            )
            ctx = EDDPotentials._formula_sampling_context(state, ["Al", "O", "Sc"])
            @test ctx.varvol_by_formula["Al2O3"] == 20.0
            @test ctx.varvol_by_formula["Al2O3Sc"] == 30.0

            transform = EDDPotentials._make_seed_text_transform(ctx)
            out = transform("#SPECIES=Al,O,Sc\n#VARVOL=1\n")
            @test occursin(r"(?m)^#FORMULA=", out)
            @test occursin(r"(?m)^#VARVOL=(20|30)$", out)
        end
    end

    @testset "Formula Sampling Diagnosis" begin
        mktempdir() do temp
            write(
                joinpath(temp, "seed.cell"),
                "#SPECIES=Al,O,Sc\n#NFORM=1\n#VARVOL=1\n\n%BLOCK EXTERNAL_PRESSURE\n0 0 0\n0 0\n0\n%ENDBLOCK EXTERNAL_PRESSURE\n",
            )
            state = EDDPotentials.BuilderState(
                seedfile="seed.cell",
                seedfile_calc="dummy.cell",
                workdir=temp,
                formula_sampling_enabled=true,
                formula_list=["Al2O3", "Al2O3Sc"],
                formula_target_atom_volumes=Dict(
                    "Al" => 10.0,
                    "O" => 10.0,
                    "Sc" => 10.0,
                ),
            )
            io = IOBuffer()
            results = EDDPotentials.diagnose_formula_sampling(
                state,
                ["Al", "O", "Sc"];
                nsamples=2,
                io,
            )
            out = String(take!(io))
            @test length(results) == 2
            @test occursin("Sample 1", out)
            @test occursin("----------------------------------------", out)
            @test occursin("User settings", out)
            @test occursin("buildcell input", out)
            @test occursin("formula = ", out)
            @test occursin("seedfile = ", out)
            @test occursin("#FORMULA=", out)
            @test occursin("#VARVOL=", out)
            @test occursin("%BLOCK EXTERNAL_PRESSURE", out)
            @test all(r -> r.formula in ["Al2O3", "Al2O3Sc"], results)
            @test all(r -> r.varvol in (20.0, 30.0), results)
        end
    end

    @testset "Formula Sampling Diagnosis Save" begin
        with_fake_buildcell_formula_diag() do
            mktempdir() do temp
                write(joinpath(temp, "seed.cell"), "#SPECIES=Al,O\n#NFORM=1\n")
                state = EDDPotentials.BuilderState(
                    seedfile="seed.cell",
                    seedfile_calc="dummy.cell",
                    workdir=temp,
                    formula_sampling_enabled=true,
                    formula_list=["Al2O3"],
                    formula_target_atom_volumes=Dict(
                        "Al" => 10.0,
                        "O" => 10.0,
                    ),
                )
                io = IOBuffer()
                results = EDDPotentials.diagnose_formula_sampling(
                    state,
                    ["Al", "O"];
                    nsamples=1,
                    io,
                    save=true,
                )
                out = String(take!(io))
                @test length(results) == 1
                @test !isnothing(results[1].saved_seed_path)
                @test !isnothing(results[1].saved_buildcell_output_path)
                @test isfile(results[1].saved_seed_path)
                @test isfile(results[1].saved_buildcell_output_path)
                @test occursin("#FORMULA=Al2O3", read(results[1].saved_seed_path, String))
                @test occursin("saved_seed =", out)
                @test occursin("saved_buildcell_output =", out)
            end
        end
    end

    @testset "Formula VARVOL Missing Element Errors" begin
        mktempdir() do temp
            write(joinpath(temp, "seed.cell"), "#NFORM=1\n")
            state = EDDPotentials.BuilderState(
                seedfile="seed.cell",
                seedfile_calc="dummy.cell",
                workdir=temp,
                formula_sampling_enabled=true,
                formula_list=["Al2O3Sc"],
                formula_target_atom_volumes=Dict(
                    "Al" => 10.0,
                    "O" => 10.0,
                ),
            )
            @test_throws ErrorException EDDPotentials._formula_sampling_context(state, ["Al", "O", "Sc"])
        end
    end

    @testset "Formula Sampling Requires Single Seed" begin
        mktempdir() do temp
            write(joinpath(temp, "A.cell"), "#NFORM=1\n")
            write(joinpath(temp, "B.cell"), "#NFORM=1\n")
            state = EDDPotentials.BuilderState(
                seedfile=["A.cell", "B.cell"],
                seedfile_calc="dummy.cell",
                workdir=temp,
                formula_sampling_enabled=true,
            )
            @test_throws ErrorException EDDPotentials._formula_sampling_context(
                state,
                ["Li", "Ta", "O", "Cl"],
            )
        end
    end
end
