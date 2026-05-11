using EDDPotentials
using CellBase
using Test

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
        #SLACK=0.25
        """
        out = EDDPotentials._inject_formula_directive(seed, "SiO2")
        @test occursin(r"(?m)^#FORMULA=SiO2$", out)
        @test !occursin(r"(?m)^\\s*#SPECIES\\s*=", out)
        @test !occursin(r"(?m)^\\s*#NATOM\\s*=", out)
        @test !occursin(r"(?m)^\\s*#FORMULA\\s*=SiO$", out)
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
                max_reduced_formula_coeff=3,
                require_charge_neutral=true,
                oxidation_states=Dict(
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
