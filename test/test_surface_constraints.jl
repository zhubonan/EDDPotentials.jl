using Test
using EDDPotentials
using EDDPotentials.Workflow: _generate_random_structures
using CellBase
using LinearAlgebra
import EDDPotentials.Potential: get_cell, eddp_energy, eddp_forces, eddp_stress, calculate!

mutable struct SurfaceToyCalc <: AbstractCalc
    cell::Cell
    reference::Matrix{Float64}
end

get_cell(calc::SurfaceToyCalc) = calc.cell
eddp_energy(calc::SurfaceToyCalc; kwargs...) = 0.5 * sum(abs2, positions(calc.cell) .- calc.reference)
eddp_forces(calc::SurfaceToyCalc; kwargs...) = -(positions(calc.cell) .- calc.reference)
eddp_stress(calc::SurfaceToyCalc; kwargs...) = Matrix{Float64}(I, 3, 3)
calculate!(calc::SurfaceToyCalc; kwargs...) = calc

function _surface_cell_lines()
    [
        "%BLOCK LATTICE_CART",
        "6 0 0",
        "0 6 0",
        "0 0 10",
        "%ENDBLOCK LATTICE_CART",
        "%BLOCK POSITIONS_FRAC",
        "Ar 0.25 0.25 0.20 # bulk1 % POSAMP=0 FIX",
        "Ar 0.75 0.25 0.20 # bulk2 % POSAMP=0 NOMOVE",
        "Ar 0.50 0.50 0.60 # surf",
        "%ENDBLOCK POSITIONS_FRAC",
        "FIX_ALL_CELL : true",
        "FIX_VOL : true",
        "%BLOCK IONIC_CONSTRAINTS",
        "1 Ar 3 1.0 0.0 0.0",
        "%ENDBLOCK IONIC_CONSTRAINTS",
    ]
end

function _with_fake_surface_buildcell(f)
    mktempdir() do temp
        bindir = joinpath(temp, "bin")
        mkdir(bindir)
        output = joinpath(temp, "surface-output.cell")
        open(output, "w") do io
            for line in _surface_cell_lines()
                println(io, line)
            end
        end

        timeout_script = joinpath(bindir, "timeout")
        open(timeout_script, "w") do io
            println(io, "#!/bin/sh")
            println(io, "shift")
            println(io, "exec \"\$@\"")
        end
        chmod(timeout_script, 0o755)

        buildcell_script = joinpath(bindir, "buildcell")
        open(buildcell_script, "w") do io
            println(io, "#!/bin/sh")
            println(io, "cat >/dev/null")
            println(io, "cat '$(output)'")
        end
        chmod(buildcell_script, 0o755)

        old_path = get(ENV, "PATH", "")
        ENV["PATH"] = bindir * ":" * old_path
        try
            f(temp)
        finally
            ENV["PATH"] = old_path
        end
    end
end

@testset "Surface buildcell constraints" begin
    @testset "parse buildcell constraints" begin
        cell = EDDPotentials.Search.read_buildcell_cell(_surface_cell_lines())
        fix = cell.arrays[:buildcell_fix_mask]
        @test size(fix) == (3, 3)
        @test all(fix[:, 1])
        @test !any(fix[:, 2])
        @test fix[:, 3] == [true, false, false]
        @test cell.arrays[:buildcell_nomove_mask] == [false, true, false]

        info = cell.metadata[:buildcell_cell_constraints]
        @test info[:fix_all_cell] === true
        @test info[:fix_volume] === true
        @test info[:cell_mask] === nothing
    end

    @testset "convert parsed constraints" begin
        cell = EDDPotentials.Search.read_buildcell_cell(_surface_cell_lines())
        constraints = EDDPotentials.Simulation._cell_step_constraints(cell)
        @test any(c -> c isa FixAtoms && c.atoms == [1], constraints)
        @test any(c -> c isa FixCartesian && c.atoms == [3] && c.mask == (true, false, false), constraints)
        @test any(c -> c isa FixCellConstraint && all(c.mask), constraints)
        @test any(c -> c isa FixVolumeConstraint, constraints)
    end

    @testset "CELL_CONSTRAINTS block" begin
        lines = [
            "%BLOCK LATTICE_CART",
            "4 0 0",
            "0 4 0",
            "0 0 4",
            "%ENDBLOCK LATTICE_CART",
            "%BLOCK POSITIONS_FRAC",
            "Ar 0 0 0",
            "%ENDBLOCK POSITIONS_FRAC",
            "%BLOCK CELL_CONSTRAINTS",
            "1 2 0 0 0 3",
            "%ENDBLOCK CELL_CONSTRAINTS",
        ]
        cell = EDDPotentials.Search.read_buildcell_cell(lines)
        constraints = EDDPotentials.Simulation._cell_step_constraints(cell)
        cell_constraint = only(filter(c -> c isa FixCellConstraint, constraints))
        @test cell_constraint.mask == (false, false, true, true, true, false)
    end

    @testset "relax honors parsed fixed atoms" begin
        cell = EDDPotentials.Search.read_buildcell_cell(_surface_cell_lines())
        pos0 = copy(positions(cell))
        target = copy(pos0)
        target[:, 1] .+= [0.6, 0.2, 0.1]
        target[:, 2] .+= [0.1, 0.3, 0.2]
        target[:, 3] .+= [0.0, 0.4, 0.2]
        calc = SurfaceToyCalc(cell, target)

        result = relax!(calc;
            multi=false,
            relax_cell=true,
            method="tpsd",
            iterations=20,
            maxstep=0.1,
            force_threshold=1e-8,
            symmetry="off",
        )
        pos1 = positions(cell)
        @test pos1[:, 1] ≈ pos0[:, 1]
        @test pos1[1, 3] ≈ pos0[1, 3]
        @test norm(pos1[2:3, 3] - pos0[2:3, 3]) > 0
        @test cellmat(cell) ≈ [6.0 0.0 0.0; 0.0 6.0 0.0; 0.0 0.0 10.0]
        @test result.iterations > 0

        cell2 = EDDPotentials.Search.read_buildcell_cell(_surface_cell_lines())
        pos2 = copy(positions(cell2))
        target2 = copy(pos2)
        target2[:, 1] .+= [0.4, 0.0, 0.0]
        calc2 = SurfaceToyCalc(cell2, target2)
        relax!(calc2;
            multi=false,
            relax_cell=false,
            method="tpsd",
            iterations=3,
            maxstep=0.1,
            symmetry="off",
            honor_cell_constraints=false,
        )
        @test norm(positions(cell2)[:, 1] - pos2[:, 1]) > 0
    end

    @testset "fake buildcell preserves constraints" begin
        _with_fake_surface_buildcell() do temp
            seed = joinpath(temp, "Ar-surface.cell")
            write(seed, "#SURFACE\n")
            cell = build_one(seed; timeout=5)
            @test haskey(cell.arrays, :buildcell_fix_mask)
            @test get(cell.metadata[:buildcell_cell_constraints], :fix_all_cell, false) === true
        end
    end

    @testset "RSS path preserves parsed constraints on relaxed result" begin
        _with_fake_surface_buildcell() do temp
            seed = joinpath(temp, "Ar-surface.cell")
            write(seed, "#SURFACE\n")
            cell0 = EDDPotentials.Search.read_buildcell_cell(_surface_cell_lines())
            calc0 = lj_baseline_calc(cell0; sigma=3.4, epsilon=0.01, rcut=8.0)
            res = build_and_relax(
                seed,
                calc0.nninterface,
                calc0.cf;
                relax_option=RelaxOption(
                    method="tpsd",
                    relax_cell=true,
                    iterations=2,
                    maxstep=0.05,
                    symmetry="off",
                ),
                nmax=100,
                core_size=0.0,
            )
            cell = get_cell(res.relax.calc)
            @test haskey(cell.arrays, :buildcell_fix_mask)
            @test get(cell.metadata[:buildcell_cell_constraints], :fix_all_cell, false) === true
            @test cellmat(cell) ≈ [6.0 0.0 0.0; 0.0 6.0 0.0; 0.0 0.0 10.0]
        end
    end

    @testset "Builder constrained surface seed smoke" begin
        _with_fake_surface_buildcell() do temp
            seed = joinpath(temp, "Ar-surface.cell")
            write(seed, "#SURFACE\n")
            state = EDDPotentials.BuilderState(
                seedfile=basename(seed),
                seedfile_calc=basename(seed),
                workdir=temp,
                n_initial=1,
                per_generation=1,
                dft_mode="acrud",
                dft_kwargs=Dict{String,Any}(
                    "ap_cmd" => joinpath(temp, "bin", "ap"),
                    "code" => "ml",
                    "calculator" => "MatterSim",
                    "singlepoint" => true,
                ),
            )
            state.build_timeout = 5.0
            builder = Builder(
                EDDPotentials.BuilderOption(
                    state,
                    EDDPotentials.CellFeatureConfig(elements=["Ar"], rcut2=8.0, rcut3=8.0, p2=[6, 12, 2], p3=[], q3=[]),
                    nothing,
                    EDDPotentials.RssSetting(),
                    EDDPotentials.TrainingOption(),
                    nothing,
                ),
            )
            builder.state.builder_file_path = joinpath(temp, "link.toml")

            ap = joinpath(temp, "bin", "ap")
            open(ap, "w") do io
                println(io, "#!/bin/sh")
                println(io, "exit 0")
            end
            chmod(ap, 0o755)

            _generate_random_structures(builder, 0)
            outdir = joinpath(temp, "gen0")
            @test count(endswith(".res"), readdir(outdir)) == 1

            cmd = EDDPotentials.airsspy_crud_cmd(
                joinpath(temp, "acrud-work");
                ap_cmd=ap,
                code="ml",
                calculator="MatterSim",
                singlepoint=true,
            )
            @test cmd.exec[2:3] == ["run", "crud"]
            @test "--calculator" in cmd.exec
            @test "MatterSim" in cmd.exec
        end
    end
end
