using Test
using EDDPotentials
using CellBase
using LinearAlgebra
using Configurations
import EDDPotentials.Potential: get_cell, eddp_energy, eddp_forces, eddp_stress, calculate!

isdefined(Main, :_h2_cell) || include("utils.jl")

mutable struct SymmetryDummyCalc <: AbstractCalc
    cell::Cell
end

get_cell(calc::SymmetryDummyCalc) = calc.cell
eddp_energy(calc::SymmetryDummyCalc; kwargs...) = 0.0
eddp_forces(calc::SymmetryDummyCalc; kwargs...) = zeros(3, length(calc.cell))
eddp_stress(calc::SymmetryDummyCalc; kwargs...) = zeros(3, 3)
calculate!(calc::SymmetryDummyCalc; kwargs...) = calc

function _diamond_cell()
    a = 3.57
    frac = [
        0.0 0.25
        0.0 0.25
        0.0 0.25
    ]
    Cell(Lattice(a, a, a), [:C, :C], a .* frac)
end

function _with_fake_buildcell(f)
    mktempdir() do temp
        bindir = joinpath(temp, "bin")
        mkdir(bindir)
        script = joinpath(bindir, "buildcell")
        open(script, "w") do io
            println(io, "#!/bin/sh")
            println(io, "cat >/dev/null")
            println(io, "cat <<'EOF'")
            println(io, "%BLOCK LATTICE_CART")
            println(io, "3 0 0")
            println(io, "0 3 0")
            println(io, "0 0 3")
            println(io, "%ENDBLOCK LATTICE_CART")
            println(io, "%BLOCK POSITIONS_FRAC")
            println(io, "C 0 0 0")
            println(io, "%ENDBLOCK POSITIONS_FRAC")
            println(io, "EOF")
        end
        chmod(script, 0o755)
        old_path = get(ENV, "PATH", "")
        ENV["PATH"] = bindir * ":" * old_path
        try
            f(temp)
        finally
            ENV["PATH"] = old_path
        end
    end
end

function _is_symmetry_invariant_vectors(constraint, calc, V; atol=1e-10)
    rotations = EDDPotentials.Simulation._cartesian_rotations(constraint, calc)
    for atom in axes(V, 2), isym in eachindex(rotations)
        source = constraint.mappings[isym, atom]
        if !isapprox(V[1:3, atom], rotations[isym] * V[1:3, source]; atol)
            return false
        end
    end
    true
end

function _is_symmetry_invariant_matrix(constraint, calc, S; atol=1e-10)
    rotations = EDDPotentials.Simulation._cartesian_rotations(constraint, calc)
    all(R -> isapprox(S, R * S * transpose(R); atol), rotations)
end

@testset "Symmetry-aware relaxation" begin
    @testset "options and config" begin
        @test RelaxOption().symmetry == "auto"
        @test RelaxOption().convergence == "step_energy"
        @test RelaxOption().energy_threshold ≈ 1e-6
        @test RelaxOption().symmetry_tolerance ≈ 1e-5
        @test RelaxOption().max_cell_strain ≈ 0.0125
        @test RelaxOption().cell_damping ≈ 1.0
        @test RelaxOption().stress_threshold_gpa ≈ -1.0
        @test RelaxOption().stress_threshold ≈ EDDPotentials.Potential.GPaToeVAng(0.1)
        @test RelaxOption().external_pressure_gpa === nothing
        @test EDDPotentials.Simulation._stress_threshold_internal(RelaxOption()) ≈ EDDPotentials.Potential.GPaToeVAng(0.1)
        @test EDDPotentials.Simulation._external_pressure_internal(RelaxOption()) ≈ [0.0]
        @test EDDPotentials.Simulation._external_pressure_internal(RelaxOption(external_pressure_gpa=100.0)) ≈ [EDDPotentials.Potential.GPaToeVAng(100.0)]
        @test EDDPotentials.Simulation._external_pressure_internal(RelaxOption(external_pressure_gpa=-5.0)) ≈ [EDDPotentials.Potential.GPaToeVAng(-5.0)]
        @test_deprecated EDDPotentials.Simulation._external_pressure_internal(RelaxOption(external_pressure=ones(6))) ≈ ones(6)
        @test_throws ArgumentError EDDPotentials.Simulation._external_pressure_internal(
            RelaxOption(external_pressure=[1.0], external_pressure_gpa=100.0),
        )
        @test_throws ArgumentError EDDPotentials.Simulation._external_pressure_internal(
            RelaxOption(external_pressure=zeros(6), external_pressure_gpa=100.0),
        )
        @test_throws ArgumentError EDDPotentials.Simulation._normalize_symmetry_mode("maybe")
        @test_throws ArgumentError EDDPotentials.Simulation._normalize_relax_convergence_mode("maybe")

        mktempdir() do temp
            path = joinpath(temp, "relax.toml")
            open(path, "w") do io
                println(io, "symmetry = \"on\"")
                println(io, "symmetry_tolerance = 1.0e-4")
                println(io, "max_cell_strain = 0.005")
                println(io, "cell_damping = 0.02")
                println(io, "stress_threshold_gpa = 0.1")
                println(io, "external_pressure_gpa = 100.0")
                println(io, "convergence = \"force\"")
            end
            opts = from_toml(RelaxOption, path)
            @test opts.symmetry == "on"
            @test opts.symmetry_tolerance ≈ 1e-4
            @test opts.max_cell_strain ≈ 0.005
            @test opts.cell_damping ≈ 0.02
            @test opts.stress_threshold_gpa ≈ 0.1
            @test EDDPotentials.Simulation._stress_threshold_internal(opts) ≈ EDDPotentials.Potential.GPaToeVAng(0.1)
            @test opts.external_pressure_gpa ≈ 100.0
            @test EDDPotentials.Simulation._external_pressure_internal(opts) ≈ [EDDPotentials.Potential.GPaToeVAng(100.0)]
            @test opts.convergence == "force"
        end
    end

    @testset "projection" begin
        cell = _diamond_cell()
        calc = SymmetryDummyCalc(cell)
        constraint = SymmetryConstraint(cell; symprec=1e-5)
        @test constraint.active
        @test size(constraint.mappings, 2) == 2

        opt = NativeTPSD(log_level=:minimal)
        F = [
            1.0 0.2
            2.0 -0.4
            3.0 0.7
        ]
        apply_force_constraints!([constraint], opt, calc, F)
        @test _is_symmetry_invariant_vectors(constraint, calc, F)

        opt_cell = NativeTPSD(relax_cell=true, log_level=:minimal)
        S = [
            1.0 0.2 0.3
            0.4 2.0 0.5
            0.6 0.7 4.0
        ]
        apply_stress_constraints!([constraint], opt_cell, calc, S)
        @test _is_symmetry_invariant_matrix(constraint, calc, S)

        opt_pos = NativeTPSD(relax_cell=false, log_level=:minimal)
        pos_old = copy(CellBase.positions(cell))
        pos_new = pos_old .+ [
            0.01 0.02
            0.03 0.04
            0.05 0.06
        ]
        @test EDDPotentials.Simulation.apply_cartesian_position_constraints!([constraint], opt_pos, calc, pos_old, pos_new) === pos_new
        step = pos_new .- pos_old
        @test _is_symmetry_invariant_vectors(constraint, calc, step)

        pos_new_cell = pos_old .+ [
            0.01 0.02
            0.03 0.04
            0.05 0.06
        ]
        @test EDDPotentials.Simulation.apply_cartesian_position_constraints!([constraint], opt_cell, calc, pos_old, pos_new_cell) === pos_new_cell
        @test pos_new_cell ≈ pos_old .+ [
            0.01 0.02
            0.03 0.04
            0.05 0.06
        ]
    end

    @testset "RSS pressure conversion" begin
        @test EDDPotentials.Search._rss_external_pressure_vector(100.0) ≈ [EDDPotentials.Potential.GPaToeVAng(100.0)]
        opts = RelaxOption(external_pressure_gpa=100.0)
        expected = EDDPotentials.Potential.GPaToeVAng(100.0) .* Matrix{Float64}(I, 3, 3)
        @test EDDPotentials.Search._rss_hyper_external_pressure_matrix(opts) ≈ expected

        legacy_opts = RelaxOption(external_pressure=EDDPotentials.Search._rss_external_pressure_vector(100.0))
        @test_deprecated EDDPotentials.Search._rss_hyper_external_pressure_matrix(legacy_opts) ≈ expected

        zero_opts = RelaxOption(external_pressure_gpa=0.0)
        @test EDDPotentials.Search._rss_hyper_external_pressure_matrix(zero_opts) ≈ zeros(3, 3)
        tensile_opts = RelaxOption(external_pressure_gpa=-5.0)
        tensile_expected = EDDPotentials.Potential.GPaToeVAng(-5.0) .* Matrix{Float64}(I, 3, 3)
        @test EDDPotentials.Search._rss_hyper_external_pressure_matrix(tensile_opts) ≈ tensile_expected

        hot_opts = EDDPotentials.Search._rss_hot_airss_option(opts, (; n_rounds=2, md_temperature=500.0))
        @test hot_opts.external_pressure == [0.0]
        @test hot_opts.external_pressure_gpa == opts.external_pressure_gpa
        @test hot_opts.relax_s_tol ≈ EDDPotentials.Simulation._stress_threshold_internal(opts)
        @test hot_opts.relax_s_tol_gpa ≈ opts.stress_threshold_gpa
        @test hot_opts.cycles == 2
        @test hot_opts.md_temperature ≈ 500.0
    end

    @testset "auto mode and buildcell metadata" begin
        cell = _diamond_cell()
        calc = SymmetryDummyCalc(cell)
        auto_constraints = EDDPotentials.Simulation._resolved_step_constraints(calc, RelaxOption(symmetry="auto"))
        @test isempty(auto_constraints)

        cell.metadata[:symmetry_generate] = true
        auto_constraints = EDDPotentials.Simulation._resolved_step_constraints(calc, RelaxOption(symmetry="auto"))
        @test any(c -> c isa SymmetryConstraint && c.active, auto_constraints)

        off_constraints = EDDPotentials.Simulation._resolved_step_constraints(calc, RelaxOption(symmetry="off"))
        @test isempty(off_constraints)

        _with_fake_buildcell() do temp
            seed = joinpath(temp, "C.cell")
            open(seed, "w") do io
                println(io, "SYMMETRY_GENERATE")
            end
            generated = build_one(seed; timeout=5)
            @test get(generated.metadata, :symmetry_generate, false) === true

            transformed = build_one(seed; timeout=5, init_structure_transform=cell -> deepcopy(cell))
            @test get(transformed.metadata, :symmetry_generate, false) === true

            seed2 = joinpath(temp, "C-no-sym.cell")
            write(seed2, "")
            generated2 = build_one(seed2; timeout=5)
            @test get(generated2.metadata, :symmetry_generate, false) === false
        end
    end
end
