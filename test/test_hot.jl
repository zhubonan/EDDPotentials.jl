#=
Tests for the hot-AIRSS protocol
=#
using EDDPotentials
using EDDPotentials:
    HotAIRSSOption,
    HotAIRSSResult,
    lj_baseline_calc,
    eddp_energy
using CellBase
using Test
using Unitful: ustrip

# ─── Helper: create a simple LJ calculator ───

function _make_lj_calc(; sigma=3.4, epsilon=0.01, rcut=8.0)
    cell = Cell(
        Lattice(10.0, 10.0, 10.0),
        [:Ar, :Ar],
        [[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]],
    )
    lj_baseline_calc(cell; sigma, epsilon, rcut)
end

function _make_lj_calc_displaced(; sigma=3.4, epsilon=0.01, rcut=8.0)
    cell = Cell(
        Lattice(10.0, 10.0, 10.0),
        [:Ar, :Ar],
        [[1.0, 1.0, 1.0], [6.0, 6.0, 6.0]],
    )
    lj_baseline_calc(cell; sigma, epsilon, rcut)
end

# ─── Tests ───

@testset "Hot-AIRSS" begin

    @testset "Option defaults" begin
        opts = HotAIRSSOption()
        @test opts.cycles == 10
        @test opts.relax_cell == true
        @test opts.md_steps == 50
        @test opts.md_temperature == 1000.0
        @test opts.md_integrator == "langevin"
        @test opts.relax_method == "tpsd"
    end

    @testset "Option from kwargs" begin
        opts = HotAIRSSOption(; cycles=3, md_temperature=500.0, relax_cell=false)
        @test opts.cycles == 3
        @test opts.md_temperature == 500.0
        @test opts.relax_cell == false
    end

    @testset "Basic run" begin
        calc = _make_lj_calc()
        result = hot_airss!(calc;
            cycles=2,
            relax_itermax=10,
            md_steps=5,
            md_temperature=100.0,
            md_seed=42,
        )

        @test result isa HotAIRSSResult
        @test length(result.energy_history) == 2 * 2 + 1  # 2 cycles + final relax
        @test length(result.relax_iterations) == 2 + 1     # 2 cycles + final relax
        @test result.cycles_completed == 2
        @test isfinite(result.final_energy)
        @test isfinite(result.final_fmax)
        @test isfinite(result.final_smax)
    end

    @testset "Energy history pattern" begin
        calc = _make_lj_calc()
        result = hot_airss!(calc;
            cycles=3,
            relax_itermax=20,
            md_steps=10,
            md_temperature=500.0,
            md_seed=42,
        )

        # Should have 2*cycles + 1 entries: [R, M, R, M, R, M, R]
        @test length(result.energy_history) == 7

        # All energies should be finite
        @test all(isfinite, result.energy_history)

        # The final relaxation energy should be the last entry
        @test result.energy_history[end] == result.final_energy
    end

    @testset "Variable cell wrapping" begin
        calc = _make_lj_calc()

        # With relax_cell=true (default), VariableCellCalc should be used
        result = hot_airss!(calc;
            cycles=1,
            relax_itermax=5,
            md_steps=3,
            md_temperature=100.0,
            relax_cell=true,
            md_seed=42,
        )
        @test result isa HotAIRSSResult
        @test isfinite(result.final_energy)
        @test result.final_smax >= 0.0  # smax computed when relax_cell=true
    end

    @testset "Fixed cell" begin
        calc = _make_lj_calc()

        result = hot_airss!(calc;
            cycles=1,
            relax_itermax=5,
            md_steps=3,
            md_temperature=100.0,
            relax_cell=false,
            md_seed=42,
        )
        @test result isa HotAIRSSResult
        @test isfinite(result.final_energy)
        @test result.final_smax == 0.0  # smax is 0 when fixed cell
    end

    @testset "Langevin integrator" begin
        calc = _make_lj_calc()
        result = hot_airss!(calc;
            cycles=1,
            relax_itermax=5,
            md_steps=5,
            md_temperature=200.0,
            md_integrator="langevin",
            md_friction=0.01,
            md_seed=123,
        )
        @test isfinite(result.final_energy)
    end

    @testset "Berendsen integrator" begin
        calc = _make_lj_calc()
        result = hot_airss!(calc;
            cycles=1,
            relax_itermax=5,
            md_steps=5,
            md_temperature=200.0,
            md_integrator="berendsen",
            md_taut=50.0,
            md_seed=123,
        )
        @test isfinite(result.final_energy)
    end

    @testset "NVE integrator" begin
        calc = _make_lj_calc()
        result = hot_airss!(calc;
            cycles=1,
            relax_itermax=5,
            md_steps=5,
            md_temperature=200.0,
            md_integrator="nve",
        )
        @test isfinite(result.final_energy)
    end

    @testset "Unknown integrator throws" begin
        calc = _make_lj_calc()
        @test_throws ArgumentError hot_airss!(calc;
            cycles=1,
            md_integrator="unknown",
        )
    end

    @testset "Unknown relax method throws" begin
        calc = _make_lj_calc()
        @test_throws ArgumentError hot_airss!(calc;
            cycles=1,
            relax_method="unknown",
        )
    end

    @testset "Deterministic relaxation" begin
        # The relaxation phase is deterministic (TPSD has no randomness).
        # Verify that identical starting configurations give identical results.
        function _run_once()
            calc = _make_lj_calc()
            hot_airss!(calc;
                cycles=1,
                relax_itermax=10,
                md_steps=0,
                md_seed=42,
            )
        end

        r1 = _run_once()
        r2 = _run_once()
        # With md_steps=0, the energy history is [relax, relax_final] — fully deterministic
        @test r1.energy_history ≈ r2.energy_history
    end

    @testset "Callback" begin
        calc = _make_lj_calc()
        phases = Symbol[]
        energies = Float64[]

        hot_airss!(calc;
            cycles=2,
            relax_itermax=5,
            md_steps=3,
            md_temperature=100.0,
            md_seed=42,
            callback=info -> begin
                push!(phases, info.phase)
                push!(energies, info.energy)
            end,
        )

        # Should get: relax, md, relax, md, final
        @test phases == [:relax, :md, :relax, :md, :final]
        @test length(energies) == 5
        @test all(isfinite, energies)
    end

    @testset "Verbose output" begin
        calc = _make_lj_calc()
        # Just make sure it doesn't error with verbose=true
        result = hot_airss!(calc;
            cycles=1,
            relax_itermax=3,
            md_steps=2,
            md_temperature=100.0,
            verbose=true,
            md_seed=42,
        )
        @test isfinite(result.final_energy)
    end

    @testset "Convergence reduces energy" begin
        # Start from a displaced configuration
        calc = _make_lj_calc_displaced()
        e_start = eddp_energy(calc)

        result = hot_airss!(calc;
            cycles=1,
            relax_itermax=50,
            md_steps=0,
            md_seed=42,
        )
        # The final relaxed energy should be lower than the starting energy
        @test result.final_energy <= e_start
    end
end
