using EDDPotentials
using CellBase
using Test
using LinearAlgebra
using EDDPotentials.Potential: set_parameter!, get_parameter, post_calculate!, requires_forces, pre_calculate!

@testset "Modifier Interface" begin
    # Test abstract modifier interface
    @test AbstractModifier isa Type

    # Test that concrete modifiers are subtypes
    @test ExtraDimensionPenaltyModifier <: AbstractModifier
    @test AdditiveCalculatorModifier <: AbstractModifier
end

@testset "ExtraDimensionPenaltyModifier" begin
    # Create a 4D cell for testing
    cell = Cell(Lattice(5.0, 5.0, 5.0), [:H, :H], Float64[
        0.0 1.0
        0.0 2.0
        0.0 0.5
        0.1 0.2  # Extra dimension
    ])

    @testset "Construction" begin
        # Test basic construction
        mod = ExtraDimensionPenaltyModifier(0.01)
        @test mod.alpha ≈ 0.01
        @test mod.start_dim == 4

        # Test custom start_dim
        mod2 = ExtraDimensionPenaltyModifier(0.02; start_dim=3)
        @test mod2.alpha ≈ 0.02
        @test mod2.start_dim == 3
    end

    @testset "Modifier properties" begin
        mod = ExtraDimensionPenaltyModifier(0.01)
        @test requires_forces(mod) == true
    end

    @testset "post_calculate!" begin
        mod = ExtraDimensionPenaltyModifier(0.1)
        pos = positions(cell)

        # Initial values
        energy0 = 1.0
        forces0 = zeros(4, 2)
        stress0 = zeros(3, 3)

        # Apply modifier
        energy, forces, stress = post_calculate!(mod, cell, energy0, forces0, stress0)

        # Check energy penalty: E' = E + alpha * sum(x_4^2)
        expected_penalty = 0.1 * (0.1^2 + 0.2^2)
        @test energy ≈ energy0 + expected_penalty

        # Check forces: F'_4 = F_4 - 2*alpha*x_4
        @test forces[1:3, :] ≈ forces0[1:3, :]  # First 3 dims unchanged
        @test forces[4, 1] ≈ -2 * 0.1 * 0.1
        @test forces[4, 2] ≈ -2 * 0.1 * 0.2

        # Stress unchanged
        @test stress ≈ stress0
    end

    @testset "Parameter access" begin
        mod = ExtraDimensionPenaltyModifier(0.01)

        # Test get_parameter
        @test get_parameter(mod, :alpha) ≈ 0.01
        @test get_parameter(mod, :start_dim) == 4

        # Test set_parameter!
        set_parameter!(mod, :alpha, 0.05)
        @test mod.alpha ≈ 0.05
    end

    @testset "Display" begin
        mod = ExtraDimensionPenaltyModifier(0.001; start_dim=4)
        str = sprint(show, mod)
        @test occursin("ExtraDimensionPenaltyModifier", str)
        @test occursin("0.001", str)
        @test occursin("4", str)
    end
end

@testset "VolumeConstraintModifier.post_calculate!" begin
        cell = Cell(Lattice(4,4,4), [:H,:H], [0 1.0; 0 0.0; 0 0])
        V0 = CellBase.volume(cell)
        mod = VolumeConstraintModifier(cell; k=2.0)

        e, f, s = post_calculate!(mod, cell, 0.0, zeros(3,2), zeros(3,3))
        @test e ≈ 0.0 atol=1e-12
        @test s ≈ zeros(3,3) atol=1e-12

        cell2 = Cell(Lattice(5,4,4), [:H,:H], [0 1.0; 0 0.0; 0 0])
        V1 = CellBase.volume(cell2)
        e2, f2, s2 = post_calculate!(mod, cell2, 0.0, zeros(3,2), zeros(3,3))
        @test e2 ≈ 0.5 * 2.0 * (V1 - V0)^2 / V0
        @test f2 ≈ zeros(3,2)
        p_expected = 2.0 * (V1 - V0) / V0
        @test s2 ≈ diagm([p_expected, p_expected, p_expected])
    end

    @testset "AdditiveCalculatorModifier.post_calculate!" begin
        using EDDPotentials.Potential: PairpotCalc, PairpotWorkspace, build_lj_spec
        cell = Cell(Lattice(15,15,15), [:Ar,:Ar], [0 4.0; 0 0.0; 0 0])
        spec = build_lj_spec(((:Ar,:Ar), 3.4); epsilon=0.01, cutoff=12.0)
        pair_calc = PairpotCalc(cell, PairpotWorkspace(cell, spec); nl=NeighbourList(cell, 12.0; savevec=true))
        add_mod = AdditiveCalculatorModifier(pair_calc)

        add_e = eddp_energy(pair_calc)
        add_f = copy(eddp_forces(pair_calc))
        add_s = copy(eddp_stress(pair_calc))

        e, f, s = post_calculate!(add_mod, cell, 1.0, ones(3,2), zeros(3,3))
        @test e ≈ 1.0 + add_e
        @test f ≈ ones(3,2) .+ add_f
        @test s ≈ add_s
    end

@testset "Modifier parameter access" begin
    mod_penalty = ExtraDimensionPenaltyModifier(0.01)
    mod_vol = VolumeConstraintModifier(100.0, 10.0)

    @testset "get_parameter" begin
        @test get_parameter(mod_penalty, :alpha) ≈ 0.01
        @test get_parameter(mod_vol, :k) ≈ 10.0
    end

    @testset "set_parameter!" begin
        set_parameter!(mod_penalty, :alpha, 0.05)
        @test mod_penalty.alpha ≈ 0.05

        set_parameter!(mod_vol, :k, 20.0)
        @test mod_vol.k ≈ 20.0
    end
end
