using EDDPotentials
using CellBase
using Test
using LinearAlgebra
using EDDPotentials.EDDPotentialsCore: set_parameter!, get_parameter, post_calculate!, modifies_positions, requires_forces, pre_calculate!, deformgradient

@testset "Modifier Interface" begin
    # Test abstract modifier interface
    @test AbstractModifier isa Type

    # Test that concrete modifiers are subtypes
    @test ExtraDimensionPenaltyModifier <: AbstractModifier
    # CellRelaxationModifier removed
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
        @test modifies_positions(mod) == false
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

# removed @testset "CellRelaxationModifier"
  # end removed

@testset "Modifier parameter access" begin
    mod_penalty = ExtraDimensionPenaltyModifier(0.01)
    cell = Cell(Lattice(5.0, 5.0, 5.0), [:H], [[2.5, 2.5, 2.5]])
    mod_relax = CellRelaxationModifier(cell)

    @testset "get_parameter" begin
        @test get_parameter(mod_penalty, :alpha) ≈ 0.01
        @test get_parameter(mod_relax, :lattice_factor) ≈ 1.0
    end

    @testset "set_parameter!" begin
        set_parameter!(mod_penalty, :alpha, 0.05)
        @test mod_penalty.alpha ≈ 0.05

        set_parameter!(mod_relax, :lattice_factor, 2.0)
        @test mod_relax.lattice_factor ≈ 2.0
    end
end
