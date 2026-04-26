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
    @test CellRelaxationModifier <: AbstractModifier
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

@testset "CellRelaxationModifier" begin
    # Create a test cell
    cell = Cell(Lattice(5.0, 5.0, 5.0), [:H, :H], Float64[
        0.0 2.5
        0.0 2.5
        0.0 2.5
    ])

    @testset "Construction" begin
        # Test basic construction
        mod = CellRelaxationModifier(cell)
        @test mod.lattice_factor ≈ 2.0  # 2 atoms
        @test size(mod.external_pressure) == (3, 3)
        @test all(mod.external_pressure .≈ 0.0)

        # Test with external pressure
        P = diagm([1.0, 1.0, 1.0])
        mod2 = CellRelaxationModifier(cell; external_pressure=P)
        @test mod2.external_pressure ≈ P
    end

    @testset "Modifier properties" begin
        mod = CellRelaxationModifier(cell)
        @test requires_forces(mod) == true
        @test modifies_positions(mod) == true
    end

    @testset "Deformation gradient" begin
        mod = CellRelaxationModifier(cell)

        # Initially, deformation gradient should be identity
        dgrad = deformgradient(mod, cell)
        @test dgrad ≈ I(3) atol = 1e-10

        # Deform the cell
        new_cellmat = cellmat(cell) * 1.1  # 10% expansion
        set_cellmat!(cell, new_cellmat; scale_positions=false)

        dgrad2 = deformgradient(mod, cell)
        @test dgrad2 ≈ 1.1 * I(3) atol = 1e-10
    end

    @testset "Augmented positions" begin
        mod = CellRelaxationModifier(cell)

        # Get augmented positions
        aug_pos = get_augmented_positions(mod, cell)

        # Should have 2 atomic positions + 3 lattice positions
        @test size(aug_pos) == (3, 5)

        # First 2 columns are atomic positions (transformed by deformation gradient)
        # Last 3 columns are lattice positions (scaled deformation gradient)

        # For identity deformation, atomic positions should be unchanged
        @test aug_pos[1:3, 1:2] ≈ positions(cell)

        # Lattice positions should be scaled identity matrix * lattice_factor
        expected_lpos = mod.lattice_factor * I(3)
        @test aug_pos[1:3, 3:5] ≈ expected_lpos
    end

    @testset "Set augmented positions" begin
        mod = CellRelaxationModifier(cell)
        orig_cellmat = copy(cellmat(cell))

        # Get current augmented positions
        aug_pos = get_augmented_positions(mod, cell)

        # Modify lattice part (scale by 1.05)
        aug_pos[:, 3:5] .*= 1.05

        # Set back
        set_augmented_positions!(mod, cell, aug_pos)

        # Cell should be scaled
        @test cellmat(cell) ≈ orig_cellmat * 1.05 atol = 1e-10
    end

    @testset "post_calculate! with pressure" begin
        mod = CellRelaxationModifier(cell; external_pressure=diagm([1.0, 1.0, 1.0]))

        # Mock source results
        energy0 = 10.0
        forces0 = ones(3, 2) * 0.1
        stress0 = diagm([2.0, 2.0, 2.0])

        # Apply modifier
        pre_calculate!(mod, cell)  # Compute deformation gradient
        energy, forces, stress = post_calculate!(mod, cell, energy0, forces0, stress0)

        # Energy should include PV term: E + P*V
        vol = volume(cell)
        expected_energy = energy0 + vol * tr(mod.external_pressure) / 3
        @test energy ≈ expected_energy

        # Forces should be augmented [atomic_forces, virial_forces]
        @test size(forces) == (3, 5)  # 2 atoms + 3 lattice vectors

        # Stress unchanged (source stress returned)
        @test stress ≈ stress0
    end

    @testset "Display" begin
        # Test display with hydrostatic pressure
        mod = CellRelaxationModifier(cell; external_pressure=diagm([3.0, 3.0, 3.0]))
        str = sprint(show, mod)
        @test occursin("CellRelaxationModifier", str)
        @test occursin("3.0", str) || occursin("P=3.0", str)

        # Test display with non-diagonal pressure
        mod2 = CellRelaxationModifier(cell; external_pressure=rand(3, 3))
        str2 = sprint(show, mod2)
        @test occursin("CellRelaxationModifier", str2)
        @test occursin("3×3 Matrix", str2)
    end
end

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
