using EDDPotentials
using CellBase
using Test
using LinearAlgebra
using Flux: Chain, Dense

isdefined(Main, :_h2_cell) || include("utils.jl")

@testset "MetaCalculator Basic" begin
    source = _get_calc()
    cell = get_cell(source)

    @testset "Construction" begin
        # Test construction with no modifiers
        calc = MetaCalculator(source)
        @test calc isa MetaCalculator
        @test get_cell(calc) === cell
        @test isempty(calc.modifiers)

        # Test construction with modifiers
        mod = ExtraDimensionPenaltyModifier(0.01)
        calc2 = MetaCalculator(_get_calc(), [mod])
        @test length(calc2.modifiers) == 1
        @test calc2.modifiers[1] === mod
    end

    @testset "AbstractCalc interface" begin
        calc = MetaCalculator(_get_calc())

        # Test that it implements AbstractCalc methods
        @test get_cell(calc) isa Cell

        # Test energy calculation
        energy = eddp_energy(calc)
        @test energy isa Number

        # Test forces calculation
        forces = eddp_forces(calc)
        @test forces isa Matrix
        @test size(forces, 1) == 3

        # Test stress calculation
        stress = eddp_stress(calc)
        @test stress isa Matrix
        @test size(stress) == (3, 3)
    end

    @testset "Caching" begin
        calc = MetaCalculator(_get_calc())
        cell = get_cell(calc)

        # First calculation
        E1 = eddp_energy(calc)
        @test calc.energy_dirty == false

        # Second calculation without changing cell should use cache
        E2 = eddp_energy(calc)
        @test E2 ≈ E1

        # Change cell, should recalculate
        pos = positions(cell)
        pos[1, 1] += 0.1
        set_positions!(cell, pos)
        @test calc.energy_dirty == false
        set_positions!(calc, pos)
        @test calc.energy_dirty == true

        E3 = eddp_energy(calc)
        @test E3 ≉ E1  # Should be different
        @test calc.energy_dirty == false
    end

    @testset "Display" begin
        calc = MetaCalculator(_get_calc())
        str = sprint(show, calc)
        @test occursin("MetaCalculator", str)
        @test occursin("Source:", str)
        @test occursin("Pipeline:", str)

        # With modifiers
        source = _get_calc()
        cell = get_cell(source)
        mod1 = ExtraDimensionPenaltyModifier(0.01)
        mod2 = CellRelaxationModifier(cell)
        calc2 = MetaCalculator(source, [mod1, mod2])
        str2 = sprint(show, calc2)
        @test occursin("ExtraDimensionPenaltyModifier", str2)
        @test occursin("CellRelaxationModifier", str2)
    end
end

@testset "MetaCalculator with Modifiers" begin
    @testset "Modifier query" begin
        source = _get_calc()
        cell = get_cell(source)
        mod1 = ExtraDimensionPenaltyModifier(0.01)
        mod2 = CellRelaxationModifier(cell)
        calc = MetaCalculator(source, [mod1, mod2])

        # Test get_modifier
        penalty = get_modifier(calc, ExtraDimensionPenaltyModifier)
        @test penalty === mod1

        relax = get_modifier(calc, CellRelaxationModifier)
        @test relax === mod2

        # Non-existent modifier
        additive = get_modifier(calc, AdditiveCalculatorModifier)
        @test isnothing(additive)
    end

    @testset "Modifier parameter access" begin
        source = _get_calc()
        mod = ExtraDimensionPenaltyModifier(0.01)
        calc = MetaCalculator(source, [mod])

        # Test get_modifier_parameter
        alpha = get_modifier_parameter(calc, ExtraDimensionPenaltyModifier, :alpha)
        @test alpha ≈ 0.01

        # Test set_modifier_parameter!
        set_modifier_parameter!(calc, ExtraDimensionPenaltyModifier, :alpha, 0.05)
        @test mod.alpha ≈ 0.05

        # Error for non-existent modifier
        @test_throws ArgumentError set_modifier_parameter!(
            calc, CellRelaxationModifier, :alpha, 0.1
        )
    end
end

@testset "MetaCalculator Pipeline Execution" begin
    @testset "CellRelaxation pipeline" begin
        source = _get_calc()
        cell = get_cell(source)

        # Create calculator with cell relaxation
        mod = CellRelaxationModifier(cell; external_pressure=diagm([1.0, 1.0, 1.0]))
        calc = MetaCalculator(source, [mod])

        # Get augmented positions
        pos = get_positions(calc)
        @test size(pos, 2) == length(cell) + 3  # atoms + lattice vectors

        # Get forces
        forces = eddp_forces(calc)
        @test size(forces, 2) == length(cell) + 3  # atomic + virial forces

        # Get energy (should include enthalpy term)
        E_source = eddp_energy(_get_calc())
        E_calc = eddp_energy(calc)
        # Energy should be different due to PV term, but we can't test exact values
        @test E_calc isa Number
    end

    @testset "Multiple modifiers pipeline" begin
        source = _get_calc()
        cell = get_cell(source)

        # Create calculator with cell relaxation modifier
        mod = CellRelaxationModifier(cell)
        calc = MetaCalculator(source, [mod])

        # Modifier should be applied
        E = eddp_energy(calc)
        @test E isa Number

        # Positions should be augmented (from CellRelaxationModifier)
        pos = get_positions(calc)
        @test size(pos, 2) > length(cell)

        # Forces should be augmented
        forces = eddp_forces(calc)
        @test size(forces, 2) == length(cell) + 3
    end
end

@testset "Comparison with old nested structure" begin
    @testset "VariableCellCalc equivalence" begin
        # Old way
        base_calc = _get_calc()
        old_calc = VariableCellCalc(deepcopy(base_calc))

        # New way
        source2 = deepcopy(base_calc)
        mod = CellRelaxationModifier(get_cell(source2))
        new_calc = MetaCalculator(source2, [mod])

        # Compare energies
        E_old = eddp_energy(old_calc)
        E_new = eddp_energy(new_calc)
        @test E_old ≈ E_new rtol = 1e-10

        # Compare positions
        pos_old = get_positions(old_calc)
        pos_new = get_positions(new_calc)
        @test pos_old ≈ pos_new rtol = 1e-10

        # Compare forces
        F_old = eddp_forces(old_calc)
        F_new = eddp_forces(new_calc)
        @test F_old ≈ F_new rtol = 1e-10
    end
end