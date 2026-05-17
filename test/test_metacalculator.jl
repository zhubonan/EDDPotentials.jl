using EDDPotentials
using CellBase
using Test
using LinearAlgebra

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

    @testset "Direct source cell mutation invalidates cache" begin
        calc = MetaCalculator(_get_calc())
        cell = get_cell(calc)
        E1 = eddp_energy(calc)

        pos = positions(cell)
        pos[1, 1] += 0.25
        set_positions!(cell, pos)

        E2 = eddp_energy(calc)
        @test E2 ≉ E1
    end

    @testset "Display" begin
        calc = MetaCalculator(_get_calc())
        str = sprint(show, calc)
        @test occursin("MetaCalculator", str)
        @test occursin("Source:", str)
        @test occursin("Pipeline:", str)

        # With modifiers
        source = _get_calc()
        mod1 = ExtraDimensionPenaltyModifier(0.01)
        calc2 = MetaCalculator(source, [mod1])
        str2 = sprint(show, calc2)
        @test occursin("ExtraDimensionPenaltyModifier", str2)
    end
end

@testset "MetaCalculator with Modifiers" begin
    @testset "Modifier query" begin
        source = _get_calc()
        mod1 = ExtraDimensionPenaltyModifier(0.01)
        calc = MetaCalculator(source, [mod1])

        # Test get_modifier
        penalty = get_modifier(calc, ExtraDimensionPenaltyModifier)
        @test penalty === mod1

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
            calc, AdditiveCalculatorModifier, :alpha, 0.1
        )
    end
end

@testset "MetaCalculator modifier energy/force composition" begin
    pos_4d = Float64[
        0.0 1.0
        0.0 2.0
        0.0 0.5
        0.1 0.3
    ]
    cell4 = Cell(Lattice(8.0, 8.0, 8.0), [:H, :H], pos_4d)
    cf = CellFeature([:H]; p2=[2.0], p3=Float64[], q3=Float64[])
    model = LinearInterface([0.1, 0.2])
    source = EDDPCalculator(cell4, cf, model; core_repulsion=nothing)

    alpha = 0.5
    mod = ExtraDimensionPenaltyModifier(alpha; start_dim=4)
    mc = MetaCalculator(source, [mod])

    E_src = eddp_energy(source)
    E_meta = eddp_energy(mc)
    expected_penalty = alpha * sum(pos_4d[4, :] .^ 2)
    @test E_meta ≈ E_src + expected_penalty atol = 1e-10

    F_src = eddp_forces(source)
    F_meta = eddp_forces(mc)
    @test F_meta[1:3, :] ≈ F_src[1:3, :] atol = 1e-10
    @test F_meta[4, :] ≈ F_src[4, :] .- 2 * alpha .* pos_4d[4, :] atol = 1e-10

    # Stress should be unchanged by penalty modifier
    S_src = eddp_stress(source)
    S_meta = eddp_stress(mc)
    @test S_meta ≈ S_src atol = 1e-10

    # Modifier parameter override via set_modifier_parameter!
    alpha2 = 2.0
    set_modifier_parameter!(mc, ExtraDimensionPenaltyModifier, :alpha, alpha2)
    E_meta2 = eddp_energy(mc)
    expected_penalty2 = alpha2 * sum(pos_4d[4, :] .^ 2)
    @test E_meta2 ≈ E_src + expected_penalty2 atol = 1e-10
end
