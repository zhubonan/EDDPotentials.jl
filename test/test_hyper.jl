using EDDPotentials
using EDDPotentials:
    scheduled_penalty_relax!, hyper_gen_relax, bP,
    MetaCalculator, ExtraDimensionPenaltyModifier,
    create_calculator, set_modifier_parameter!, get_modifier_parameter
using EDDPotentials.EDDPotentialsCore:
    LinearInterface, ManualFluxBackPropInterface, CellFeature, nfeatures,
    add_dimensions
using CellBase, Test, LinearAlgebra, Flux

@testset "Hyper" begin
    function _make_hyper_calc(; alpha=0.01)
        cell3d = Cell(Lattice(5,5,5), [:H,:H], [0 1.5; 0 0.0; 0 0])
        hcell = Cell(Lattice(5,5,5), [:H,:H], [0 1.5; 0 0.0; 0 0; 0.1 0.2])
        cf = CellFeature([:H]; p2=[2.0,3.0], p3=Float64[], q3=Float64[])
        model = LinearInterface(rand(nfeatures(cf)))
        calc = EDDPCalculator(cf, model, Dict{Symbol,Float64}(); core_repulsion=nothing)
        penalty = ExtraDimensionPenaltyModifier(alpha; start_dim=4)
        MetaCalculator(create_calculator(calc, hcell), [penalty])
    end

    @testset "bP" begin
        p = bP(1.0)
        @test p[1,1] ≈ p[2,2] ≈ p[3,3]
        @test p[1,2] ≈ 0 atol=1e-15
    end

    @testset "scheduled_penalty_relax! basic" begin
        calc = _make_hyper_calc()
        result = scheduled_penalty_relax!(calc;
            schedule_every=3, max_iterations=6, alpha_start=0.01, ramp_alpha=false)
        @test !isempty(result.results)
        @test length(result.extra_dim_norms) > 0
        @test !isempty(result.round_diagnostics)
        d = result.round_diagnostics[1]
        @test d.round == 1
        @test d.alpha_before ≈ 0.01
        @test isfinite(d.energy)
    end

    @testset "scheduled_penalty_relax! ramp_alpha" begin
        calc = _make_hyper_calc()
        result = scheduled_penalty_relax!(calc;
            schedule_every=3, max_iterations=9, alpha_start=0.001,
            ramp_alpha=true, alpha_ramp_factor=10.0, max_alpha=1.0)
        if length(result.round_diagnostics) >= 2
            @test result.round_diagnostics[2].alpha_before > result.round_diagnostics[1].alpha_before
        end
    end

    @testset "scheduled_penalty_relax! uses remainder iterations" begin
        noop_relax(calc; itermax, kwargs...) = (; iteration=itermax)

        short = scheduled_penalty_relax!(
            _make_hyper_calc();
            schedule_every=10,
            max_iterations=5,
            alpha_start=0.01,
            relax_fn=noop_relax,
            ramp_alpha=false,
            pos_tol=-1.0,
            extra_dim_tol=0.0,
        )
        @test [r.iteration for r in short.results] == [5]

        remainder = scheduled_penalty_relax!(
            _make_hyper_calc();
            schedule_every=3,
            max_iterations=7,
            alpha_start=0.01,
            relax_fn=noop_relax,
            ramp_alpha=false,
            pos_tol=-1.0,
            extra_dim_tol=0.0,
        )
        @test [r.iteration for r in remainder.results] == [3, 3, 1]
    end

    @testset "relax! hyperspace MetaCalculator returns result" begin
        cell = Cell(Lattice(5,5,5), [:H,:H], [0 1.5; 0 0.0; 0 0])
        cf = CellFeature([:H]; p2=[2.0], p3=Float64[], q3=Float64[])
        model = LinearInterface(rand(nfeatures(cf)))
        base = EDDPCalculator(cf, model, Dict{Symbol,Float64}(); core_repulsion=nothing)
        calc = MetaCalculator(create_calculator(base, cell))

        result = relax!(
            calc;
            hyper_ndim=1,
            iterations=1,
            hyper_schedule_every=1,
            hyper_target_iterations=1,
            relax_cell=false,
            multi=false,
        )
        @test result isa EDDPotentials.RelaxResult
    end

end
