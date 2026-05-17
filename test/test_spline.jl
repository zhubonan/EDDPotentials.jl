using EDDPotentials
using EDDPotentials.Potential: fr, get_workspace, evaluate_p2, evaluate_p3, evaluate_q3, build_spline_table, SplineTable, compute_fv_gv, compute_fv!, compute_fv_gv!, suggest_rcut, reset!
using CellBase
using Random
using Test

isdefined(Main, :_h2_cell) || include("utils.jl")

@testset "Spline Accuracy" begin

    @testset "evaluate_svector vs direct ^" begin
        powers = (2.0, 2.517, 3.175, 4.007, 5.057, 6.385, 8.063, 10.0)
        rcut = 5.0
        tbl = build_spline_table(powers, rcut, 3000; f=fr)

        for r in [0.01, 0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
            fval = fr(r, rcut)
            direct = ntuple(i -> fval^powers[i], length(powers))
            spline = Tuple(EDDPotentials.Potential.evaluate_svector(tbl, Float64(r)))
            for i in eachindex(powers)
                @test abs(spline[i] - direct[i]) < max(1e-8, abs(direct[i]) * 1e-11)
            end
        end

        max_abs = 0.0
        for r in [4.9, 4.99, 4.999]
            fval = fr(r, rcut)
            direct = ntuple(i -> fval^powers[i], length(powers))
            spline = Tuple(EDDPotentials.Potential.evaluate_svector(tbl, Float64(r)))
            for i in eachindex(powers)
                max_abs = max(max_abs, abs(spline[i] - direct[i]))
            end
        end
        @test max_abs < 1e-7
    end

    @testset "feature vectors: spline vs direct (Float64 exponents)" begin
        cell = _h2o_cell()
        geom_powers = EDDPotentials.Potential.genp(2, 10, 8)
        cf = CellFeature([:H, :O]; p2=geom_powers, p3=geom_powers, q3=geom_powers, rcut2=5.0, rcut3=5.0)

        rcut = suggest_rcut(cf; shell=0.0)
        nl = NeighbourList(cell, rcut; savevec=true)

        ws_spline = get_workspace(cf, nl, true)
        ws_direct = get_workspace(cf, nl, true)

        compute_fv_gv!(ws_spline, cf, cell; nl, use_spline=true)
        compute_fv_gv!(ws_direct, cf, cell; nl, use_spline=false)

        feat_err = maximum(abs.(ws_spline.fvec .- ws_direct.fvec))
        @test feat_err < 1e-6

        grad_err = maximum(abs.(ws_spline.gvec .- ws_direct.gvec))
        @test grad_err < 1e-5

        stress_err = maximum(abs.(ws_spline.stotv .- ws_direct.stotv))
        @test stress_err < 1e-5
    end

    @testset "feature vectors: spline vs direct (multi-species)" begin
        cell = _lco_cell(rng=Random.MersenneTwister(5))
        geom_powers = EDDPotentials.Potential.genp(2, 8, 6)
        cf = CellFeature([:Li, :Co, :O]; p2=geom_powers, p3=geom_powers, q3=geom_powers, rcut2=4.0, rcut3=4.0)

        rcut = suggest_rcut(cf; shell=0.0)
        nl = NeighbourList(cell, rcut; savevec=true)

        ws_spline = get_workspace(cf, nl, true)
        ws_direct = get_workspace(cf, nl, true)

        compute_fv_gv!(ws_spline, cf, cell; nl, use_spline=true)
        compute_fv_gv!(ws_direct, cf, cell; nl, use_spline=false)

        feat_err = maximum(abs.(ws_spline.fvec .- ws_direct.fvec))
        @test feat_err < 1e-6

        grad_err = maximum(abs.(ws_spline.gvec .- ws_direct.gvec))
        @test grad_err < 1e-4

        stress_err = maximum(abs.(ws_spline.stotv .- ws_direct.stotv))
        @test stress_err < 1e-4
    end

    @testset "spline_ngrid kwarg" begin
        cf_default = CellFeature([:C]; p2=[2.0, 3.0, 4.0], rcut2=5.0, spline_ngrid=3000)
        cf_coarse = CellFeature([:C]; p2=[2.0, 3.0, 4.0], rcut2=5.0, spline_ngrid=100)
        @test size(cf_default.p2_table.coeffs, 2) == 2999
        @test size(cf_coarse.p2_table.coeffs, 2) == 99
    end

    @testset "evaluate_p2/p3/q3 dispatch (Int exponents)" begin
        cf_int = CellFeature([:C]; p2=2:4, p3=2:3, q3=2:3, rcut2=5.0, rcut3=5.0)
        r = 2.5
        fijp2, fij2 = evaluate_p2(cf_int, r)
        @test fijp2 ≈ fij2 .^ cf_int.p2
        pij, ftmp = evaluate_p3(cf_int, r)
        @test pij ≈ ftmp .^ cf_int.p3
        qjk, ftmp2 = evaluate_q3(cf_int, r)
        @test qjk ≈ ftmp2 .^ cf_int.q3
    end

    @testset "empty exponents" begin
        cf_empty = CellFeature([:C]; p2=[], p3=[], q3=[], rcut2=5.0, rcut3=5.0)
        @test cf_empty.p2_table.nintervals == 0
        @test cf_empty.p3_table.nintervals == 0
        @test cf_empty.q3_table.nintervals == 0
    end
end
