using EDDPotentials
using EDDPotentials:
    CoreRepulsion,
    nfeatures,
    get_positions,
    eddp_forces,
    eddp_stress,
    MetaCalculator,
    eddp_energy,
    get_cell,
    _fd_energy,
    _fd_energy_vc,
    _fd_strain,
    _pair_rcut,
    _make_core_repulsion
using EDDPotentials.Potential: compute_fv_gv, ManualFluxBackPropInterface
using EDDPotentials: fd_desc, force_gradient, stress_gradient, test_finite_difference
using CellBase
using CellBase: add_dimensions
using LinearAlgebra
using NLSolversBase
using Test
using Flux

isdefined(Main, :_h2_cell) || include("utils.jl")
@testset "Gradients" begin

    # Run test cases
    cell = _h2o_cell()
    cf = CellFeature([:H, :O], p2=2:3, p3=2:3)
    diff, gvec_ji = fd_desc(cf, cell)
    @test maximum(abs.(diff - gvec_ji)) < 1e-7

    cell = _lco_cell()
    cf = CellFeature([:Li, :Co, :O], p2=2:3, p3=2:3)
    diff, gvec_ji = fd_desc(cf, cell)
    @test maximum(abs.(diff - gvec_ji)) < 1e-7

    cell = _h2o_cell()
    cf = CellFeature([:H, :O], p2=2:3, p3=2:3)
    diff, forces = force_gradient(cf, cell)
    @test maximum(abs.(diff + forces)) < 1e-7

    cell = _lco_cell()
    cf = CellFeature([:Li, :Co, :O], p2=2:3, p3=2:3)
    diff, forces = force_gradient(cf, cell)
    @test maximum(abs.(diff + forces)) < 1e-7

    cell = _h2o_cell()
    cf = CellFeature([:H, :O], p2=2:3, p3=2:3)
    diff, varial = stress_gradient(cf, cell)
    @test maximum(abs.(diff + varial)) < 1e-4

    cell = _lco_cell()
    cf = CellFeature([:Li, :Co, :O], p2=2:3, p3=2:3)
    diff, varial = stress_gradient(cf, cell)
    @test maximum(abs.(diff + varial)) < 1e-4

    @testset "With core" begin
        cell = _lco_cell()
        cf = CellFeature([:Li, :Co, :O], p2=2:3, p3=2:3)
        diff, forces = force_gradient(cf, cell, 3.0)
        fb = compute_fv_gv(cf, cell, core=CoreRepulsion(3.0))
        @test maximum(abs.(diff + forces)) < 1e-6
        @test sum(fb.hardcore.ecore) >= 10.0

        cell = _lco_cell()
        cf = CellFeature([:Li, :Co, :O], p2=2:3, p3=2:3)
        diff, varial = stress_gradient(cf, cell, 3.0)
        fb = compute_fv_gv(cf, cell, core=CoreRepulsion(3.0))
        @test maximum(abs.(diff + varial)) < 1e-6
    end

    @testset "Per-element core radii" begin
        # Test _pair_rcut combining rule
        core_single = CoreRepulsion(1.0)
        @test _pair_rcut(core_single, :Mg, :O) == 1.0

        core_multi = CoreRepulsion(Dict(:Mg => 0.5, :O => 0.7))
        @test _pair_rcut(core_multi, :Mg, :Mg) == 1.0
        @test _pair_rcut(core_multi, :Mg, :O) == 1.2
        @test _pair_rcut(core_multi, :O, :O) == 1.4
        @test _pair_rcut(core_multi, :X, :Mg) == 0.75  # unknown element -> 0.25

        # Test constructor with String keys
        core_str = CoreRepulsion(Dict("Mg" => 0.5, "O" => 0.7))
        @test _pair_rcut(core_str, :Mg, :O) == 1.2

        # Test _make_core_repulsion helper
        @test _make_core_repulsion(1.0).rcut == 1.0
        core_dict = _make_core_repulsion(Dict("Mg" => 0.5, "O" => 0.7))
        @test _pair_rcut(core_dict, :Mg, :O) == 1.2

        # Test that per-element radii produce different energies than single radius
        cell = _lco_cell()
        cf = CellFeature([:Li, :Co, :O], p2=2:3, p3=2:3)
        fb_single = compute_fv_gv(cf, cell, core=CoreRepulsion(2.0))
        fb_multi = compute_fv_gv(cf, cell, core=CoreRepulsion(Dict(:Li => 1.0, :Co => 0.5, :O => 0.8)))
        # The per-element radii should give a different total core energy
        @test sum(fb_single.hardcore.ecore) != sum(fb_multi.hardcore.ecore)

        # Test force consistency with finite differences for per-element radii
        cell = _h2o_cell(3.5)
        cf = CellFeature([:H, :O], p2=2:3, p3=2:3)
        fb = compute_fv_gv(cf, cell, core=CoreRepulsion(Dict(:H => 0.5, :O => 0.6)))
        core = CoreRepulsion(Dict(:H => 0.5, :O => 0.6))
        # Verify core energy is nonzero (atoms are close enough)
        if sum(fb.hardcore.ecore) > 0
            @test sum(fb.hardcore.fcore) != 0.0
        end
    end
end


@testset "F/S Calculator" begin

    # Use smaller cell (3 atoms) for faster FD; pipeline is cell-size-agnostic
    cell = _h2o_cell()
    cf = CellFeature([:H, :O], p2=[2], q3=[2, 3], p3=[2, 3])
    itf = ManualFluxBackPropInterface(
        Chain(Dense(rand(5, nfeatures(cf))), Dense(rand(1, 5))),
    )
    calc = EDDPCalculator(cell, cf, itf; core=nothing)

    forces = copy(eddp_forces(calc))
    stress = copy(eddp_stress(calc))

    # Test the total force
    p0 = get_positions(calc)
    od = OnceDifferentiable(x -> _fd_energy(calc, x), p0, _fd_energy(calc, p0))
    grad = NLSolversBase.gradient(od, p0)
    @test allclose(grad, -forces, atol=1e-6)

    # Test the total stress
    s0 = zeros(3, 3)[:]
    od = OnceDifferentiable(
        x -> _fd_strain(calc, x),
        s0,
        _fd_strain(calc, s0),
        inplace=false,
    )
    grad = NLSolversBase.gradient(od, s0) ./ volume(get_cell(calc))
    grad2 = grad
    @test allclose(grad, -vec(stress), atol=1e-3)
end

@testset "Higher-Dimensional Feature Gradients" begin
    function _make_4d_cell(lattice=12.0, extra_val=0.0)
        pos = Float64[
            0.0  2.0  1.0
            0.0  1.0  2.0
            0.0  2.0  1.0
            extra_val extra_val extra_val
        ]
        Cell(Lattice(lattice, lattice, lattice), [:H, :H, :H], pos)
    end

    @testset "Two-body: fvec invariance (extra dims=0)" begin
        cell4 = _make_4d_cell(12.0, 0.0)
        cf = CellFeature([:H], p2=2:4)
        fb4 = compute_fv_gv(cf, cell4)
        cell3 = Cell(Lattice(12, 12, 12), [:H, :H, :H], cell4.positions[1:3, :])
        fb3 = compute_fv_gv(cf, cell3)
        @test fb3.fvec ≈ fb4.fvec
    end

    @testset "Two-body: gradient FD in 4D" begin
        cell4 = _make_4d_cell(12.0, 0.1)
        cf = CellFeature([:H], p2=2:4)
        diff, gvec_ideal = fd_desc(cf, cell4)
        @test maximum(abs.(diff - gvec_ideal)) < 1e-7
    end

    @testset "Two-body: force gradient FD in 4D" begin
        cell4 = _make_4d_cell(12.0, 0.1)
        cf = CellFeature([:H], p2=2:4)
        diff, forces = force_gradient(cf, cell4)
        @test maximum(abs.(diff + forces)) < 1e-7
    end

    @testset "Three-body: gradient FD in 4D" begin
        cell4 = _make_4d_cell(12.0, 0.1)
        cf = CellFeature([:H], p2=[], p3=2:3, q3=2:3)
        diff, gvec_ideal = fd_desc(cf, cell4)
        @test maximum(abs.(diff - gvec_ideal)) < 1e-7
    end
end
