using EDDPotentials
using EDDPotentials:
    CoreRepulsion,
    compute_fv_gv,
    get_energy,
    get_cell,
    _fd_energy,
    _fd_energy_vc,
    _fd_strain
using EDDPotentials: fd_desc, force_gradient, stress_gradient, test_finite_difference
using CellBase
using LinearAlgebra
using NLSolversBase
using Test
using Flux


@testset "Gradients" begin

    # Run test cases
    cell = _h2o_cell()
    cf = CellFeature([:H, :O], p2=2:2)
    diff, gvec_ji = fd_desc(cf, cell)
    @test maximum(abs.(diff - gvec_ji)) < 1e-7

    cell = _lco_cell()
    cf = CellFeature([:Li, :Co, :O], p2=2:2)
    diff, gvec_ji = fd_desc(cf, cell)
    @test maximum(abs.(diff - gvec_ji)) < 1e-7

    cell = _h2o_cell()
    cf = CellFeature([:H, :O], p2=2:2)
    diff, forces = force_gradient(cf, cell)
    @test maximum(abs.(diff + forces)) < 1e-7

    cell = _lco_cell()
    cf = CellFeature([:Li, :Co, :O], p2=2:2)
    diff, forces = force_gradient(cf, cell)
    @test maximum(abs.(diff + forces)) < 1e-7

    cell = _h2o_cell()
    cf = CellFeature([:H, :O], p2=2:2)
    diff, varial = stress_gradient(cf, cell)
    @test maximum(abs.(diff + varial)) < 1e-4

    cell = _lco_cell()
    cf = CellFeature([:Li, :Co, :O], p2=2:2)
    diff, varial = stress_gradient(cf, cell)
    @test maximum(abs.(diff + varial)) < 1e-4

    @testset "With core" begin
        cell = _lco_cell()
        cf = CellFeature([:Li, :Co, :O], p2=2:2)
        diff, forces = force_gradient(cf, cell, 3.0)
        fb = compute_fv_gv(cf, cell, core=CoreRepulsion(3.0))
        @test maximum(abs.(diff + forces)) < 1e-6
        @test sum(fb.hardcore.ecore) >= 10.0

        cell = _lco_cell()
        cf = CellFeature([:Li, :Co, :O], p2=2:2)
        diff, varial = stress_gradient(cf, cell, 3.0)
        fb = compute_fv_gv(cf, cell, core=CoreRepulsion(3.0))
        @test maximum(abs.(diff + varial)) < 1e-6
    end
end


@testset "F/S Calculator" begin

    calc = _get_lco_calc()
    ntot = EDDPotentials.nfeatures(calc.cf)
    model = Chain(Dense(ones(1, ntot)))
    itf = EDDPotentials.ManualFluxBackPropInterface(model)

    forces = copy(EDDPotentials.get_forces(calc))
    stress = copy(EDDPotentials.get_stress(calc))

    # Test the total force
    p0 = EDDPotentials.get_positions(calc)
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

    # Test wrapper
    # NOTE Somehow this is needed here - possible BUG?
    vc = EDDPotentials.VariableCellCalc(calc)
    epos = EDDPotentials.get_positions(vc)
    eforce = copy(EDDPotentials.get_forces(vc))

    od = OnceDifferentiable(
        x -> _fd_energy_vc(vc, x),
        epos,
        _fd_energy_vc(vc, epos);
        inplace=false,
    )
    grad = NLSolversBase.gradient(od, epos)
    @test allclose(grad, -eforce, atol=1e-4, rtol=1e-4)

    @test test_finite_difference(calc)
end
