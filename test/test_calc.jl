using LinearAlgebra
using EDDPotentials
using EDDPotentials: get_cell, get_forces, get_stress, get_energy
using Test
using CellBase
using Flux

@testset "Calc" begin
    cell = _h2_cell()

    cf = EDDPotentials.CellFeature(unique(species(cell)), p2=[2], q3=[2, 3], p3=[2, 3])

    function _test_forces_fd(calc, amp=1e-7, atol=1e-5)
        ftmp = copy(EDDPotentials.get_forces(calc))
        etmp = EDDPotentials.get_energy(calc)
        positions(get_cell(calc))[1] += amp
        @test EDDPotentials._need_calc(calc, true)
        tmp = (get_energy(calc) - etmp) / amp
        @test -tmp ≈ ftmp[1] atol = atol
    end

    function _test_forces_fd_vc(calc; amp=1e-7, atol=1e-5, idx=1)
        if !isa(calc, EDDPotentials.VariableCellCalc)
            calc = VariableCellCalc(calc)
        end
        ftmp = copy(EDDPotentials.get_forces(calc))
        etmp = EDDPotentials.get_enthalpy(calc)
        positions(get_cell(calc))[idx] += amp
        @test EDDPotentials._need_calc(calc, true)
        tmp = (EDDPotentials.get_enthalpy(calc) - etmp) / amp
        @test -tmp ≈ ftmp[idx] atol = atol
    end



    @testset "MBP" begin
        nnitf = EDDPotentials.ManualFluxBackPropInterface(
            Chain(Dense(rand(5, EDDPotentials.nfeatures(cf))), Dense(rand(1, 5))),
        )
        calc = EDDPotentials.NNCalc(cell, cf, nnitf; core=nothing)
        nnitf.chain(calc.v)

        # Test copying positions
        cell2 = deepcopy(cell)
        positions(cell2) .= 0.0
        EDDPotentials.copycell!(cell, cell2)
        @test EDDPotentials.is_equal(cell, cell2)

        eng = EDDPotentials.get_energy(calc)
        @test isa(eng, Float64)

        forces = EDDPotentials.get_forces(calc)
        # Newton's second law
        @test all(isapprox.(sum(forces, dims=2), 0, atol=1e-10))

        stress = EDDPotentials.get_stress(calc)
        @test size(stress) == (3, 3)
        @test any(stress .!== 0.0)

        # Test against small displacements finite displacements
        _test_forces_fd(calc)
        _test_forces_fd_vc(calc)
    end

    @testset "MBP&Embedding" begin
        embed = EDDPotentials.CellEmbedding(cf, 2)
        nnitf = EDDPotentials.ManualFluxBackPropInterface(cf, 5; embedding=embed)
        calc = EDDPotentials.NNCalc(cell, cf, nnitf; core=nothing)
        nnitf.chain(calc.v)

        eng = EDDPotentials.get_energy(calc)
        @test isa(eng, Float64)

        forces = EDDPotentials.get_forces(calc)
        # Newton's second law
        @test all(isapprox.(sum(forces, dims=2), 0, atol=1e-10))

        stress = EDDPotentials.get_stress(calc)
        @test size(stress) == (3, 3)
        @test any(stress .!== 0.0)

        # Test against small displacements finite displacements
        _test_forces_fd(calc)
        _test_forces_fd_vc(calc)
    end

    @testset "Flux&Embedding" begin
        embed = EDDPotentials.CellEmbedding(cf, 2)
        model = EDDPotentials.flux_mlp_model(cf, 5; embedding=embed)
        nnitf = EDDPotentials.FluxInterface(model)
        calc = EDDPotentials.NNCalc(cell, cf, nnitf; core=nothing)
        nnitf.model(calc.v)

        eng = EDDPotentials.get_energy(calc)
        @test isa(eng, Float64)

        forces = EDDPotentials.get_forces(calc)
        # Newton's second law
        @test all(isapprox.(sum(forces, dims=2), 0, atol=1e-10))

        stress = EDDPotentials.get_stress(calc)
        @test size(stress) == (3, 3)
        @test any(stress .!== 0.0)

        # Test against small displacements finite displacements
        _test_forces_fd(calc)
        _test_forces_fd_vc(calc)
    end


    @testset "Ensemble" begin
        nnitfs = [
            EDDPotentials.ManualFluxBackPropInterface(
                Chain(Dense(rand(5, EDDPotentials.nfeatures(cf))), Dense(rand(1, 5))),
            ) for _ = 1:5
        ]
        nnitf = EDDPotentials.EnsembleNNInterface(Tuple(nnitfs), repeat([0.2], 5))
        calc = EDDPotentials.NNCalc(cell, cf, nnitf; core=nothing)
        eng = get_energy(calc)
        std_tot = EDDPotentials.get_energy_std(calc)
        @test eng != 0.0
        @test std_tot != 0.0

        _test_forces_fd(calc)
        _test_forces_fd_vc(calc)
    end

    @testset "Linear" begin
        nnitf = EDDPotentials.LinearInterface(rand(EDDPotentials.nfeatures(cf)))
        calc = EDDPotentials.NNCalc(cell, cf, nnitf; core=nothing)
        eng = get_energy(calc)
        forces = get_forces(calc)
        stress = get_stress(calc)

        @test isa(eng, Float64)

        @test all(isapprox.(sum(forces, dims=2), 0, atol=1e-10))
        @test size(stress) == (3, 3)
        @test any(stress .!== 0.0)

        _test_forces_fd(calc)
        _test_forces_fd_vc(calc)
    end

    @testset "VCFilter" begin
        nnitf = EDDPotentials.LinearInterface(rand(EDDPotentials.nfeatures(cf)))
        calc = EDDPotentials.NNCalc(cell, cf, nnitf)
        filter = EDDPotentials.VariableCellCalc(calc)
        _test_forces_fd(filter)
        _test_forces_fd_vc(filter)
        stress = get_stress(filter)
        @test size(stress) == (3, 3)
        @test any(stress .!== 0.0)

        # Sizes of the positions
        pos = get_positions(filter)
        @test size(pos) == (3, 5)
        pos[1] += 1e-9
        set_positions!(filter, pos)
        @test EDDPotentials._need_calc(filter, false)

        # External pressure
        filter =
            EDDPotentials.VariableCellCalc(calc; external_pressure=diagm([3.0, 3.0, 3.0]))
        _test_forces_fd_vc(filter, idx=1)
        _test_forces_fd_vc(filter, idx=nions(get_cell(calc)) + 3)
        @test get_energy(filter) != EDDPotentials.get_enthalpy(filter)
        @test get_pressure(filter) != 0
        @test get_pressure(calc) != 0
    end

    @testset "Neigh" begin
        # Check NeighbourList rebuild
        nnitf = EDDPotentials.LinearInterface(rand(EDDPotentials.nfeatures(cf)))
        calc = EDDPotentials.NNCalc(cell, cf, nnitf)
        p1 = calc.last_nn_build_pos[1]
        positions(get_cell(calc))[1] += 3.0
        EDDPotentials.calculate!(calc; rebuild_nl=false)
        @test p1 != calc.last_nn_build_pos[1]

        # This should not Trigger rebuild
        p1 = calc.last_nn_build_pos[1]
        positions(get_cell(calc))[1] += 0.001
        EDDPotentials.calculate!(calc; rebuild_nl=false)
        @test p1 == calc.last_nn_build_pos[1]
    end
end


@testset "Relax" begin
    cell = _h2_cell(10, 1.5)
    cf = EDDPotentials.CellFeature(
        unique(species(cell));
        rcut2=3.5,
        p2=[6, 12],
        p3=[],
        q3=[],
    )

    nnitf = EDDPotentials.LinearInterface(rand(EDDPotentials.nfeatures(cf)))
    # Attractive potential with -5f(x)^6 + f(x)^12
    EDDPotentials.setparamvector!(nnitf, [0, -5, 1])
    calc = EDDPotentials.NNCalc(cell, cf, nnitf)

    # Test RelaxOption
    opts = EDDPotentials.RelaxOption()
    relax = EDDPotentials.Relax(calc, opts)
    relax = EDDPotentials.Relax(calc)

    # Perform relaxation
    output = EDDPotentials.multirelax!(relax)
    output = EDDPotentials.relax!(relax)

    # Expected distance
    rexp = (1 - (5 / 2 / 2^6)^(1 / 6)) * 3.5
    dd = distance_between(cell[1], cell[2])
    @test dd ≈ rexp atol = 1e-6

    # Test recording trajectory
    cell = _h2_cell(10, 1.5)
    calc = EDDPotentials.NNCalc(cell, cf, nnitf)
    res = EDDPotentials.relax!(calc; keep_trajectory=true)
    @test res.converged

    dd = distance_between(cell[1], cell[2])
    @test dd ≈ rexp atol = 1e-6

    @test length(res.relax.trajectory) > 1
    @test :enthalpy in keys(res.relax.trajectory[1].metadata)
end

@testset "Opt" begin
    cell = _h2_cell()
    calc = EDDPotentials.lj_like_calc(cell; rc=6.0)
    traj = []
    @test EDDPotentials.opt_tpsd(calc, trajectory=traj)
    @test length(traj) > 1
    @test maximum(norm.(eachcol(EDDPotentials.get_forces(calc)))) < 1e-4
    @test EDDPotentials.opt_tpsd(EDDPotentials.VariableCellCalc(calc))
    @test maximum(EDDPotentials.get_stress(calc)) < 1e-4


    cell = _h2_cell()
    calc = EDDPotentials.lj_like_calc(cell; rc=6.0)
    # With external pressure
    p = 1e-2
    vc = EDDPotentials.VariableCellCalc(calc, external_pressure=diagm([p, p, p]))
    global vc
    EDDPotentials.opt_tpsd(vc;)
    @test maximum(abs.(EDDPotentials.get_forces(vc.calc))) < 1e-3
    @test maximum(EDDPotentials.get_stress(vc.calc)) > 1e-3
    @test maximum(EDDPotentials.get_stress(vc) .- vc.external_pressure) < 1e-4
end
