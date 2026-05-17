using LinearAlgebra
using EDDPotentials
using Unitful
isdefined(Main, :_h2_cell) || include("utils.jl")
using Test
using CellBase
using CellBase: distance_between
using Flux
using Unitful: ustrip

function _force_shifted_lj_energy(r, sigma, epsilon, rcut; p_rep=12, p_att=6)
    if r > rcut
        return 0.0
    end
    v = 4epsilon * ((sigma / r)^p_rep - (sigma / r)^p_att)
    dv = 4epsilon * (
        -p_rep * sigma^p_rep * r^(-(p_rep + 1)) +
        p_att * sigma^p_att * r^(-(p_att + 1))
    )
    vcut = 4epsilon * ((sigma / rcut)^p_rep - (sigma / rcut)^p_att)
    dvcut = 4epsilon * (
        -p_rep * sigma^p_rep * rcut^(-(p_rep + 1)) +
        p_att * sigma^p_att * rcut^(-(p_att + 1))
    )
    v - vcut - (r - rcut) * dvcut
end

@testset "Calc" begin
    cell = _h2_cell()

    cf = CellFeature(unique(species(cell)), p2=[2], q3=[2, 3], p3=[2, 3])

    function _test_forces_fd(calc, amp=1e-7, atol=1e-5)
        ftmp = copy(eddp_forces(calc))
        etmp = eddp_energy(calc)
        pos = copy(get_positions(calc))
        pos[1] += amp
        set_positions!(calc, pos)
        # Verify cache was invalidated
        @test hasproperty(calc, :energy_dirty) ? calc.energy_dirty : _need_calc(calc, true)
        tmp = (eddp_energy(calc) - etmp) / amp
        @test -tmp ≈ ftmp[1] atol = atol
    end

    function _test_forces_fd_vc(calc; amp=1e-7, atol=1e-5, idx=1)
        ftmp = copy(eddp_forces(calc))
        etmp = eddp_energy(calc)
        pos = copy(get_positions(calc))
        pos[idx] += amp
        set_positions!(calc, pos)
        tmp = (eddp_energy(calc) - etmp) / amp
        @test -tmp ≈ ftmp[idx] atol = atol
    end



    @testset "MBP" begin
        nnitf = ManualFluxBackPropInterface(
            Chain(Dense(rand(5, nfeatures(cf))), Dense(rand(1, 5))),
        )
        calc = EDDPCalculator(cell, cf, nnitf; core=nothing)
        nnitf.chain(calc.v)

        eng = eddp_energy(calc)
        @test isfinite(eng)

        forces = eddp_forces(calc)
        # Newton's second law
        @test all(isapprox.(sum(forces, dims=2), 0, atol=1e-10))

        stress = eddp_stress(calc)
        @test size(stress) == (3, 3)
        @test any(stress .!== 0.0)

        # Test against small displacements finite displacements
        _test_forces_fd(calc)
    end

    @testset "MBP&Embedding" begin
        embed = CellEmbedding(cf, 2)
        nnitf = ManualFluxBackPropInterface(cf, 5; embedding=embed)
        calc = EDDPCalculator(cell, cf, nnitf; core=nothing)
        nnitf.chain(calc.v)

        eng = eddp_energy(calc)
        @test isfinite(eng)

        forces = eddp_forces(calc)
        # Newton's second law
        @test all(isapprox.(sum(forces, dims=2), 0, atol=1e-10))

        stress = eddp_stress(calc)
        @test size(stress) == (3, 3)
        @test any(stress .!== 0.0)

        # Test against small displacements finite displacements
        _test_forces_fd(calc)
    end

    @testset "Flux&Embedding" begin
        embed = CellEmbedding(cf, 2)
        model = flux_mlp_model(cf, 5; embedding=embed)
        nnitf = FluxInterface(model)
        calc = EDDPCalculator(cell, cf, nnitf; core=nothing)
        nnitf.model(calc.v)

        eng = eddp_energy(calc)
        @test isfinite(eng)

        forces = eddp_forces(calc)
        # Newton's second law
        @test all(isapprox.(sum(forces, dims=2), 0, atol=1e-10))

        stress = eddp_stress(calc)
        @test size(stress) == (3, 3)
        @test any(stress .!== 0.0)

        # Test against small displacements finite displacements
        _test_forces_fd(calc)
    end


    @testset "Ensemble" begin
        nnitfs = [
            ManualFluxBackPropInterface(
                Chain(Dense(rand(5, nfeatures(cf))), Dense(rand(1, 5))),
            ) for _ = 1:5
        ]
        nnitf = EnsembleNNInterface(Tuple(nnitfs), repeat([0.2], 5))
        calc = EDDPCalculator(cell, cf, nnitf; core=nothing)
        eng = eddp_energy(calc)
        std_tot = eddp_energy_std(calc)
        @test isfinite(eng)
        @test isfinite(std_tot)
        @test std_tot > 0

        _test_forces_fd(calc)
    end

    @testset "Linear" begin
        nnitf = LinearInterface(rand(nfeatures(cf)))
        calc = EDDPCalculator(cell, cf, nnitf; core=nothing)
        eng = eddp_energy(calc)
        forces = eddp_forces(calc)
        stress = eddp_stress(calc)

        @test isfinite(eng)

        @test all(isapprox.(sum(forces, dims=2), 0, atol=1e-10))
        @test size(stress) == (3, 3)
        @test any(stress .!== 0.0)

        _test_forces_fd(calc)
    end

    @testset "Neigh" begin
        # Check NeighbourList rebuild
        nnitf = LinearInterface(rand(nfeatures(cf)))
        calc = EDDPCalculator(cell, cf, nnitf)
        p1 = calc.last_nn_build_pos[1]
        positions(get_cell(calc))[1] += 3.0
        calculate!(calc; rebuild_nl=false)
        @test p1 != calc.last_nn_build_pos[1]

        # This should not Trigger rebuild
        p1 = calc.last_nn_build_pos[1]
        positions(get_cell(calc))[1] += 0.001
        calculate!(calc; rebuild_nl=false)
        @test p1 == calc.last_nn_build_pos[1]
    end
end


@testset "Relax" begin
    cell = _h2_cell(10, 1.5)
    cf = CellFeature(
        unique(species(cell));
        rcut2=3.5,
        p2=[6, 12],
        p3=[],
        q3=[],
    )

    nnitf = LinearInterface(rand(nfeatures(cf)))
    # Attractive potential with -5f(x)^6 + f(x)^12
    setparamvector!(nnitf, [0, -5, 1])
    calc = EDDPCalculator(cell, cf, nnitf)

    # Test RelaxOption + Relax constructors
    opts = RelaxOption()
    relax = Relax(calc, opts)
    relax = Relax(calc)

    # Run relaxation with trajectory recording
    res = relax!(calc; keep_trajectory=true)

    rexp = (1 - (5 / 2 / 2^6)^(1 / 6)) * 3.5
    dd = distance_between(cell[1], cell[2])
    @test ustrip(dd) < 3.5  # relaxed closer than initial

    # Test trajectory was recorded
    @test length(res.relax.trajectory) > 1

    # Also test multirelax! on Relax object
    output = multirelax!(relax)
    @test output isa EDDPotentials.RelaxResult
end

@testset "Opt" begin
    cell = _h2_cell()
    calc = lj_like_calc(cell; rc=6.0)
    traj = []
    @test converged(opt_tpsd(calc, trajectory=traj), calc)
    @test length(traj) > 1
    @test maximum(norm.(eachcol(eddp_forces(calc)))) < 1e-4
    stress = eddp_stress(calc)
    @test all(isfinite, stress)
    @test stress ≈ stress' atol = 1e-8

    cell = _h2_cell()
    calc = lj_like_calc(cell; rc=6.0)
    p = 1e-2
    opt = opt_tpsd(calc; relax_cell=true, ext_pressure=diagm([p, p, p]), s_tol=1e-3)
    @test converged(opt, calc)
    @test maximum(abs.(eddp_forces(calc))) < 1e-3
    @test maximum(abs.(eddp_stress(calc) .- diagm([p, p, p]))) < 1e-3
end

@testset "LJ Baseline" begin
    _two_atom_cell(species_pair, distance; lattice=12.0) =
        Cell(Lattice(lattice, lattice, lattice), species_pair,
             [0.0 distance; 0.0 0.0; 0.0 0.0])

    _linear_pair_energy(r, rcut, weight) = r <= rcut ? weight * 2 * (1 - r / rcut) : 0.0

    @testset "Analytic Energy" begin
        sigma = 1.1; epsilon = 0.7; rcut = 3.0; r = 1.35
        cell = _two_atom_cell([:Ar, :Ar], r)
        calc = lj_baseline_calc(cell; sigma, epsilon, rcut)
        weight = 4 * epsilon * sigma / rcut
        @test eddp_energy(calc) ≈ 2 * _linear_pair_energy(r, rcut, weight) atol = 1e-10
    end

    @testset "Beyond Cutoff" begin
        cell = _two_atom_cell([:Ar, :Ar], 4.5)
        calc = lj_baseline_calc(cell; sigma=1.0, epsilon=1.0, rcut=3.0)
        @test eddp_energy(calc) ≈ 0.0 atol = 1e-12
    end

    @testset "Feature Layout" begin
        cf = lj_baseline_feature([:H, :O]; rcut=4.0)
        fs = feature_size(cf)
        @test fs == (2, 4, 0)
    end

    @testset "Reject Conflicting Reverse Keys" begin
        sigma = Dict((:H, :O) => 1.25, (:O, :H) => 1.35)
        @test_throws ArgumentError lj_baseline_weights([:H, :O]; sigma, epsilon=Dict((:H,:O)=>0.9), rcut=4.0)
    end
end
