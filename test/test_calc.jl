using LinearAlgebra
using EDDPotentials
using Unitful
using EDDPotentials:
    CellFeature,
    MetaCalculator,
    ExtraDimensionPenaltyModifier,
    NNCalc,
    lj_baseline_calc,
    lj_baseline_feature,
    lj_baseline_weights,
    nfeatures,
    get_cell,
    get_positions,
    set_positions!,
    calculate!,
    eddp_forces,
    eddp_stress,
    eddp_energy,
    eddp_enthalpy,
    eddp_energy_std,
    eddp_pressure,
    RelaxOption,
    Relax,
    multirelax!,
    relax!,
    lj_like_calc,
    opt_tpsd
using EDDPotentials.EDDPotentialsCore:
    _need_calc,
    ManualFluxBackPropInterface,
    CellEmbedding,
    flux_mlp_model,
    feature_size,
    FluxInterface,
    EnsembleNNInterface,
    LinearInterface,
    setparamvector!
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
        calc = NNCalc(cell, cf, nnitf; core=nothing)
        nnitf.chain(calc.v)

        # # Test copying positions
        # cell2 = deepcopy(cell)
        # positions(cell2) .= 0.0
        # EDDPotentials.copycell!(cell, cell2)
        # @test EDDPotentials.is_equal(cell, cell2)

        eng = eddp_energy(calc)
        @test isa(eng, Float64)

        forces = eddp_forces(calc)
        # Newton's second law
        @test all(isapprox.(sum(forces, dims=2), 0, atol=1e-10))

        stress = eddp_stress(calc)
        @test size(stress) == (3, 3)
        @test any(stress .!== 0.0)

        # Test against small displacements finite displacements
        _test_forces_fd(calc)
        _test_forces_fd_vc(calc)
    end

    @testset "MBP&Embedding" begin
        embed = CellEmbedding(cf, 2)
        nnitf = ManualFluxBackPropInterface(cf, 5; embedding=embed)
        calc = NNCalc(cell, cf, nnitf; core=nothing)
        nnitf.chain(calc.v)

        eng = eddp_energy(calc)
        @test isa(eng, Float64)

        forces = eddp_forces(calc)
        # Newton's second law
        @test all(isapprox.(sum(forces, dims=2), 0, atol=1e-10))

        stress = eddp_stress(calc)
        @test size(stress) == (3, 3)
        @test any(stress .!== 0.0)

        # Test against small displacements finite displacements
        _test_forces_fd(calc)
        _test_forces_fd_vc(calc)
    end

    @testset "Flux&Embedding" begin
        embed = CellEmbedding(cf, 2)
        model = flux_mlp_model(cf, 5; embedding=embed)
        nnitf = FluxInterface(model)
        calc = NNCalc(cell, cf, nnitf; core=nothing)
        nnitf.model(calc.v)

        eng = eddp_energy(calc)
        @test isa(eng, Float64)

        forces = eddp_forces(calc)
        # Newton's second law
        @test all(isapprox.(sum(forces, dims=2), 0, atol=1e-10))

        stress = eddp_stress(calc)
        @test size(stress) == (3, 3)
        @test any(stress .!== 0.0)

        # Test against small displacements finite displacements
        _test_forces_fd(calc)
        _test_forces_fd_vc(calc)
    end


    @testset "Ensemble" begin
        nnitfs = [
            ManualFluxBackPropInterface(
                Chain(Dense(rand(5, nfeatures(cf))), Dense(rand(1, 5))),
            ) for _ = 1:5
        ]
        nnitf = EnsembleNNInterface(Tuple(nnitfs), repeat([0.2], 5))
        calc = NNCalc(cell, cf, nnitf; core=nothing)
        eng = eddp_energy(calc)
        std_tot = eddp_energy_std(calc)
        @test eng != 0.0
        @test std_tot != 0.0

        _test_forces_fd(calc)
        _test_forces_fd_vc(calc)
    end

    @testset "Linear" begin
        nnitf = LinearInterface(rand(nfeatures(cf)))
        calc = NNCalc(cell, cf, nnitf; core=nothing)
        eng = eddp_energy(calc)
        forces = eddp_forces(calc)
        stress = eddp_stress(calc)

        @test isa(eng, Float64)

        @test all(isapprox.(sum(forces, dims=2), 0, atol=1e-10))
        @test size(stress) == (3, 3)
        @test any(stress .!== 0.0)

        _test_forces_fd(calc)
        _test_forces_fd_vc(calc)
    end

    @testset "Neigh" begin
        # Check NeighbourList rebuild
        nnitf = LinearInterface(rand(nfeatures(cf)))
        calc = NNCalc(cell, cf, nnitf)
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
    calc = NNCalc(cell, cf, nnitf)

    # Test RelaxOption
    opts = RelaxOption()
    relax = Relax(calc, opts)
    relax = Relax(calc)

    # Perform relaxation
    output = multirelax!(relax)
    output = relax!(relax)

    # Expected distance
    rexp = (1 - (5 / 2 / 2^6)^(1 / 6)) * 3.5
    dd = distance_between(cell[1], cell[2])
    @test ustrip(dd) < 3.5  # relaxed closer than initial

    # Test recording trajectory
    cell = _h2_cell(10, 1.5)
    calc = NNCalc(cell, cf, nnitf)
    res = relax!(calc; keep_trajectory=true)
    @test length(res.relax.trajectory) > 1
end

@testset "Opt" begin
    cell = _h2_cell()
    calc = lj_like_calc(cell; rc=6.0)
    traj = []
    @test converged(opt_tpsd(calc, trajectory=traj), calc)
    @test length(traj) > 1
    @test maximum(norm.(eachcol(eddp_forces(calc)))) < 1e-4
    @test maximum(eddp_stress(calc)) < 1e-4

    cell = _h2_cell()
    calc = lj_like_calc(cell; rc=6.0)
    p = 1e-2
    opt = opt_tpsd(calc; relax_cell=true, ext_pressure=diagm([p, p, p]), s_tol=1e-3)
    @test converged(opt, calc)
    @test maximum(abs.(eddp_forces(calc))) < 1e-3
    @test maximum(abs.(eddp_stress(calc) .- diagm([p, p, p]))) < 1e-3
end

@testset "LJ Baseline" begin
    function _two_atom_cell(species_pair, distance; lattice=12.0)
        pos = Float64[
            0.0 distance
            0.0 0.0
            0.0 0.0
        ]
        Cell(Lattice(lattice, lattice, lattice), species_pair, pos)
    end

    function _test_baseline_forces_fd(calc, amp=1e-7, atol=1e-5)
        ftmp = copy(eddp_forces(calc))
        etmp = eddp_energy(calc)
        positions(get_cell(calc))[1] += amp
        tmp = (eddp_energy(calc) - etmp) / amp
        @test -tmp ≈ ftmp[1] atol = atol
    end

    function _pair_feature_slice(elements, si, sj)
        elems = sort(unique(map(Symbol, collect(elements))))
        ne = length(elems)
        i = findfirst(==(si), elems)
        j = findfirst(==(sj), elems)
        offset = ((i - 1) * ne + (j - 1)) * 4
        start = ne + 1 + offset
        start:start + 3
    end

    @testset "Analytic Energy" begin
        sigma = 1.1
        epsilon = 0.7
        rcut = 3.0
        r = 1.35
        cell = _two_atom_cell([:Ar, :Ar], r)
        calc = lj_baseline_calc(cell; sigma, epsilon, rcut)

        @test eddp_energy(calc) ≈ _force_shifted_lj_energy(r, sigma, epsilon, rcut) atol = 1e-10
        _test_baseline_forces_fd(calc, 1e-7, 1e-5)
    end

    @testset "Beyond Cutoff" begin
        cell = _two_atom_cell([:Ar, :Ar], 4.5)
        calc = lj_baseline_calc(cell; sigma=1.0, epsilon=1.0, rcut=3.0)

        @test eddp_energy(calc) ≈ 0.0 atol = 1e-12
        @test maximum(abs.(eddp_forces(calc))) ≈ 0.0 atol = 1e-12
        @test maximum(abs.(eddp_stress(calc))) ≈ 0.0 atol = 1e-12
    end

    @testset "Feature Layout And Symmetric Lookup" begin
        sigma = Dict((:H, :H) => 1.0, (:H, :O) => 1.25, (:O, :O) => 1.4)
        epsilon = Dict((:H, :H) => 0.4, (:H, :O) => 0.9, (:O, :O) => 1.1)
        rcut = 4.0
        cf = lj_baseline_feature([:H, :O]; rcut)
        weights = lj_baseline_weights([:H, :O]; sigma, epsilon, rcut)
        ho_slice = _pair_feature_slice(cf.elements, :H, :O)
        oh_slice = _pair_feature_slice(cf.elements, :O, :H)

        @test feature_size(cf) == (2, 16, 0)
        @test EDDPotentials.EDDPotentialsCore.nbodyfeatures(cf, 2) == 16
        @test EDDPotentials.EDDPotentialsCore.nbodyfeatures(cf, 3) == 0
        @test weights[1:2] == [0.0, 0.0]
        @test weights[ho_slice] ≈ weights[oh_slice]

        r = 1.8
        cell = _two_atom_cell([:H, :O], r)
        calc = lj_baseline_calc(cell; sigma, epsilon, rcut)
        @test eddp_energy(calc) ≈ _force_shifted_lj_energy(r, sigma[(:H, :O)], epsilon[(:H, :O)], rcut) atol =
              1e-10
    end

    @testset "Neighbour List Skin" begin
        cell = _two_atom_cell([:Ar, :Ar], 1.4)
        calc = lj_baseline_calc(cell; sigma=1.0, epsilon=1.0, rcut=3.0)
        nnc = calc isa EDDPCalculator ? calc._nncalc : calc
        p1 = nnc.last_nn_build_pos[1]
        positions(get_cell(calc))[1] += 0.001
        calculate!(calc; rebuild_nl=false)
        @test p1 == nnc.last_nn_build_pos[1]
    end

    @testset "Reject Conflicting Reverse Keys" begin
        sigma = Dict((:H, :O) => 1.25, (:O, :H) => 1.35)
        epsilon = Dict((:H, :O) => 0.9)
        @test_throws ArgumentError lj_baseline_weights([:H, :O]; sigma, epsilon, rcut=4.0)
    end

    @testset "LJ in 1 extra dimension" begin
        sigma = 1.1
        epsilon = 0.7
        rcut = 3.0
        r_3d = 1.35

        function _make_lj_4d_sym(extra_val=0.0)
            pos_4d = Float64[
                0.0 r_3d
                0.0 0.0
                0.0 0.0
                extra_val extra_val
            ]
            Cell(Lattice(12.0, 12.0, 12.0), [:Ar, :Ar], pos_4d)
        end

        @testset "Energy invariance when extra dims at zero" begin
            cell4 = _make_lj_4d_sym(0.0)
            calc4 = lj_baseline_calc(cell4; sigma, epsilon, rcut)
            cell3 = Cell(Lattice(12, 12, 12), [:Ar, :Ar], cell4.positions[1:3, :])
            calc3 = lj_baseline_calc(cell3; sigma, epsilon, rcut)
            @test eddp_energy(calc4) ≈ eddp_energy(calc3) atol = 1e-10
            @test eddp_energy(calc3) ≈ _force_shifted_lj_energy(r_3d, sigma, epsilon, rcut) atol =
                1e-10
        end

        @testset "Energy differs when extra dims change pair distance" begin
            # Different extra-dim values for the two atoms so 4D distance ≠ 3D distance
            pos_4d = Float64[
                0.0 r_3d
                0.0 0.0
                0.0 0.0
                0.1 0.5
            ]
            cell4 = Cell(Lattice(12.0, 12.0, 12.0), [:Ar, :Ar], pos_4d)
            calc4 = lj_baseline_calc(cell4; sigma, epsilon, rcut)
            calc3 = lj_baseline_calc(Cell(Lattice(12, 12, 12), [:Ar, :Ar], pos_4d[1:3, :]);
                sigma, epsilon, rcut)
            @test eddp_energy(calc4) ≠ eddp_energy(calc3)
        end

        @testset "Forces FD in 4D" begin
            pos_4d = Float64[
                0.0 r_3d
                0.0 0.0
                0.0 0.0
                0.1 0.4
            ]
            cell4 = Cell(Lattice(12, 12, 12), [:Ar, :Ar], pos_4d)
            calc = lj_baseline_calc(cell4; sigma, epsilon, rcut)
            ftmp = eddp_forces(calc)
            @test size(ftmp, 1) == 4
            etmp = eddp_energy(calc)
            for d in 1:4, a in 1:2
                p0 = positions(get_cell(calc))[d, a]
                positions(get_cell(calc))[d, a] = p0 + 1e-6
                ep = eddp_energy(calc)
                positions(get_cell(calc))[d, a] = p0 - 1e-6
                em = eddp_energy(calc)
                positions(get_cell(calc))[d, a] = p0
                fd = -(ep - em) / 2e-6
                @test fd ≈ ftmp[d, a] atol = 1e-5
            end
        end

        @testset "Penalty modifier: energy and forces" begin
            alpha = 0.1
            cell4 = _make_lj_4d_sym(0.2)
            source = lj_baseline_calc(cell4; sigma, epsilon, rcut)
            mod = ExtraDimensionPenaltyModifier(alpha; start_dim=4)
            mc = MetaCalculator(source, [mod])

            E_raw = eddp_energy(source)
            E_meta = eddp_energy(mc)
            expected_penalty = alpha * sum(cell4.positions[4, :] .^ 2)
            @test E_meta ≈ E_raw + expected_penalty atol = 1e-10

            F_raw = eddp_forces(source)
            F_meta = eddp_forces(mc)
            @test F_meta[1:3, :] ≈ F_raw[1:3, :] atol = 1e-10
            @test F_meta[4, :] ≈ F_raw[4, :] .- 2 * alpha .* cell4.positions[4, :] atol =
                1e-10
        end

        @testset "Penalty drives extra dims toward zero under relaxation" begin
            cell4 = _make_lj_4d_sym(0.3)
            source = lj_baseline_calc(cell4; sigma, epsilon, rcut)
            mc = MetaCalculator(
                source,
                [ExtraDimensionPenaltyModifier(10.0; start_dim=4)],
            )
            opt_tpsd(mc; itermax=50, f_tol=1e-10)
            final_extra = positions(get_cell(mc))[4, :]
            @test maximum(abs.(final_extra)) < 0.01
        end
    end
end
