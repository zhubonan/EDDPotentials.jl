using EDDPotentials
using EDDPotentials:
    CellFeature,
    RandomEmbeddedCellFeature,
    FeatureContainer,
    StructureContainer,
    TwoBodyFeature,
    VariableCellCalc,
    NNCalc,
    get_cell,
    nfeatures,
    feature_vector,
    eddp_energy,
    eddp_enthalpy,
    eddp_forces,
    eddp_stress,
    train!,
    split,
    ginit,
    fd_desc,
    stress_gradient
using EDDPotentials.EDDPotentialsCore:
    ManualFluxBackPropInterface,
    LinearInterface,
    flux_mlp_model,
    _need_calc,
    _contract_full_feature_matrix
using Test
using Flux
using CellBase: positions, species

@testset "Random Embedded Feature" begin
    @testset "Descriptor Equivalence" begin
        cell = _h2o_cell()
        base = CellFeature([:H, :O], p2=2:4, p3=2:3, q3=2:3)
        embedded = RandomEmbeddedCellFeature(base; nembed=3, seed=7)

        direct = feature_vector(embedded, cell)
        full = feature_vector(base, cell)
        reference = _contract_full_feature_matrix(embedded, full)

        @test size(direct) == size(reference)
        @test allclose(direct, reference, atol=1e-10)
        @test nfeatures(embedded) == 3 + 3 * 9 + 4 * 18

        embedded2 = RandomEmbeddedCellFeature(base; nembed=3, seed=7)
        embedded3 = RandomEmbeddedCellFeature(base; nembed=3, seed=8)
        @test embedded.embedding == embedded2.embedding
        @test embedded.embedding != embedded3.embedding
    end

    @testset "Ternary Descriptor Equivalence" begin
        cell = _lco_cell()
        base = CellFeature([:Li, :Co, :O], p2=[2], p3=[2], q3=[2])
        embedded = RandomEmbeddedCellFeature(base; nembed=2, seed=3)

        direct = feature_vector(embedded, cell)
        full = feature_vector(base, cell)
        reference = _contract_full_feature_matrix(embedded, full)

        @test size(direct) == size(reference)
        @test allclose(direct, reference, atol=1e-10)
    end

    @testset "Ten-Element Descriptor Equivalence" begin
        elems = [:H, :He, :Li, :Be, :B, :C, :N, :O, :F, :Ne]
        pos = hcat([[0.7 * (i - 1), 0.2 * (-1)^i, 0.1 * (i % 3)] for i = 1:length(elems)]...)
        cell = Cell(Lattice(20.0, 20.0, 20.0), elems, pos)

        base = CellFeature(elems, p2=[2], p3=[2], q3=[2])
        embedded = RandomEmbeddedCellFeature(base; nembed=4, seed=13)

        direct = feature_vector(embedded, cell)
        full = feature_vector(base, cell)
        reference = _contract_full_feature_matrix(embedded, full)

        @test size(direct) == size(reference)
        @test allclose(direct, reference, atol=1e-10)
        @test EDDPotentials.feature_size(base) == (10, 100, 550)
        @test EDDPotentials.feature_size(embedded) == (4, 16, 40)
    end

    @testset "Descriptor Gradient Finite Difference" begin
        cell = _h2o_cell()
        base = CellFeature([:H, :O], p2=2:2, p3=2:2, q3=2:2)
        embedded = RandomEmbeddedCellFeature(base; nembed=3, seed=17)

        diff, gvec_ji = fd_desc(embedded, cell)
        @test maximum(abs.(diff - gvec_ji)) < 1e-7

        diff_strain, virial = stress_gradient(embedded, cell)
        @test maximum(abs.(diff_strain + virial)) < 1e-4
    end

    @testset "Multi-Species Descriptor Gradient Finite Difference" begin
        elems = [:H, :He, :Li, :Be, :B, :C, :N, :O, :F, :Ne]
        pos = hcat([[0.7 * (i - 1), 0.2 * (-1)^i, 0.1 * (i % 3)] for i = 1:length(elems)]...)
        cell = Cell(Lattice(20.0, 20.0, 20.0), elems, pos)
        base = CellFeature(elems, p2=2:2, p3=2:2, q3=2:2)
        embedded = RandomEmbeddedCellFeature(base; nembed=4, seed=19)

        diff, gvec_ji = fd_desc(embedded, cell)
        @test maximum(abs.(diff - gvec_ji)) < 1e-7

        diff_strain, virial = stress_gradient(embedded, cell)
        @test maximum(abs.(diff_strain + virial)) < 1e-4
    end

    @testset "Validation" begin
        bad_cf = CellFeature(
            [:H, :O],
            (
                TwoBodyFeature([2], (:H, :H), 4.0),
                TwoBodyFeature([2, 3], (:H, :O), 4.0),
            ),
            (),
        )
        @test_throws ErrorException RandomEmbeddedCellFeature(bad_cf; nembed=2, seed=1)
    end

    @testset "Calculator Integration" begin
        cell = _h2_cell()
        base = CellFeature(unique(species(cell)), p2=[2], q3=[2, 3], p3=[2, 3])
        embedded = RandomEmbeddedCellFeature(base; nembed=2, seed=5)

        function _test_forces_fd(calc, amp=1e-7, atol=1e-5)
            ftmp = copy(eddp_forces(calc))
            etmp = eddp_energy(calc)
            positions(get_cell(calc))[1] += amp
            @test _need_calc(calc, true)
            tmp = (eddp_energy(calc) - etmp) / amp
            @test -tmp ≈ ftmp[1] atol = atol
        end

        function _test_forces_fd_vc(calc; amp=1e-7, atol=1e-5, idx=1)
            if !isa(calc, VariableCellCalc)
                calc = VariableCellCalc(calc)
            end
            ftmp = copy(eddp_forces(calc))
            etmp = eddp_enthalpy(calc)
            positions(get_cell(calc))[idx] += amp
            @test _need_calc(calc, true)
            tmp = (eddp_enthalpy(calc) - etmp) / amp
            @test -tmp ≈ ftmp[idx] atol = atol
        end

        lin = LinearInterface(rand(nfeatures(embedded)))
        calc = NNCalc(cell, embedded, lin; core=nothing)
        @test isa(eddp_energy(calc), Float64)
        @test all(isapprox.(sum(eddp_forces(calc), dims=2), 0.0, atol=1e-10))
        @test size(eddp_stress(calc)) == (3, 3)
        _test_forces_fd(calc)
        _test_forces_fd_vc(calc)

        mbp = ManualFluxBackPropInterface(embedded, 5)
        calc2 = NNCalc(cell, embedded, mbp; core=nothing)
        @test isa(eddp_energy(calc2), Float64)
        @test all(isapprox.(sum(eddp_forces(calc2), dims=2), 0.0, atol=1e-10))
        _test_forces_fd(calc2)

        model = flux_mlp_model(embedded, 5)
        @test size(model(rand(nfeatures(embedded), 2))) == (1, 2)
    end

    @testset "Multi-Species Finite Difference" begin
        cell = _lco_cell()
        base = CellFeature(unique(species(cell)), p2=[2], q3=[2, 3], p3=[2, 3])
        embedded = RandomEmbeddedCellFeature(base; nembed=2, seed=11)

        function _test_forces_fd(calc, amp=1e-7, atol=1e-5)
            ftmp = copy(eddp_forces(calc))
            etmp = eddp_energy(calc)
            positions(get_cell(calc))[1] += amp
            @test _need_calc(calc, true)
            tmp = (eddp_energy(calc) - etmp) / amp
            @test -tmp ≈ ftmp[1] atol = atol
        end

        function _test_forces_fd_vc(calc; amp=1e-7, atol=1e-5, idx=1)
            if !isa(calc, VariableCellCalc)
                calc = VariableCellCalc(calc)
            end
            ftmp = copy(eddp_forces(calc))
            etmp = eddp_enthalpy(calc)
            positions(get_cell(calc))[idx] += amp
            @test _need_calc(calc, true)
            tmp = (eddp_enthalpy(calc) - etmp) / amp
            @test -tmp ≈ ftmp[idx] atol = atol
        end

        lin = LinearInterface(rand(nfeatures(embedded)))
        calc = NNCalc(cell, embedded, lin; core=nothing)
        _test_forces_fd(calc)
        _test_forces_fd_vc(calc)

        mbp = ManualFluxBackPropInterface(embedded, 5)
        calc2 = NNCalc(_lco_cell(), embedded, mbp; core=nothing)
        _test_forces_fd(calc2)
    end

    @testset "FeatureContainer and Serialization" begin
        cells = [_h2o_cell(4.0, s) for s in (0.95, 1.0, 1.05, 1.1)]
        energies = [0.1, 0.2, 0.25, 0.35]
        sc = StructureContainer(cells, energies)
        embedded = RandomEmbeddedCellFeature([:H, :O]; nembed=2, seed=9, p2=2:3, p3=2:2, q3=2:2)

        fc = FeatureContainer(sc, embedded; show_progress=false)
        @test length(fc.fvecs) == length(sc)
        @test size(fc.fvecs[1], 1) == nfeatures(embedded)

        fc_train, fc_test = split(fc, 3, 1; standardize=true, apply_transform=true)
        model = ManualFluxBackPropInterface(
            Chain(Dense(ginit(4, nfeatures(embedded))), Dense(ginit(1, 4))),
            xt=fc_train.xt,
            yt=fc_train.yt,
            apply_xt=false,
        )
        train!(model, fc_train, fc_test; maxIter=2, show_progress=false, earlystop=0)
        @test isfinite(sum(model(fc_train[1][1])))

        mktempdir() do dir
            path = joinpath(dir, "embedded_fc.jld2")
            EDDPotentials.save_fc(fc, path)
            fc_loaded = EDDPotentials.load_fc(path)
            @test fc_loaded.feature == embedded
            @test allclose(fc_loaded.fvecs[1], fc.fvecs[1], atol=1e-12)
        end
    end
end
