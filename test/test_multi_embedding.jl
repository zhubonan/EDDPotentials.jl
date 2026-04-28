using EDDPotentials
using EDDPotentials:
    RandomEmbeddedCellFeature,
    MultiRandomEmbeddedCellFeature,
    FeatureStore,
    StructureContainer,
    MetaCalculator,
    CellRelaxationModifier,
    is_variable_cell,
    NNCalc,
    get_cell,
    nfeatures,
    feature_vector,
    feature_size,
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
    embedding_size,
    grad_embedding!,
    recompute_features!,
    get_workspace,
    suggest_rcut,
    paramvector,
    nparams,
    forward!,
    backward!,
    gradinp!,
    setparamvector!,
    gradparam!,
    setup_fj_with_embedding
using Test
using Flux
using CellBase: positions, species

@testset "Multi-Embedding" begin

    @testset "Constructor" begin
        mcf = MultiRandomEmbeddedCellFeature(
            [:H, :O];
            nembed=3, n_multi=2, seeds=[7, 42],
            p2=2:4, p3=2:3, q3=2:3,
        )
        @test length(mcf.embeddings) == 2
        @test mcf.embeddings[1].embedding != mcf.embeddings[2].embedding
        @test sort(mcf.embeddings[1].elements) == sort(mcf.embeddings[2].elements)

        # Seed reproducibility
        mcf2 = MultiRandomEmbeddedCellFeature(
            [:H, :O]; nembed=3, n_multi=2, seeds=[7, 42], p2=2:4, p3=2:3, q3=2:3,
        )
        @test mcf == mcf2

        # Validation: wrong seed count
        @test_throws ArgumentError MultiRandomEmbeddedCellFeature(
            [:H, :O]; nembed=3, n_multi=3, seeds=[1, 2],
        )
    end

    @testset "Feature size" begin
        single = RandomEmbeddedCellFeature([:H, :O]; nembed=3, seed=7, p2=2:4, p3=2:3, q3=2:3)
        mcf = MultiRandomEmbeddedCellFeature([:H, :O]; nembed=3, n_multi=3, seeds=[7, 8, 9], p2=2:4, p3=2:3, q3=2:3)

        s1 = feature_size(single)
        sm = feature_size(mcf)
        @test sm == (3 * s1[1], 3 * s1[2], 3 * s1[3])
        @test nfeatures(mcf) == 3 * nfeatures(single)
    end

    @testset "Feature vector correctness" begin
        cell = _h2o_cell()

        emb1 = RandomEmbeddedCellFeature([:H, :O]; nembed=3, seed=7, p2=2:3, p3=2:2, q3=2:2)
        emb2 = RandomEmbeddedCellFeature([:H, :O]; nembed=3, seed=42, p2=2:3, p3=2:2, q3=2:2)
        mcf = MultiRandomEmbeddedCellFeature([emb1, emb2])

        # Individual feature vectors
        fv1 = feature_vector(emb1, cell)
        fv2 = feature_vector(emb2, cell)

        # Multi-embedding feature vector
        fv_m = feature_vector(mcf, cell)

        # Layout: [emb1_all | emb2_all] (simple concatenation)
        nf1 = nfeatures(emb1)
        @test size(fv_m) == (nf1 + nfeatures(emb2), length(cell))

        # Emb1 block: [emb1_ob | emb1_2b | emb1_3b]
        @test fv_m[1:nf1, :] ≈ fv1
        # Emb2 block: [emb2_ob | emb2_2b | emb2_3b]
        @test fv_m[nf1+1:end, :] ≈ fv2
    end

    @testset "Force finite difference" begin
        cell = _h2_cell()
        mcf = MultiRandomEmbeddedCellFeature(
            unique(species(cell));
            nembed=2, n_multi=2, seeds=[5, 99],
            p2=[2], q3=[2, 3], p3=[2, 3],
        )

        function _test_forces_fd(calc, amp=1e-7, atol=1e-5)
            ftmp = copy(eddp_forces(calc))
            etmp = eddp_energy(calc)
            positions(get_cell(calc))[1] += amp
            @test _need_calc(calc, true)
            tmp = (eddp_energy(calc) - etmp) / amp
            @test -tmp ≈ ftmp[1] atol = atol
        end

        function _test_forces_fd_vc(calc; amp=1e-7, atol=1e-5, idx=1)
            if !is_variable_cell(calc)
                calc = MetaCalculator(calc, [CellRelaxationModifier(get_cell(calc))])
            end
            ftmp = copy(eddp_forces(calc))
            etmp = eddp_enthalpy(calc)
            pos = copy(get_positions(calc))
            pos[idx] += amp
            set_positions!(calc, pos)
            @test calc.energy_dirty
            tmp = (eddp_enthalpy(calc) - etmp) / amp
            @test -tmp ≈ ftmp[idx] atol = atol
        end

        # Test with LinearInterface
        lin = LinearInterface(rand(nfeatures(mcf)))
        calc = NNCalc(cell, mcf, lin; core=nothing)
        @test isa(eddp_energy(calc), Float64)
        @test all(isapprox.(sum(eddp_forces(calc), dims=2), 0.0, atol=1e-10))
        @test size(eddp_stress(calc)) == (3, 3)
        _test_forces_fd(calc)
        _test_forces_fd_vc(calc)

        # Test with ManualFluxBackPropInterface
        mbp = ManualFluxBackPropInterface(mcf, 5)
        calc2 = NNCalc(_h2_cell(), mcf, mbp; core=nothing)
        @test isa(eddp_energy(calc2), Float64)
        @test all(isapprox.(sum(eddp_forces(calc2), dims=2), 0.0, atol=1e-10))
        _test_forces_fd(calc2)
    end

    @testset "Descriptor gradient finite difference" begin
        cell = _h2o_cell()
        mcf = MultiRandomEmbeddedCellFeature(
            [:H, :O]; nembed=3, n_multi=2, seeds=[17, 31],
            p2=2:2, p3=2:2, q3=2:2,
        )

        diff, gvec_ji = fd_desc(mcf, cell)
        @test maximum(abs.(diff - gvec_ji)) < 1e-7

        diff_strain, virial = stress_gradient(mcf, cell)
        @test maximum(abs.(diff_strain + virial)) < 1e-4
    end

    @testset "FeatureStore integration" begin
        cells = [_h2o_cell(4.0, s) for s in (0.95, 1.0, 1.05, 1.1)]
        energies = [0.1, 0.2, 0.25, 0.35]
        sc = StructureContainer(cells, energies)
        mcf = MultiRandomEmbeddedCellFeature(
            [:H, :O]; nembed=2, n_multi=2, seeds=[9, 10], p2=2:3, p3=2:2, q3=2:2,
        )

        fc = FeatureStore(sc, mcf; show_progress=false)
        @test length(fc) == length(sc)
        @test size(fc[1], 1) == nfeatures(mcf)

        fc_train, fc_val = split(fc, 3, 1; standardize=true)
        model = ManualFluxBackPropInterface(
            Chain(Dense(ginit(4, nfeatures(mcf))), Dense(ginit(1, 4))),
            xt=fc_train.xt,
            yt=fc_train.yt,
        )
        train!(model, fc_train, fc_val; maxIter=2, show_progress=false, earlystop=0)
        @test isfinite(sum(model(fc_train[1])))

        mktempdir() do dir
            path = joinpath(dir, "multi_emb_fc.jld2")
            EDDPotentials.save_fc(fc, path)
            fc_loaded = EDDPotentials.load_fc(path)
            @test fc_loaded.feature == mcf
            @test allclose(fc_loaded[1], fc[1], atol=1e-12)
        end
    end

    @testset "JSON serialization round-trip" begin
        using EDDPotentials.EDDPotentialsCore: save_model_artifact, load_model_artifact

        mcf = MultiRandomEmbeddedCellFeature(
            [:H, :O]; nembed=2, n_multi=2, seeds=[5, 6], p2=[2], p3=[2], q3=[2],
        )
        model = ManualFluxBackPropInterface(mcf, 3)

        mktempdir() do dir
            path = joinpath(dir, "model.json")
            save_model_artifact(path, model; feature_spec=mcf)
            artifact = load_model_artifact(path)
            @test artifact.feature_spec == mcf
            @test isfinite(sum(forward!(artifact.model, rand(nfeatures(mcf), 3))))
        end
    end

    @testset "M=1 equivalence with single embedding" begin
        cell = _h2o_cell()
        single = RandomEmbeddedCellFeature([:H, :O]; nembed=3, seed=7, p2=2:3, p3=2:2, q3=2:2)
        mcf = MultiRandomEmbeddedCellFeature([:H, :O]; nembed=3, n_multi=1, seeds=[7], p2=2:3, p3=2:2, q3=2:2)

        fv_single = feature_vector(single, cell)
        fv_multi = feature_vector(mcf, cell)
        @test fv_single ≈ fv_multi
        @test nfeatures(single) == nfeatures(mcf)
    end

    @testset "Large element set" begin
        elems = Symbol.("E" .* string.(1:20))
        mcf = MultiRandomEmbeddedCellFeature(
            elems; nembed=2, n_multi=3, seeds=[1, 2, 3], p2=[2], p3=[2], q3=[2],
        )
        single = RandomEmbeddedCellFeature(elems; nembed=2, seed=1, p2=[2], p3=[2], q3=[2])
        @test nfeatures(mcf) == 3 * nfeatures(single)

        # Feature sizes should be element-count independent
        s = feature_size(mcf)
        @test s == (3 * 2, 3 * 4, 3 * 6)
    end

end
