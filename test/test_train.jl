using EDDPotentials
using EDDPotentials:
    ginit,
    compute_objectives,
    compute_objectives_diff,
    levenberg_marquardt,
    train_lm!,
    train_lm_batched!,
    levenberg_marquardt_batched,
    StructureContainer,
    CellFeature,
    FeatureStore,
    nfeatures,
    reinit,
    train!,
    create_ensemble,
    predict_energy,
    TrainedModel,
    train_model,
    predict,
    EDDPCalculator,
    save_model_artifact,
    load_model_artifact,
    load_model
using EDDPotentials.EDDPotentialsCore:
    ManualFluxBackPropInterface,
    LinearInterface,
    AbstractNNInterface,
    FluxInterface,
    nparams,
    setup_fj,
    setup_fj_batched,
    paramvector

using Test
using Flux
using StatsBase
using NLSolversBase

@testset "Training" begin
    @testset "Tools" begin
        nf = 1000
        data = vcat([rand(nf, 3) for _ = 1:10], [rand(nf, 2) for _ = 1:10])

        f = zeros(length(data))
        y = rand(length(data))
        yt = StatsBase.fit(StatsBase.ZScoreTransform, reshape(y, 1, length(y)), dims=2)

        model = ManualFluxBackPropInterface(
            Chain(Dense(rand(1, nf), rand(1), tanh)),
            yt=yt,
        )
        compute_objectives(f, model, data, f)
        @test any(f != 0.0)

        jmat = rand(length(data), nparams(model))
        compute_objectives_diff(f, jmat, model, data, y)

        @test any(jmat != 0.0)

        # Setting up and minimising
        f!, j!, fj! = setup_fj(model, data, y)
        p0 = paramvector(model)

        # Manual diff
        jmat = zeros(length(y), length(p0))
        j!(jmat, p0)
        dp = 1e-4
        out = similar(y)
        f!(out, p0)
        p0[1] += dp
        out2 = similar(y)
        f!(out2, p0)
        val = (out2[1] - out[1]) / dp
        @test abs(val - jmat[1]) < 1e-7

        # Optimisation 
        od2 = OnceDifferentiable(
            f!,
            j!,
            fj!,
            p0,
            zeros(eltype(data[1]), length(data)),
            inplace=true,
        )
        opt_res = levenberg_marquardt(od2, p0; show_trace=false)

        j!(jmat, opt_res.minimizer)
        @test all(isapprox.(jmat, 0.0, atol=1e-6))


        # Test with training config
        model =
            ManualFluxBackPropInterface(Chain(Dense(rand(1, nf), rand(1))))

        train_lm!(model, data, y;)
        out = model(data[2])
        # Check we have successfully performed the fit
        @test sum(out) ≈ y[2] atol = 1e-5

        # Highly overfitted linear case....
        nf = 1000
        data = vcat([rand(nf, 3) for _ = 1:10], [rand(nf, 2) for _ = 1:10])
        y = size.(data, 2) ./ 2
        model = LinearInterface(rand(1, nf))

        opt_res, _, _ = train_lm!(model, data, y; earlystop=0)
    end

    @testset "Ensemble" begin
        path = joinpath(datadir, "training/*.res")
        path = relpath(path, pwd())
        sc = StructureContainer([path])
        cf = CellFeature([:B])
        fc = FeatureStore(sc, cf)

        # This gives fix examples
        fc_train, fc_val = split(fc, 5, 5; standardize=true)
        models = []
        model = ManualFluxBackPropInterface(
            Chain(Dense(ginit(5, nfeatures(fc.feature))), Dense(ginit(1, 5))),
            xt=fc_train.xt,
            yt=fc_train.yt,
        )
        model_ = reinit(model)
        @test any(paramvector(model_) .!= paramvector(model))
        for _ = 1:10
            model_ = reinit(model)
            res = train!(model_, fc_train, fc_val)
            push!(models, model_)
        end

        emod = create_ensemble(models, fc_train)
        out = predict_energy(emod, fc_train[1])
        @test isa(out, Real)

    end


    @testset "Flux" begin
        nf = 1000
        data = vcat([rand(nf, 3) for _ = 1:10], [rand(nf, 2) for _ = 1:10])

        f = zeros(length(data))
        y = rand(length(data))
        yt = StatsBase.fit(StatsBase.ZScoreTransform, reshape(y, 1, length(y)), dims=2)

        model = FluxInterface(Chain(Dense(rand(1, nf), rand(1))))
        compute_objectives(f, model, data, f)
        @test any(f != 0.0)

        jmat = rand(length(data), nparams(model))
        compute_objectives_diff(f, jmat, model, data, y)
        @test any(jmat != 0.0)

        # Setting up and minimising
        f!, j!, fj! = setup_fj(model, data, y)
        p0 = paramvector(model)

        # Manual finite diff
        jmat = zeros(length(y), length(p0))
        j!(jmat, p0)
        dp = 1e-4
        out = similar(y)
        f!(out, p0)
        p0[1] += dp
        out2 = similar(y)
        f!(out2, p0)
        val = (out2[1] - out[1]) / dp
        @test abs(val - jmat[1]) < 1e-7

        # Optimisation 
        od2 = OnceDifferentiable(
            f!,
            j!,
            fj!,
            p0,
            zeros(eltype(data[1]), length(data)),
            inplace=true,
        )
        opt_res = levenberg_marquardt(od2, p0; show_trace=false)
        #@test_broken opt_res.g_converged
        @test opt_res.x_converged
    end

    @testset "Batched LM" begin
        nf = 1000
        data = vcat([rand(nf, 3) for _ = 1:10], [rand(nf, 2) for _ = 1:10])
        y = rand(length(data))

        # Test setup_fj_batched with a linear model (no saturation issues)
        model = ManualFluxBackPropInterface(
            Chain(Dense(rand(1, nf), rand(1))),
        )
        fj_batch!, f_full! = setup_fj_batched(model, data, y)

        batch_idx = [1, 3, 5, 7]
        B = length(batch_idx)
        f_b = zeros(B)
        j_b = zeros(B, nparams(model))
        fj_batch!(f_b, j_b, paramvector(model), batch_idx)
        @test any(f_b .!= 0.0)
        @test any(j_b .!= 0.0)

        # Compare batch results against full-dataset computation
        f_full = zeros(length(data))
        compute_objectives(f_full, model, data, y)
        for (li, gi) in enumerate(batch_idx)
            @test abs(f_b[li] - f_full[gi]) < 1e-10
        end
    end

    @testset "train! dispatch lm_batched" begin
        path = joinpath(datadir, "training/*.res")
        path = relpath(path, pwd())
        sc = StructureContainer([path])
        cf = CellFeature([:B])
        fc = FeatureStore(sc, cf)
        fc_train, fc_val = split(fc, 5, 5; standardize=true)

        model = ManualFluxBackPropInterface(
            Chain(Dense(ginit(5, nfeatures(cf))), Dense(ginit(1, 5))),
            xt=fc_train.xt,
            yt=fc_train.yt,
        )
        # batch_size must be >= nparams for well-conditioned normal equations
        # Use batch_size = n_data to effectively test the dispatch path
        opt_res, params, history, funcs = train!(
            model, fc_train, fc_val;
            train_method=:lm_batched,
            batch_size=length(fc_train),
            max_epochs=100,
            earlystop=10,
        )
        @test length(params) > 0
        # history is only populated when callback is active (earlystop > 0)
        @test size(history, 1) >= 1
    end

    @testset "train_model" begin
        path = joinpath(datadir, "training/*.res")
        path = relpath(path, pwd())
        cf = CellFeature([:B])

        @testset "from paths with explicit spec" begin
            tm = train_model(path, cf;
                val_ratio=0.2, test_ratio=0.2, nnodes=[5], max_epochs=50, earlystop=5)
            @test tm isa TrainedModel
            @test tm.model isa ManualFluxBackPropInterface
            @test tm.feature_spec == cf
            @test tm.val_results isa TrainingResults
            @test tm.test_results isa TrainingResults
            @test length(tm.test_results.fc) > 0
            @test rmse_per_atom(tm.train_results) < 1.0
        end

        @testset "from StructureContainer" begin
            sc = StructureContainer([path])
            tm = train_model(sc, cf; val_ratio=0.2, test_ratio=0.2, nnodes=[5], max_epochs=50, earlystop=5)
            @test tm isa TrainedModel
            @test tm.val_results isa TrainingResults
            @test tm.test_results isa TrainingResults
        end

        @testset "from FeatureStore" begin
            sc = StructureContainer([path])
            fc = FeatureStore(sc, cf)
            tm = train_model(fc; val_ratio=0.2, test_ratio=0.2, nnodes=[5], max_epochs=50, earlystop=5)
            @test tm isa TrainedModel
            @test tm.val_results isa TrainingResults
            @test tm.test_results isa TrainingResults
            @test occursin("Average Spearman", sprint(show, tm.test_results))
        end

        @testset "predict on TrainedModel" begin
            tm = train_model(path, cf; val_ratio=0.2, test_ratio=0.2, nnodes=[5], max_epochs=50, earlystop=5)
            sc = StructureContainer([path])
            cell = sc.structures[1]
            result = predict(tm, cell)
            @test haskey(result, :energy)
            @test haskey(result, :forces)
            @test haskey(result, :stress)
            @test result.energy isa Real
            result_min = predict(tm, cell; forces=false, stress=false)
            @test !haskey(result_min, :forces)
            @test !haskey(result_min, :stress)
        end

        @testset "predict on EDDPCalculator" begin
            tm = train_model(path, cf; val_ratio=0.2, test_ratio=0.2, nnodes=[5], max_epochs=50, earlystop=5)
            calc = EDDPCalculator(tm)
            sc = StructureContainer([path])
            cell = sc.structures[1]
            result = predict(calc, cell)
            @test haskey(result, :energy)
            @test haskey(result, :forces)
            @test result.energy isa Real
        end

        @testset "save/load round-trip" begin
            tm = train_model(path, cf; val_ratio=0.2, test_ratio=0.2, nnodes=[5], max_epochs=50, earlystop=5)
            mktempdir() do dir
                p = joinpath(dir, "tm.json")
                save_model_artifact(p, tm)
                artifact = load_model_artifact(p)
                calc = EDDPCalculator(artifact)
                sc = StructureContainer([path])
                cell = sc.structures[1]
                energy_before = predict(tm, cell; forces=false, stress=false).energy
                energy_after = predict(calc, cell; forces=false, stress=false).energy
                @test energy_before ≈ energy_after
            end
        end

        @testset "val_ratio=0 throws" begin
            @test_throws ArgumentError train_model(path, cf; val_ratio=0.0)
        end

        @testset "test_ratio=0 throws" begin
            @test_throws ArgumentError train_model(path, cf; test_ratio=0.0)
        end

        @testset "tiny ratio yields zero structures throws" begin
            @test_throws ArgumentError train_model(path, cf; val_ratio=0.01)
            @test_throws ArgumentError train_model(path, cf; test_ratio=0.01)
        end

        @testset "pressure_gpa passthrough" begin
            tm = train_model(path, cf; val_ratio=0.2, test_ratio=0.2, nnodes=[5], max_epochs=50,
                             earlystop=5, pressure_gpa=2.0)
            @test tm isa TrainedModel
        end

        @testset "show TrainedModel" begin
            tm = train_model(path, cf; val_ratio=0.2, test_ratio=0.2, nnodes=[3], max_epochs=20, earlystop=5)
            io = IOBuffer()
            show(io, MIME("text/plain"), tm)
            s = String(take!(io))
            @test occursin("TrainedModel", s)
            @test occursin("RMSE", s)
            @test occursin("Validation", s)
            @test occursin("Test", s)
        end

        @testset "save_model_artifact from TrainedModel" begin
            tm = train_model(path, cf; val_ratio=0.2, test_ratio=0.2, nnodes=[5], max_epochs=50,
                             earlystop=5, verbose=false)
            mktempdir() do dir
                p = joinpath(dir, "model.json")
                saved = save_model_artifact(p, tm)
                @test saved == p
                @test isfile(p)
                loaded = load_model(p)
                @test loaded isa AbstractNNInterface
            end
        end

        @testset "auto-infer feature spec from paths" begin
            tm = train_model(path; val_ratio=0.2, test_ratio=0.2, nnodes=[5], max_epochs=20,
                             earlystop=5, verbose=false)
            @test tm isa TrainedModel
            @test :B in tm.feature_spec.elements
            @test tm.val_results isa TrainingResults
            @test tm.test_results isa TrainingResults
        end

        @testset "auto-infer feature spec from StructureContainer" begin
            sc = StructureContainer([path])
            tm = train_model(sc; val_ratio=0.2, test_ratio=0.2, nnodes=[5], max_epochs=20,
                             earlystop=5, verbose=false)
            @test tm isa TrainedModel
            @test :B in tm.feature_spec.elements
        end

        @testset "auto-infer with custom feature params" begin
            tm = train_model(path; val_ratio=0.2, test_ratio=0.2, nnodes=[5], max_epochs=20,
                             earlystop=5, p2=2:6, rcut2=5.0, verbose=false)
            @test tm isa TrainedModel
            @test tm.feature_spec.elements == [:B]
            @test !isempty(tm.feature_spec.two_body)
            @test tm.feature_spec.two_body[1].rcut ≈ 5.0
        end

        @testset "verbose=false suppresses output" begin
            tm = train_model(path, cf; val_ratio=0.2, test_ratio=0.2, nnodes=[3], max_epochs=10,
                             earlystop=5, verbose=false)
            @test tm isa TrainedModel
        end

        @testset "seed reproducibility" begin
            tm1 = train_model(path, cf; val_ratio=0.2, test_ratio=0.2, nnodes=[3],
                              max_epochs=10, earlystop=5, seed=123, verbose=false)
            tm2 = train_model(path, cf; val_ratio=0.2, test_ratio=0.2, nnodes=[3],
                              max_epochs=10, earlystop=5, seed=123, verbose=false)
            @test tm1.train_results.fc.labels == tm2.train_results.fc.labels
            @test tm1.val_results.fc.labels == tm2.val_results.fc.labels
            @test tm1.test_results.fc.labels == tm2.test_results.fc.labels
        end

        @testset "pre-split StructureContainers with feature_spec" begin
            sc = StructureContainer([path])
            sc_train, sc_val, sc_test = split(sc, 6, 3, 2; standardize=false)
            tm = train_model(sc_train, sc_val, sc_test, cf;
                nnodes=[5], max_epochs=30, earlystop=5, verbose=false)
            @test tm isa TrainedModel
            @test length(tm.train_results.fc) == 6
            @test length(tm.val_results.fc) == 3
            @test length(tm.test_results.fc) == 2
        end

        @testset "pre-split StructureContainers auto-infer" begin
            sc = StructureContainer([path])
            sc_train, sc_val, sc_test = split(sc, 6, 3, 2; standardize=false)
            tm = train_model(sc_train, sc_val, sc_test;
                nnodes=[5], max_epochs=30, earlystop=5, verbose=false)
            @test tm isa TrainedModel
            @test :B in tm.feature_spec.elements
        end

        @testset "pre-split paths with feature_spec" begin
            sc = StructureContainer([path])
            sc_train, sc_val, sc_test = split(sc, 6, 3, 2; standardize=false)
            mktempdir() do dir
                for (i, s) in enumerate(sc_train.structures)
                    CellBase.write_res(joinpath(dir, "train_$i.res"), s)
                end
                for (i, s) in enumerate(sc_val.structures)
                    CellBase.write_res(joinpath(dir, "val_$i.res"), s)
                end
                for (i, s) in enumerate(sc_test.structures)
                    CellBase.write_res(joinpath(dir, "test_$i.res"), s)
                end
                tm = train_model(
                    joinpath(dir, "train_*.res"),
                    joinpath(dir, "val_*.res"),
                    joinpath(dir, "test_*.res"),
                    cf;
                    nnodes=[5], max_epochs=30, earlystop=5, verbose=false)
                @test tm isa TrainedModel
            end
        end

        @testset "pre-split paths auto-infer" begin
            sc = StructureContainer([path])
            sc_train, sc_val, sc_test = split(sc, 6, 3, 2; standardize=false)
            mktempdir() do dir
                for (i, s) in enumerate(sc_train.structures)
                    CellBase.write_res(joinpath(dir, "train_$i.res"), s)
                end
                for (i, s) in enumerate(sc_val.structures)
                    CellBase.write_res(joinpath(dir, "val_$i.res"), s)
                end
                for (i, s) in enumerate(sc_test.structures)
                    CellBase.write_res(joinpath(dir, "test_$i.res"), s)
                end
                tm = train_model(
                    joinpath(dir, "train_*.res"),
                    joinpath(dir, "val_*.res"),
                    joinpath(dir, "test_*.res");
                    nnodes=[5], max_epochs=30, earlystop=5, verbose=false)
                @test tm isa TrainedModel
                @test :B in tm.feature_spec.elements
            end
        end

        @testset "pre-split FeatureStores" begin
            sc = StructureContainer([path])
            fc = FeatureStore(sc, cf)
            fc_train, fc_val, fc_test = split(fc, 6, 3, 2; standardize=true)
            tm = train_model(fc_train, fc_val, fc_test;
                nnodes=[5], max_epochs=30, earlystop=5, verbose=false)
            @test tm isa TrainedModel
            @test length(tm.train_results.fc) == 6
            @test length(tm.val_results.fc) == 3
            @test length(tm.test_results.fc) == 2
        end

        @testset "elemental_energies passthrough" begin
            e0s = Dict(:B => -5.0)
            tm = train_model(path, cf;
                val_ratio=0.2, test_ratio=0.2, nnodes=[5], max_epochs=30,
                earlystop=5, elemental_energies=e0s, verbose=false)
            @test tm.elemental_energies[:B] ≈ -5.0
        end
    end
end
