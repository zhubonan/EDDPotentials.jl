using EDDPotential
using EDDPotential: ginit

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

        model = EDDPotential.ManualFluxBackPropInterface(
            Chain(Dense(rand(1, nf), rand(1), tanh)),
            yt=yt,
        )
        EDDPotential.compute_objectives(f, model, data, f)
        @test any(f != 0.0)

        jmat = rand(length(data), EDDPotential.nparams(model))
        EDDPotential.compute_objectives_diff(f, jmat, model, data, y)

        @test any(jmat != 0.0)

        # Setting up and minimising
        f!, j!, fj! = EDDPotential.setup_fj(model, data, y)
        p0 = EDDPotential.paramvector(model)

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
        opt_res = EDDPotential.levenberg_marquardt(od2, p0; show_trace=false)

        j!(jmat, opt_res.minimizer)
        @test all(isapprox.(jmat, 0.0, atol=1e-6))


        # Test with training config
        model = EDDPotential.ManualFluxBackPropInterface(Chain(Dense(rand(1, nf), rand(1))))

        EDDPotential.train_lm!(model, data, y;)
        out = model(data[2])
        # Check we have successfully performed the fit
        @test sum(out) ≈ y[2] atol = 1e-5

        # Highly overfitted linear case....
        nf = 1000
        data = vcat([rand(nf, 3) for _ = 1:10], [rand(nf, 2) for _ = 1:10])
        y = size.(data, 2) ./ 2
        model = EDDPotential.LinearInterface(rand(1, nf))

        opt_res, _, _ = EDDPotential.train_lm!(model, data, y; earlystop=0)
    end

    @testset "Ensemble" begin
        path = joinpath(datadir, "training/*.res")
        path = relpath(path, pwd())
        sc = EDDPotential.StructureContainer([path])
        cf = EDDPotential.CellFeature([:B])
        fc = EDDPotential.FeatureContainer(sc, cf)

        # This gives fix examples
        fc_train, fc_test = split(fc, 5, 5; standardize=true, apply_transform=true)
        models = []
        model = EDDPotential.ManualFluxBackPropInterface(
            Chain(Dense(ginit(5, EDDPotential.nfeatures(fc.feature))), Dense(ginit(1, 5))),
            xt=fc_train.xt,
            yt=fc_train.yt,
            apply_xt=false,
        )
        model_ = EDDPotential.reinit(model)
        @test any(EDDPotential.paramvector(model_) .!= EDDPotential.paramvector(model))
        for _ = 1:10
            model_ = EDDPotential.reinit(model)
            res = EDDPotential.train!(model_, fc_train, fc_test)
            push!(models, model_)
        end

        emod = EDDPotential.create_ensemble(models, fc_train)
        out = EDDPotential.predict_energy(emod, fc_train[1][1])
        @test isa(out, Real)

        ## Training with multithreading - this is not optimum...
        res = EDDPotential.train_multi_threaded(
            model,
            fc_train,
            fc_test;
            nmodels=3,
            save_each_model=false,
        )
        @test isa(res, EDDPotential.EnsembleNNInterface)
    end


    @testset "Flux" begin
        nf = 1000
        data = vcat([rand(nf, 3) for _ = 1:10], [rand(nf, 2) for _ = 1:10])

        f = zeros(length(data))
        y = rand(length(data))
        yt = StatsBase.fit(StatsBase.ZScoreTransform, reshape(y, 1, length(y)), dims=2)

        model = EDDPotential.FluxInterface(Chain(Dense(rand(1, nf), rand(1))))
        EDDPotential.compute_objectives(f, model, data, f)
        @test any(f != 0.0)

        jmat = rand(length(data), EDDPotential.nparams(model))
        EDDPotential.compute_objectives_diff(f, jmat, model, data, y)
        @test any(jmat != 0.0)

        # Setting up and minimising
        f!, j!, fj! = EDDPotential.setup_fj(model, data, y)
        p0 = EDDPotential.paramvector(model)

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
        opt_res = EDDPotential.levenberg_marquardt(od2, p0; show_trace=false)
        #@test_broken opt_res.g_converged
        @test opt_res.x_converged
    end
end
