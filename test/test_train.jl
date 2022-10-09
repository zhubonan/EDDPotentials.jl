using EDDP
using EDDP: ginit

using Test
using Flux
using StatsBase
using NLSolversBase


@testset "Training Tools" begin
    data = vcat([rand(100, 3) for _ in 1:10], [rand(100, 2) for _ in 1:10])

    f = zeros(length(data))
    y = rand(length(data))
    yt = StatsBase.fit(StatsBase.ZScoreTransform, reshape(y, 1, length(y)), dims=2)

    model = EDDP.ManualFluxBackPropInterface(
        Chain(Dense(rand(1, 100), rand(1), tanh)), yt=yt)
    EDDP.compute_objectives(f, model, data, f)
    @test any(f != 0.)

    jmat = rand(length(data), EDDP.nparams(model))
    EDDP.compute_objectives_diff(f, jmat, model, data, y)

    @test any(jmat != 0.)

    # Setting up and minimising
    f!, j!, fj! = EDDP.setup_fj(model, data, y)
    p0 = EDDP.paramvector(model)

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
    od2 = OnceDifferentiable(f!, j!, fj!, p0, zeros(eltype(data[1]), length(data)), inplace=true);
    opt_res = EDDP.levenberg_marquardt(od2, p0;show_trace=false)

    j!(jmat, opt_res.minimizer)
    @test all(isapprox.(jmat, 0., atol=1e-6))


    # Test with training config
    model = EDDP.ManualFluxBackPropInterface(
        Chain(Dense(rand(1, 100), rand(1))), 
        )

    EDDP.train!(model, data, y;)
    out = model(data[2])
    # Check we have successfully performed the fit
    @test sum(out) ≈ y[2] atol=1e-5

    # Highly overfitted linear case....
    nf = 1000
    data = vcat([rand(nf, 3) for _ in 1:10], [rand(nf, 2) for _ in 1:10])
    y = size.(data, 2) ./ 2
    model = EDDP.LinearInterface(rand(1, nf))

    opt_res, _, _ = EDDP.train!(model, data, y;earlystop=0)
    @show sum(model(data[2])) ≈ 1.5 atol=1e-1
end

@testset "Ensemble" begin
    path = "/home/bonan/appdir/jdev/CellTools-project/EDDP.jl/test/data/training/*.res" 
    path = relpath(path, pwd())
    sc = EDDP.StructureContainer([path])
    cf = EDDP.CellFeature(EDDP.FeatureOptions(elements=[:B]))
    fc = EDDP.FeatureContainer(sc, cf)

    # This gives fix examples
    tdata = EDDP.training_data(fc;ratio_test=0.5)
    tdata.x_train

    # Scale X
    EDDP.transform_x!(tdata.xt, tdata.x_train)
    EDDP.transform_x!(tdata.xt, tdata.x_test)
    @test std(reduce(hcat, tdata.x_train)[end, :]) ≈ 1 atol=1e-7
    @test mean(reduce(hcat, tdata.x_train)[end, :]) ≈ 0 atol=1e-7

    models = []

    model = EDDP.ManualFluxBackPropInterface(
        Chain(Dense(ginit(5, EDDP.nfeatures(fc.feature))), Dense(ginit(1, 5))),
        xt=nothing, yt=tdata.yt
    )
    model_ = EDDP.reinit(model)
    @test any(EDDP.paramvector(model_) .!= EDDP.paramvector(model))
    for _ in 1:10
        model_ = EDDP.reinit(model)
        res = EDDP.train!(model_, tdata.x_train, tdata.y_train; y_test=tdata.y_test, x_test=tdata.x_test)
        push!(models, model_)
    end

    emod = EDDP.create_ensemble(models, tdata.x_train, tdata.y_train)
    emod(tdata.x_train[1])
    out = EDDP.predict_energy.(Ref(emod), tdata.x_train)
    @test size(out) == (length(tdata.y_train),)
    

    ## Test distributed training
    res = EDDP.train_multi_distributed(model, tdata.x_train, tdata.y_train; 
            y_test=tdata.y_test, x_test=tdata.x_test, nmodels=3)
    @test isa(res, EDDP.EnsembleNNInterface)
end