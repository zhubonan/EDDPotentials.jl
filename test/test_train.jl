using EDDP
using EDDP: ginit

using Test
using Flux
using StatsBase
using NLSolversBase


@testset "Training Tools" begin
    data = vcat([rand(100, 3) for _ in 1:10], [rand(100, 2) for _ in 1:10])
    model = EDDP.ManualFluxBackPropInterface(
        Chain(Dense(rand(1, 100), rand(1), tanh)))

    f = zeros(length(data))
    y = size.(data, 2) ./ 2.
    EDDP.compute_objectives(f, model, data, f)
    @test any(f != 0.)

    jmat = rand(length(data), EDDP.nparams(model))
    EDDP.compute_objectives_diff(f, jmat, model, data, y)

    @test any(jmat != 0.)

    # Setting up and minimising
    f!, j!, fj! = EDDP.setup_fj(model, data, y)
    p0 = EDDP.paramvector(model)
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
    @test sum(out) ≈ 1.5 atol=1e-5

    data = vcat([rand(100, 3) for _ in 1:10], [rand(100, 2) for _ in 1:10])
    y = size.(data, 2) ./ 2
    model = EDDP.LinearInterface(rand(1, 100))

    opt_res, _, _ = EDDP.train!(model, data, y;)
    @test sum(model(data[2])) ≈ 1.5 atol=0.1
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
    global tdata
    @test std(reduce(hcat, tdata.x_train)[end, :]) ≈ 1 atol=1e-7
    @test mean(reduce(hcat, tdata.x_train)[end, :]) ≈ 0 atol=1e-7

    EDDP.transform_x!(tdata.xt, tdata.x_test)

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