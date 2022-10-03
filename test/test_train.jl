using EDDP

using Test
using Flux
using NLSolversBase


@testset "Training Tools" begin
    data = vcat([rand(100, 3) for _ in 1:10], [rand(100, 2) for _ in 1:10])
    models = [EDDP.ManualFluxBackPropInterface(
        Chain(Dense(rand(1, 100), rand(1), tanh)), 
        x
        ) for x in (2, 3)]

    f = zeros(length(data))
    y = size.(data, 2) ./ 2.
    EDDP.compute_objectives_bp(f, models, data, f)
    @test any(f != 0.)

    jmat = rand(length(data), EDDP.nparams(models[1]))
    EDDP.compute_objectives_diff_bp(f, jmat, models, data, f)

    @test any(jmat != 0.)

    # Setting up and minimising
    f!, j!, fj! = EDDP.setup_fj(models[1], data, y)
    p0 = EDDP.paramvector(models[1])
    od2 = OnceDifferentiable(f!, j!, fj!, p0, zeros(eltype(data[1]), length(data)), inplace=true);
    opt_res = EDDP.levenberg_marquardt(od2, p0;show_trace=false)

    j!(jmat, opt_res.minimizer)
    @test all(isapprox.(jmat, 0., atol=1e-6))


    # Test with training config
    model = EDDP.ManualFluxBackPropInterface(
        Chain(Dense(rand(1, 100), rand(1))), 
        2
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