using EDDP
using Flux
using Test


@testset "NN Interface" begin
    chain = Chain(Dense(rand(10, 10)), Dense(rand(10, 10)))
    itf = EDDP.ManualFluxBackPropInterface(chain, 10)
    inp = rand(10, 10)
    output = EDDP.forward!(itf, inp)
    EDDP.backward!(itf)
    @test size(output) == (10, 10)
    gvec = EDDP.paramvector(chain)
    g1 = copy(gvec)
    EDDP.gradparam!(gvec, itf)
    @test any(g1 .!= gvec)

    gout = similar(itf.gchain.layers[1].gx)
    EDDP.gradinp!(gout, itf)

    @test EDDP.nparams(itf) == 220
    @test size(EDDP.paramvector(itf)) == (220, )
    pvec = zeros(220) 
    EDDP.paramvector!(pvec, itf)
    @test pvec[1] == chain.layers[1].weight[1]

    # Setting parameter vectors
    pvec[1:10] .= 0.
    EDDP.setparamvector!(itf, pvec)
    @test all(chain.layers[1].weight[1:10] .== 0.)
end