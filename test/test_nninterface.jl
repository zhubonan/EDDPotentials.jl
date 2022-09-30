using EDDP
using EDDP: LinearInterface, ManualFluxBackPropInterface, paramvector, setparamvector!, nparams, gradparam!, gradinp!
using Flux
using Test


@testset "NN Interface" begin

    @testset "Linear" begin
        coeff = [0.1 0.2 0.3]
        x = repeat(transpose(coeff), 1, 2)
        l = LinearInterface(coeff)

        @test size(EDDP.forward!(l, x)) == (1, 2)
        @test l(x) == [sum(coeff.*coeff) sum(coeff.*coeff)]
        
        @test begin
            grad = similar(x)
            gradinp!(grad, l, x)
            grad[:, 1] == l.param[:]
        end

        grad = copy(paramvector(l))
        @show grad
        gradparam!(grad, l, x)
        @show grad
        @test grad == sum(x, dims=2)[:]
    end

    @testset "MBP" begin
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
end