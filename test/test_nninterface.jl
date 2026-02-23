using EDDPotentials
using EDDPotentials.EDDPotentialsCore:
    LinearInterface,
    paramvector,
    setparamvector!,
    nparams,
    gradparam!,
    gradinp!
using Flux
using Test

@testset "NN Interface" begin

    @testset "Linear" begin
        coeff = [0.1 0.2 0.3]
        x = repeat(transpose(coeff), 1, 2)
        l = LinearInterface(coeff)

        @test size(EDDPotentials.EDDPotentialsCore.forward!(l, x)) == (1, 2)
        @test l(x) == [sum(coeff .* coeff) sum(coeff .* coeff)]

        @test begin
            grad = similar(x)
            gradinp!(grad, l, x)
            grad[:, 1] == l.param[:]
        end
        EDDPotentials.EDDPotentialsCore.backward!(l)
        grad = copy(paramvector(l))
        gradparam!(grad, l, x)
        @test grad == sum(x, dims=2)[:]
    end

    @testset "Flux" begin
        l = EDDPotentials.EDDPotentialsCore.FluxInterface(Dense(10 => 1))
        x = rand(10, 2)

        @test size(EDDPotentials.EDDPotentialsCore.forward!(l, x)) == (1, 2)

        grad = similar(x)
        gradinp!(grad, l, x)
        @test begin
            grad[:, 1] == l.model.weight[:]
        end

        EDDPotentials.EDDPotentialsCore.backward!(l)
        grad = zeros(EDDPotentials.EDDPotentialsCore.nparams(l))
        gradparam!(grad, l, x)
        @test allclose(grad[1:size(x, 1)], sum(x, dims=2)[:], atol=1e-6)
    end


    @testset "MBP" begin
        chain = Chain(Dense(rand(10, 10)), Dense(rand(10, 10)))
        itf = EDDPotentials.EDDPotentialsCore.ManualFluxBackPropInterface(chain)
        inp = rand(10, 10)
        output = EDDPotentials.EDDPotentialsCore.forward!(itf, inp)
        EDDPotentials.EDDPotentialsCore.backward!(itf)
        @test size(output) == (10, 10)
        gvec = EDDPotentials.EDDPotentialsCore.paramvector(chain)
        g1 = copy(gvec)
        EDDPotentials.EDDPotentialsCore.gradparam!(gvec, itf)
        @test any(g1 .!= gvec)

        gout = similar(itf.gchains[1].layers[1].gx)
        EDDPotentials.EDDPotentialsCore.gradinp!(gout, itf)

        @test EDDPotentials.EDDPotentialsCore.nparams(itf) == 220
        @test size(EDDPotentials.EDDPotentialsCore.paramvector(itf)) == (220,)
        pvec = zeros(220)
        EDDPotentials.EDDPotentialsCore.paramvector!(pvec, itf)
        @test pvec[1] == chain.layers[1].weight[1]

        # Setting parameter vectors
        pvec[1:10] .= 0.0
        EDDPotentials.EDDPotentialsCore.setparamvector!(itf, pvec)
        @test all(chain.layers[1].weight[1:10] .== 0.0)
    end

    @testset "Ensemble" begin
        function linearitf()
            coeff = [0.1 0.2 0.3]
            LinearInterface(coeff)
        end

        # Test combining two Linear interfaces
        itf = EDDPotentials.EDDPotentialsCore.EnsembleNNInterface((linearitf(), linearitf()), [0.5, 0.5])
        coeff = [0.1, 0.2, 0.3]
        x = repeat(coeff, 1, 2)
        @test size(EDDPotentials.EDDPotentialsCore.forward!(itf, x)) == (1, 2)
        @test itf(x) == [sum(coeff .* coeff) sum(coeff .* coeff)]
        itf(x)  # Forward step - implied
        @test begin
            grad = similar(x)
            gradinp!(grad, itf)
            grad[:] == [itf.models[1].param[:]..., itf.models[2].param[:]...]
        end
        EDDPotentials.EDDPotentialsCore.backward!(itf)
        grad = copy(paramvector(itf))
        gradparam!(grad, itf)
        @test grad == repeat(sum(x, dims=2)[:], 2)

        # test combining two manual back prop interfaces
        function _get_chainitf()
            chain = Chain(Dense(rand(10, 10)), Dense(rand(10, 10)))
            EDDPotentials.EDDPotentialsCore.ManualFluxBackPropInterface(chain)
        end
        inp = rand(10, 10)
        itf = EDDPotentials.EDDPotentialsCore.EnsembleNNInterface(
            (_get_chainitf(), _get_chainitf()),
            [0.8, 0.2],
        )
        # Forward step
        itf(inp)
        EDDPotentials.EDDPotentialsCore.backward!(itf)
        # Collect gradients
        gv = similar(inp)
        gv .= 0
        EDDPotentials.EDDPotentialsCore.gradinp!(gv, itf)

        # Manually compute the gradients....
        g1 = EDDPotentials.EDDPotentialsCore.gradinp!(zeros(size(inp)...), itf.models[1])
        g2 = EDDPotentials.EDDPotentialsCore.gradinp!(zeros(size(inp)...), itf.models[2])
        g3 = @. g1 * 0.8 + g2 * 0.2
        @test g3 == gv

        gp = zeros(nparams(itf))
        EDDPotentials.EDDPotentialsCore.paramvector!(gp, itf)
        @test gp[1:nparams(itf.models[1])] == EDDPotentials.EDDPotentialsCore.paramvector(itf.models[1])
    end
end
