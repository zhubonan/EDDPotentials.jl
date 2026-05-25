using EDDPotentials
using EDDPotentials.Potential: LinearInterface, ManualFluxBackPropInterface, FluxInterface, EnsembleNNInterface, clear_transient_gradients!, forward!, backward!, paramvector, paramvector!, setparamvector!, nparams, gradparam!, gradinp!
using Flux
using Test

isdefined(Main, :allclose) || include("utils.jl")

@testset "NN Interface" begin

    @testset "Linear" begin
        coeff = [0.1 0.2 0.3]
        x = repeat(transpose(coeff), 1, 2)
        l = LinearInterface(coeff)

        @test size(forward!(l, x)) == (1, 2)
        @test l(x) == [sum(coeff .* coeff) sum(coeff .* coeff)]

        @test begin
            grad = similar(x)
            gradinp!(grad, l, x)
            grad[:, 1] == l.param[:]
        end
        backward!(l)
        grad = copy(paramvector(l))
        gradparam!(grad, l, x)
        @test grad == sum(x, dims=2)[:]
    end

    @testset "Flux" begin
        l = FluxInterface(Dense(10 => 1; init=Potential.glorot_uniform_f64))
        x = rand(10, 2)

        @test size(forward!(l, x)) == (1, 2)

        grad = similar(x)
        gradinp!(grad, l, x)
        @test begin
            grad[:, 1] == l.model.weight[:]
        end

        backward!(l)
        grad = zeros(nparams(l))
        gradparam!(grad, l, x)
        @test allclose(grad[1:size(x, 1)], sum(x, dims=2)[:], atol=1e-6)
    end


    @testset "MBP" begin
        chain = Chain(Dense(rand(10, 10)), Dense(rand(10, 10)))
        itf = ManualFluxBackPropInterface(chain)
        @test itf.gradient_cache_size == 8
        inp = rand(10, 10)
        output = forward!(itf, inp)
        backward!(itf)
        @test size(output) == (10, 10)
        gvec = paramvector(chain)
        g1 = copy(gvec)
        gradparam!(gvec, itf)
        @test any(g1 .!= gvec)

        gout = similar(itf.gchains[1].layers[1].gx)
        gradinp!(gout, itf)

        @test nparams(itf) == 220
        @test size(paramvector(itf)) == (220,)
        pvec = zeros(220)
        paramvector!(pvec, itf)
        @test pvec[1] == chain.layers[1].weight[1]

        # Setting parameter vectors
        pvec[1:10] .= 0.0
        setparamvector!(itf, pvec)
        @test all(chain.layers[1].weight[1:10] .== 0.0)
    end

    @testset "MBP gradient cache" begin
        chain = Chain(Dense(rand(5, 10), rand(5)), Dense(rand(1, 5), rand(1)))

        @test_throws ArgumentError ManualFluxBackPropInterface(chain; gradient_cache_size=0)

        itf = ManualFluxBackPropInterface(chain; gradient_cache_size=3)
        @test itf.gradient_cache_size == 3

        for bsize in 2:6
            forward!(itf, rand(10, bsize))
            @test length(itf.gchains) <= 3
        end
        @test [g.n for g in itf.gchains] == [6, 5, 4]

        forward!(itf, rand(10, 4))
        @test [g.n for g in itf.gchains] == [4, 6, 5]

        forward!(itf, rand(10, 7))
        @test [g.n for g in itf.gchains] == [7, 4, 6]
        @test !(5 in [g.n for g in itf.gchains])

        backward!(itf)
        gout = zeros(10, 7)
        gradinp!(gout, itf)
        @test any(x -> !iszero(x), gout)

        clear_transient_gradients!(itf)
        @test isempty(itf.gchains)
    end

    @testset "Ensemble" begin
        function linearitf()
            coeff = [0.1 0.2 0.3]
            LinearInterface(coeff)
        end

        # Test combining two Linear interfaces
        itf = EnsembleNNInterface((linearitf(), linearitf()), [0.5, 0.5])
        coeff = [0.1, 0.2, 0.3]
        x = repeat(coeff, 1, 2)
        @test size(forward!(itf, x)) == (1, 2)
        @test itf(x) == [sum(coeff .* coeff) sum(coeff .* coeff)]
        itf(x)  # Forward step - implied
        @test begin
            grad = similar(x)
            gradinp!(grad, itf)
            grad[:] == [itf.models[1].param[:]..., itf.models[2].param[:]...]
        end
        backward!(itf)
        grad = copy(paramvector(itf))
        gradparam!(grad, itf)
        @test grad == [repeat(sum(x, dims=2)[:], 2); 0.0; 0.0]
        @test length(paramvector(itf)) == nparams(itf)
        pvec = paramvector(itf)
        pvec[end-1:end] .= [0.25, 0.75]
        setparamvector!(itf, pvec)
        @test itf.weights ≈ [0.25, 0.75]

        single = EnsembleNNInterface((linearitf(),), [1.0])
        @test length(paramvector(single)) == nparams(single)

        # test combining two manual back prop interfaces
        function _get_chainitf()
            chain = Chain(Dense(rand(10, 10)), Dense(rand(10, 10)))
            ManualFluxBackPropInterface(chain)
        end
        inp = rand(10, 10)
        itf = EnsembleNNInterface(
            (_get_chainitf(), _get_chainitf()),
            [0.8, 0.2],
        )
        # Forward step
        itf(inp)
        backward!(itf)
        # Collect gradients
        gv = similar(inp)
        gv .= 0
        gradinp!(gv, itf)

        # Manually compute the gradients....
        g1 = gradinp!(zeros(size(inp)...), itf.models[1])
        g2 = gradinp!(zeros(size(inp)...), itf.models[2])
        g3 = @. g1 * 0.8 + g2 * 0.2
        @test g3 == gv

        gp = zeros(nparams(itf))
        paramvector!(gp, itf)
        @test gp[1:nparams(itf.models[1])] == paramvector(itf.models[1])
        @test gp[end-1:end] == itf.weights
    end
end
