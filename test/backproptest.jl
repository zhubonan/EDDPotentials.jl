using Flux

using CellTools: backward!, forward!, ChainGradients
using Test



@testset "Backprop" begin
    allclose(x, y, tol=1e-5) = all(abs.(x .- y) .< tol)

    @testset "single layer" begin
        d1 = Dense(5=>10, tanh)
        d1.bias .= rand(10)
        chain = Chain(d1)
        chaing = ChainGradients(chain, 10)
        x = rand(Float32, 5, 10)
        forward!(chaing, chain, x)
        backward!(chaing, chain)

        # Now use backprop via Zygote

        f(z) = sum(d1.σ.(z * x .+ d1.bias))
        @test f(d1.weight) ≈ sum(chain(x))
        d1gw = Flux.gradient(f, d1.weight)[1]
        @test allclose(d1gw, chaing.layers[1].gw, 1e-5)
    end

    @testset "dual layer" begin

        # Two layer propagation
        d1 = Dense(5=>10, tanh)
        d2 = Dense(10=>8)
        chain = Chain(d1, d2)
        chaing = ChainGradients(chain, 10)
        x = rand(Float32, 5, 10)
        forward!(chaing, chain, x)
        backward!(chaing, chain)

        # Now use backprop via Zygote

        f(z) = sum(d2(d1.σ.(z * x .+ d1.bias)))
        fb(z) = sum(d2(d1.σ.(d1.weight * x .+ z)))
        f2(z) = sum( d2.σ.(z * d1(x) .+ d2.bias))
        fb2(z) = sum( d2.σ.(d2.weight * d1(x) .+ z))
        @test all(f(d1.weight) .≈ sum(chain(x)))
        @test all(f2(d2.weight) .≈ sum(chain(x)))

        #  Gradient of the weight
        d1gw = Flux.gradient(f, d1.weight)[1]
        @test allclose(d1gw, chaing.layers[1].gw)

        #  Gradient of the bias
        d1gb = Flux.gradient(fb, d1.bias)[1]
        @test allclose(d1gb, chaing.layers[1].gb)

        #  Gradient of the weight, second layer
        d2gw = Flux.gradient(f2, d2.weight)[1]
        @test allclose(d2gw, chaing.layers[2].gw)

        d2gb = Flux.gradient(fb2, d2.bias)[1]
        @test allclose(d2gb, chaing.layers[2].gb)

        #  Upstream gradient of the first layer
        d2gx = Flux.gradient(x -> sum(d2(x)), d1(x))[1]
        @test allclose(d2gx, chaing.layers[2].gx)
    end
    
    @testset "jacobian" begin
        ## Compare hand written backprop vs Zytoge
        chain = CellTools.generate_chain(21, [5])
        inp = [rand(Float32, 21)]
        gd = Flux.gradient(() -> mean(chain(inp[1])), Flux.params(chain))
        gp! = CellTools.setup_fg_backprop(chain, inp, [1.0])
        jmat = zeros(eltype(inp[1]), 1, 116)
        gp!([0.], jmat, CellTools.paramvector(chain))

        @test allclose(jmat[1, 1:105], vec(gd.grads[gd.params[1]]))
        @test allclose(jmat[1, 106:110], vec(gd.grads[gd.params[2]]))
        @test allclose(jmat[1, 111:115], vec(gd.grads[gd.params[3]]))
        @test allclose(jmat[1, 116:116], vec(gd.grads[gd.params[4]]))
    end

end