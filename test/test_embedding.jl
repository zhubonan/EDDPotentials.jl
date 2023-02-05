using EDDP:
    TwoBodyFeature, ThreeBodyFeature, CellFeature, withgradient, nfeatures
using Test
using Flux

using ChainRulesCore
using ChainRulesTestUtils

include("utils.jl")

@testset "Embedding" begin
    cell = _h2o_cell()
    nl = NeighbourList(cell, 4.0)
    cf = CellFeature([:H, :O], p2=2:4, p3=2:4)

    fvec1 = zeros(EDDP.nfeatures(cf), length(cell))
    EDDP.feature_vector!(fvec1, cf.two_body, cf.three_body, cell; nl, offset=2)

    # Test apply two-body embedding
    b2 = EDDP.BodyEmbedding(cf.two_body, 2)
    v2 = EDDP.two_body_view(cf, fvec1)
    @test size(b2(v2)) == (3, 2)

    # Test apply three-body embedding
    b3 = EDDP.BodyEmbedding(cf.three_body, 2)
    v3 = EDDP.three_body_view(cf, fvec1)
    @test size(b3(v3)) == (9, 2)

    # All together - converting an full feature vector
    ce = EDDP.CellEmbedding(cf, 2)
    out = ce(fvec1)
    global ce, out, fvec1
    @test size(out, 1) == 2 + 6 + 18
end

@testset "Embedding Backprop" begin
    nat = 10
    cf = EDDP.CellFeature([:O, :H])
    ce = EDDP.CellEmbedding(cf, 5)
    bg = EDDP.BodyEmbeddingGradient(ce.two_body, nat)

    inp = rand(nfeatures(cf), nat)
    out = ce(inp)
    inp_2bd = rand(EDDP.feature_size(cf)[2], nat)
    out_2bd = ce.two_body(inp_2bd)

    # Backprop
    EDDP.forward!(bg, ce.two_body, out_2bd, inp_2bd, 1, 1)

    layers = [ce, Dense(rand(5, size(out, 1)), rand(5)), Dense(rand(1, 5))]
    chain = Chain(layers)
    chaing = EDDP.ChainGradients(chain, nat)

    # CellEmbeddingGradient
    cg = EDDP.CellEmbeddingGradient(ce, nat)
    EDDP.forward!(cg, ce, chaing.layers[2], inp, 1, 1)
    fill!(cg.gu, 1)
    EDDP.backprop!(cg, ce)


    # Check chain
    EDDP.forward!(chaing, chain, inp)
    EDDP.backward!(chaing, chain)
    # Check results
    @test all(chain(inp) .≈ chaing.layers[end].out)

    # Check against Flux
    param = Flux.params(chain)
    grad = Flux.gradient(() -> sum(chain(inp)), param)

    gflux = grad[chain[1].two_body.weight]
    gmbp = chaing.layers[1].two_body.gw
    @test all(gmbp .≈ gmbp)

    gflux = grad[chain[1].three_body.weight]
    gmbp = chaing.layers[1].three_body.gw
    @test all(gmbp .≈ gmbp)

    # Evaluation mode - test the gradients of the input matrix
    EDDP.forward!(chaing, chain, inp)
    EDDP.backward!(chaing, chain; weight_and_bias=false)
    grad, = Flux.gradient(inp -> sum(chain(inp)), inp)
    @test all(grad .≈ chaing.layers[1].gx)
end


@testset "Embedding rrules" begin

    # Test differentiating through body embedding with matrix input (batch input)
    be = EDDP.BodyEmbedding(rand(2, 1), 2)
    w = be.weight
    features = rand(2, 4)
    test_rrule(EDDP._apply_embedding_batch, w, features, check_thunked_output_tangent=true)

    # Test differentiating  through cell embedding with matrix input (batch input) 

    cf = EDDP.CellFeature([:H, :O])
    ce = EDDP.CellEmbedding(cf, 2, 2)
    features = rand(EDDP.nfeatures(cf), 2)
    test_rrule(
        EDDP._apply_embedding_cell,
        EDDP.feature_size(ce.cf)[1] ⊢ NoTangent(),
        EDDP.feature_size(ce.cf)[2] ⊢ NoTangent(),
        EDDP.feature_size(ce.cf)[3] ⊢ NoTangent(),
        ce.two_body.weight,
        ce.three_body.weight,
        features,
        check_thunked_output_tangent=true,
    )

end
