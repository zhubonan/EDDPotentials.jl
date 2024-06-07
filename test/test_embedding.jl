using EDDPotential: TwoBodyFeature, ThreeBodyFeature, CellFeature, nfeatures
using Test
using Flux

using ChainRulesCore
using ChainRulesTestUtils

@testset "Embedding" begin
    cell = _h2o_cell()
    nl = NeighbourList(cell, 4.0; savevec=true)
    cf = CellFeature([:H, :O], p2=2:4, p3=2:4)

    fvec1 = zeros(EDDPotential.nfeatures(cf), length(cell))
    workspace = EDDPotential.GradientWorkspace(fvec1)
    EDDPotential.compute_fv!(workspace, cf.two_body, cf.three_body, cell; nl)

    # Test apply two-body embedding
    b2 = EDDPotential.BodyEmbedding(cf.two_body, 2)
    v2 = EDDPotential.two_body_view(cf, fvec1)
    @test size(b2(v2)) == (3, 2)

    # Test apply three-body embedding
    b3 = EDDPotential.BodyEmbedding(cf.three_body, 2)
    v3 = EDDPotential.three_body_view(cf, fvec1)
    @test size(b3(v3)) == (9, 2)

    # All together - converting an full feature vector
    ce = EDDPotential.CellEmbedding(cf, 2)
    out = ce(fvec1)
    global ce, out, fvec1
    @test size(out, 1) == 2 + 6 + 18
end

@testset "Embedding Backprop" begin
    nat = 10
    cf = EDDPotential.CellFeature([:O, :H])
    ce = EDDPotential.CellEmbedding(cf, 5)
    bg = EDDPotential.BodyEmbeddingGradient(ce.two_body, nat)

    inp = rand(nfeatures(cf), nat)
    out = ce(inp)
    inp_2bd = rand(EDDPotential.feature_size(cf)[2], nat)
    out_2bd = ce.two_body(inp_2bd)

    # Backprop
    EDDPotential.forward!(bg, ce.two_body, out_2bd, inp_2bd, 1, 1)

    layers = [ce, Dense(rand(5, size(out, 1)), rand(5)), Dense(rand(1, 5))]
    chain = Chain(layers)
    chaing = EDDPotential.ChainGradients(chain, nat)

    # CellEmbeddingGradient
    cg = EDDPotential.CellEmbeddingGradient(ce, nat)
    EDDPotential.forward!(cg, ce, chaing.layers[2], inp, 1, 1)
    fill!(cg.gu, 1)
    EDDPotential.backprop!(cg, ce)


    # Check chain
    EDDPotential.forward!(chaing, chain, inp)
    EDDPotential.backward!(chaing, chain)
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
    EDDPotential.forward!(chaing, chain, inp)
    EDDPotential.backward!(chaing, chain; weight_and_bias=false)
    grad, = Flux.gradient(inp -> sum(chain(inp)), inp)
    @test all(grad .≈ chaing.layers[1].gx)
end


@testset "Embedding rrules" begin

    # Test differentiating through body embedding with matrix input (batch input)
    be = EDDPotential.BodyEmbedding(rand(2, 1), 2)
    w = be.weight
    features = rand(2, 4)
    test_rrule(EDDPotential._apply_embedding_batch, w, features, check_thunked_output_tangent=true)

    # Test differentiating  through cell embedding with matrix input (batch input) 

    cf = EDDPotential.CellFeature([:H, :O])
    ce = EDDPotential.CellEmbedding(cf, 2, 2)
    features = rand(EDDPotential.nfeatures(cf), 2)
    test_rrule(
        EDDPotential._apply_embedding_cell,
        EDDPotential.feature_size(ce.cf)[1] ⊢ NoTangent(),
        EDDPotential.feature_size(ce.cf)[2] ⊢ NoTangent(),
        EDDPotential.feature_size(ce.cf)[3] ⊢ NoTangent(),
        ce.two_body.weight,
        ce.three_body.weight,
        features,
        check_thunked_output_tangent=true,
    )
end
