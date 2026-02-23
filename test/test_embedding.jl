using EDDPotentials: TwoBodyFeature, ThreeBodyFeature, CellFeature, nfeatures
using Test
using Flux

using ChainRulesCore
using ChainRulesTestUtils

@testset "Embedding" begin
    cell = _h2o_cell()
    nl = NeighbourList(cell, 4.0; savevec=true)
    cf = CellFeature([:H, :O], p2=2:4, p3=2:4)

    fvec1 = zeros(EDDPotentials.nfeatures(cf), length(cell))
    workspace = EDDPotentials.EDDPotentialsCore.GradientWorkspace(fvec1)
    EDDPotentials.EDDPotentialsCore.compute_fv!(workspace, cf.two_body, cf.three_body, cell; nl)

    # Test apply two-body embedding
    b2 = EDDPotentials.EDDPotentialsCore.BodyEmbedding(cf.two_body, 2)
    v2 = EDDPotentials.EDDPotentialsCore.two_body_view(cf, fvec1)
    @test size(b2(v2)) == (3, 2)

    # Test apply three-body embedding
    b3 = EDDPotentials.EDDPotentialsCore.BodyEmbedding(cf.three_body, 2)
    v3 = EDDPotentials.EDDPotentialsCore.three_body_view(cf, fvec1)
    @test size(b3(v3)) == (9, 2)

    # All together - converting an full feature vector
    ce = EDDPotentials.EDDPotentialsCore.CellEmbedding(cf, 2)
    out = ce(fvec1)
    global ce, out, fvec1
    @test size(out, 1) == 2 + 6 + 18
end

@testset "Embedding Backprop" begin
    nat = 10
    cf = EDDPotentials.CellFeature([:O, :H])
    ce = EDDPotentials.EDDPotentialsCore.CellEmbedding(cf, 5)
    bg = EDDPotentials.EDDPotentialsCore.BodyEmbeddingGradient(ce.two_body, nat)

    inp = rand(nfeatures(cf), nat)
    out = ce(inp)
    inp_2bd = rand(EDDPotentials.feature_size(cf)[2], nat)
    out_2bd = ce.two_body(inp_2bd)

    # Backprop
    EDDPotentials.EDDPotentialsCore.forward!(bg, ce.two_body, out_2bd, inp_2bd, 1, 1)

    layers = [ce, Dense(rand(5, size(out, 1)), rand(5)), Dense(rand(1, 5))]
    chain = Chain(layers)
    chaing = EDDPotentials.EDDPotentialsCore.ChainGradients(chain, nat)

    # CellEmbeddingGradient
    cg = EDDPotentials.EDDPotentialsCore.CellEmbeddingGradient(ce, nat)
    EDDPotentials.EDDPotentialsCore.forward!(cg, ce, chaing.layers[2], inp, 1, 1)
    fill!(cg.gu, 1)
    EDDPotentials.EDDPotentialsCore.backprop!(cg, ce)


    # Check chain
    EDDPotentials.EDDPotentialsCore.forward!(chaing, chain, inp)
    EDDPotentials.EDDPotentialsCore.backward!(chaing, chain)
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
    EDDPotentials.EDDPotentialsCore.forward!(chaing, chain, inp)
    EDDPotentials.EDDPotentialsCore.backward!(chaing, chain; weight_and_bias=false)
    grad, = Flux.gradient(inp -> sum(chain(inp)), inp)
    @test all(grad .≈ chaing.layers[1].gx)
end


@testset "Embedding rrules" begin

    # Test differentiating through body embedding with matrix input (batch input)
    be = EDDPotentials.EDDPotentialsCore.BodyEmbedding(rand(2, 1), 2)
    w = be.weight
    features = rand(2, 4)
    test_rrule(
        EDDPotentials.EDDPotentialsCore._apply_embedding_batch,
        w,
        features,
        check_thunked_output_tangent=true,
    )

    # Test differentiating  through cell embedding with matrix input (batch input) 

    cf = EDDPotentials.CellFeature([:H, :O])
    ce = EDDPotentials.EDDPotentialsCore.CellEmbedding(cf, 2, 2)
    features = rand(EDDPotentials.nfeatures(cf), 2)
    test_rrule(
        EDDPotentials.EDDPotentialsCore._apply_embedding_cell,
        EDDPotentials.feature_size(ce.cf)[1] ⊢ NoTangent(),
        EDDPotentials.feature_size(ce.cf)[2] ⊢ NoTangent(),
        EDDPotentials.feature_size(ce.cf)[3] ⊢ NoTangent(),
        ce.two_body.weight,
        ce.three_body.weight,
        features,
        check_thunked_output_tangent=true,
    )
end
