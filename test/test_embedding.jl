using EDDP: TwoBodyFeature, ThreeBodyFeature, CellFeature, FeatureOptions, withgradient, nfeatures
using Test

include("utils.jl")

@testset "Embedding" begin
    cell = _h2o_cell()
    nl = NeighbourList(cell, 4.0)
    cf = CellFeature([:H, :O], p2=2:4, p3=2:4)

    fvec1 = zeros(EDDP.nfeatures(cf), length(cell))
    EDDP.feature_vector!(fvec1, cf.two_body, cf.three_body, cell;nl, offset=2)

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
    @test length(out) == 2 + 6 + 18
end