using EDDP: TwoBodyFeature, ThreeBodyFeature, CellFeature, FeatureOptions, withgradient, nfeatures
using Test

@testset "Features" begin
    tb = TwoBodyFeature(2:8, [:H, :H], 4.)
    ttb = ThreeBodyFeature(2:8, 2:8, [:H, :H, :O], 4.)
    cf = CellFeature([:H, :O], [tb], [ttb])
    @test cf.two_body[1] == tb
    @test cf.three_body[1] == ttb

    # Constructor
    opts = FeatureOptions(elements=[:O, :B])
    cf2 = CellFeature(opts)
    @test cf2.elements == [:B, :O]
    @test cf2.two_body[1].p == opts.p2
    cftot = cf2 + cf
    @test cftot.elements == [:B, :H, :O]
    @test length(cftot.two_body) == 4
    @test length(cftot.three_body) == 5

    # Test feature functions
    @test all(cf2.two_body[1](4.0) .== 0)
    @test all(cf2.three_body[1](4.0, 4.0, 4.0) .== 0)
    @test all(cf2.three_body[1](3.0, 4.0, 4.0) .== 0)
    @test any(cf2.three_body[1](3.0, 3.0, 3.0) .!= 0)

    e, g = withgradient(cf2.two_body[1], 3.0)
    @test size(g, 1) == nfeatures(cf2.two_body[1])
    e, g = withgradient(cf2.three_body[1], 3.0, 3.0, 3.0)
    @test size(g, 2) == nfeatures(cf2.three_body[1])
    @test size(g, 1) == 3
end