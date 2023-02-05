using EDDP:
    TwoBodyFeature, ThreeBodyFeature, CellFeature, withgradient, nfeatures
using Test

include("utils.jl")

@testset "Features" begin
    # Test permequal
    @test EDDP.permequal((:A, :B), :A, :B)
    @test !EDDP.permequal((:A, :B), :B, :A)
    @test EDDP.permequal((:A, :B, :C), :A, :B, :C)
    @test EDDP.permequal((:A, :B, :C), :A, :C, :B)
    @test !EDDP.permequal((:B, :C, :A), :A, :B, :C)
    @test !EDDP.permequal((:C, :B, :A), :A, :C, :B)

    # Feature
    tb = TwoBodyFeature(2:8, [:H, :H], 4.0)
    ttb = ThreeBodyFeature(2:8, 2:8, [:H, :H, :O], 4.0)
    cf = CellFeature([:H, :O], [tb], [ttb])
    @test cf.two_body[1] == tb
    @test cf.three_body[1] == ttb

    # Constructor
    cf2 = CellFeature([:O, :B])
    @test cf2.elements == [:B, :O]
    @test length(cf2.two_body) == 4
    @test length(cf2.three_body) == 6

    cftot = cf2 + cf
    @test cftot.elements == [:B, :H, :O]
    @test length(cftot.two_body) == 5
    @test length(cftot.three_body) == 7

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


@testset "Features New" begin
    cell = _h2_cell()
    nl = NeighbourList(cell, 4.0)
    cf = CellFeature([:H], p2=2:4, p3=2:4)

    # Test new optimised routine for computing the features
    fvec1 = zeros(EDDP.nfeatures(cf), length(cell))
    EDDP.feature_vector!(fvec1, cf.two_body, cf.three_body, cell; nl, offset=1)
    fvec2 = zeros(EDDP.nfeatures(cf), length(cell))
    EDDP.feature_vector2!(fvec2, cf.two_body, cell; nl, offset=1)
    EDDP.feature_vector3!(fvec2, cf.three_body, cell; nl, offset=1 + 3)
    # Check consistency
    @test all(isapprox.(fvec2[2:4, :], fvec1[2:4, :], atol=1e-7))
    @test all(isapprox.(fvec2[4:end, :], fvec1[4:end, :], atol=1e-7))


    cell = _h2o_cell()
    nl = NeighbourList(cell, 4.0)
    cf = CellFeature([:H], p2=2:4, p3=2:4)

    # Test new optimised routine for computing the features
    fvec1 = zeros(EDDP.nfeatures(cf), length(cell))
    EDDP.feature_vector!(fvec1, cf.two_body, cf.three_body, cell; nl, offset=1)
    fvec2 = zeros(EDDP.nfeatures(cf), length(cell))
    EDDP.feature_vector2!(fvec2, cf.two_body, cell; nl, offset=1)
    EDDP.feature_vector3!(fvec2, cf.three_body, cell; nl, offset=1 + 3)
    # Check consistency
    @test all(isapprox.(fvec2[2:4, :], fvec1[2:4, :], atol=1e-7))
    @test all(isapprox.(fvec2[4:end, :], fvec1[4:end, :], atol=1e-7))


    # Test with multi specie atoms
    cell = _h2o_cell()
    nl = NeighbourList(cell, 4.0)
    cf = CellFeature([:H, :O], p2=2:4, p3=2:4)
    n1, n2, n3 = EDDP.feature_size(cf)

    @test length(cf.two_body) == 4
    @test length(cf.three_body) == 6

    # Test new optimised routine for computing the features
    fvec1 = zeros(EDDP.nfeatures(cf), length(cell))
    EDDP.feature_vector!(fvec1, cf.two_body, cf.three_body, cell; nl, offset=n1)

    # Reference is the previous method 
    fvec2 = zeros(EDDP.nfeatures(cf), length(cell))
    EDDP.feature_vector2!(fvec2, cf.two_body, cell; nl, offset=n1)
    EDDP.feature_vector3!(fvec2, cf.three_body, cell; nl, offset=n1 + n2)
    # Check consistency
    @test all(isapprox.(fvec2[3:n1+n2, :], fvec1[3:n1+n2, :], atol=1e-7))
    @test all(isapprox.(fvec2[n1+n2+1:end, :], fvec1[n1+n2+1:end, :], atol=1e-7))
end
