using EDDPotentials
using EDDPotentials: TwoBodyFeature, ThreeBodyFeature, CellFeature, nfeatures
using Test


@testset "Features" begin
    # Test permequal
    @test EDDPotentials.permequal((:A, :B), :A, :B)
    @test !EDDPotentials.permequal((:A, :B), :B, :A)
    @test EDDPotentials.permequal((:A, :B, :C), :A, :B, :C)
    @test EDDPotentials.permequal((:A, :B, :C), :A, :C, :B)
    @test !EDDPotentials.permequal((:B, :C, :A), :A, :B, :C)
    @test !EDDPotentials.permequal((:C, :B, :A), :A, :C, :B)

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

    # Test feature cutoff functions
    @test all(cf2.two_body[1](4.0) .== 0)
    # Two-body feature should be non-zero inside cutoff
    fval_inner = cf2.two_body[1](2.0)
    @test any(fval_inner .!= 0)

end

@testset "Analytic Feature Vector" begin
    positions = Float64[
        0.0 1.0 0.0
        0.0 0.0 2.0
        0.0 0.0 0.0
    ]
    cell = Cell(Lattice(10.0, 10.0, 10.0), [:O, :H, :H], positions)
    cf = CellFeature([:H, :O]; p2=[2, 3], p3=[2, 3], q3=[1], rcut2=3.0, rcut3=3.0)
    fvec = EDDPotentials.feature_vector(cf, cell)
    expected = zeros(size(fvec))

    rcut = 3.0
    r_oh1 = 1.0
    r_oh2 = 2.0
    r_hh = sqrt(5.0)
    f_oh1 = 2 * (1 - r_oh1 / rcut)
    f_oh2 = 2 * (1 - r_oh2 / rcut)
    f_hh = 2 * (1 - r_hh / rcut)

    expected[1, :] = [0.0, 1.0, 1.0]
    expected[2, :] = [1.0, 0.0, 0.0]

    i_hh, j_hh = EDDPotentials.EDDPotentialsCore.lookup(cf, :H, :H)
    expected[i_hh:j_hh, :] = [
        0.0 f_hh^2 f_hh^2
        0.0 f_hh^3 f_hh^3
    ]

    i_ho, j_ho = EDDPotentials.EDDPotentialsCore.lookup(cf, :H, :O)
    expected[i_ho:j_ho, :] = [
        0.0 f_oh1^2 f_oh2^2
        0.0 f_oh1^3 f_oh2^3
    ]

    i_oh, j_oh = EDDPotentials.EDDPotentialsCore.lookup(cf, :O, :H)
    expected[i_oh:j_oh, :] = [
        f_oh1^2 + f_oh2^2 0.0 0.0
        f_oh1^3 + f_oh2^3 0.0 0.0
    ]

    i_ohh, j_ohh = EDDPotentials.EDDPotentialsCore.lookup(cf, :O, :H, :H)
    expected[i_ohh:j_ohh, :] = [
        f_oh1^2 * f_oh2^2 * f_hh 0.0 0.0
        f_oh1^3 * f_oh2^3 * f_hh 0.0 0.0
    ]

    i_hho, j_hho = EDDPotentials.EDDPotentialsCore.lookup(cf, :H, :H, :O)
    expected[i_hho:j_hho, :] = [
        0.0 f_hh^2 * f_oh1^2 * f_oh2 f_hh^2 * f_oh2^2 * f_oh1
        0.0 f_hh^3 * f_oh1^3 * f_oh2 f_hh^3 * f_oh2^3 * f_oh1
    ]

    i_oho, j_oho = EDDPotentials.EDDPotentialsCore.lookup(cf, :O, :H, :O)
    i_ooo, j_ooo = EDDPotentials.EDDPotentialsCore.lookup(cf, :O, :O, :O)
    i_hhh, j_hhh = EDDPotentials.EDDPotentialsCore.lookup(cf, :H, :H, :H)
    i_hoo, j_hoo = EDDPotentials.EDDPotentialsCore.lookup(cf, :H, :O, :O)

    @test size(fvec) == (nfeatures(cf), 3)
    @test fvec[i_oho:j_oho, :] == zeros(2, 3)
    @test fvec[i_ooo:j_ooo, :] == zeros(2, 3)
    @test fvec[i_hhh:j_hhh, :] == zeros(2, 3)
    @test fvec[i_hoo:j_hoo, :] == zeros(2, 3)
    @test fvec ≈ expected atol = 1e-12
end


# @testset "Features New" begin
#     cell = _h2_cell()
#     nl = NeighbourList(cell, 4.0)
#     cf = CellFeature([:H], p2=2:4, p3=2:4)

#     # Test new optimised routine for computing the features
#     fvec1 = zeros(EDDPotentials.nfeatures(cf), length(cell))
#     EDDPotentials.feature_vector!(fvec1, cf.two_body, cf.three_body, cell; nl, offset=1)
#     fvec2 = zeros(EDDPotentials.nfeatures(cf), length(cell))
#     EDDPotentials.feature_vector2!(fvec2, cf.two_body, cell; nl, offset=1)
#     EDDPotentials.feature_vector3!(fvec2, cf.three_body, cell; nl, offset=1 + 3)
#     # Check consistency
#     @test all(isapprox.(fvec2[2:4, :], fvec1[2:4, :], atol=1e-7))
#     @test all(isapprox.(fvec2[4:end, :], fvec1[4:end, :], atol=1e-7))


#     cell = _h2o_cell()
#     nl = NeighbourList(cell, 4.0)
#     cf = CellFeature([:H], p2=2:4, p3=2:4)

#     # Test new optimised routine for computing the features
#     fvec1 = zeros(EDDPotentials.nfeatures(cf), length(cell))
#     EDDPotentials.feature_vector!(fvec1, cf.two_body, cf.three_body, cell; nl, offset=1)
#     fvec2 = zeros(EDDPotentials.nfeatures(cf), length(cell))
#     EDDPotentials.feature_vector2!(fvec2, cf.two_body, cell; nl, offset=1)
#     EDDPotentials.feature_vector3!(fvec2, cf.three_body, cell; nl, offset=1 + 3)
#     # Check consistency
#     @test all(isapprox.(fvec2[2:4, :], fvec1[2:4, :], atol=1e-7))
#     @test all(isapprox.(fvec2[4:end, :], fvec1[4:end, :], atol=1e-7))


#     # Test with multi specie atoms
#     cell = _h2o_cell()
#     nl = NeighbourList(cell, 4.0)
#     cf = CellFeature([:H, :O], p2=2:4, p3=2:4)
#     n1, n2, n3 = EDDPotentials.feature_size(cf)

#     @test length(cf.two_body) == 4
#     @test length(cf.three_body) == 6

#     # Test new optimised routine for computing the features
#     fvec1 = zeros(EDDPotentials.nfeatures(cf), length(cell))
#     EDDPotentials.feature_vector!(fvec1, cf.two_body, cf.three_body, cell; nl, offset=n1)

#     # Reference is the previous method 
#     fvec2 = zeros(EDDPotentials.nfeatures(cf), length(cell))
#     EDDPotentials.feature_vector2!(fvec2, cf.two_body, cell; nl, offset=n1)
#     EDDPotentials.feature_vector3!(fvec2, cf.three_body, cell; nl, offset=n1 + n2)
#     # Check consistency
#     @test all(isapprox.(fvec2[3:n1+n2, :], fvec1[3:n1+n2, :], atol=1e-7))
#     @test all(isapprox.(fvec2[n1+n2+1:end, :], fvec1[n1+n2+1:end, :], atol=1e-7))
# end
