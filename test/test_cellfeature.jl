using EDDPotentials
using Test


@testset "CellFeature" begin
    # Constructor with defaults
    cf = CellFeature([:O, :B])
    @test collect(cf.elements) == [:B, :O]
    @test nfeatures(cf) == length(cf.elements) + nbodyfeatures(cf, 2) + nbodyfeatures(cf, 3)
    @test nbodyfeatures(cf, 1) == 2
    @test nbodyfeatures(cf, 2) == 4 * 7
    @test nbodyfeatures(cf, 3) == 6 * 7 * 7

    fs = feature_size(cf)
    @test fs[1] == 2
    @test fs[2] == nbodyfeatures(cf, 2)
    @test fs[3] == nbodyfeatures(cf, 3)

    @test spec_index(cf, :B) == 1
    @test spec_index(cf, :O) == 2
    @test spec_symbol(cf, 1) == :B
    @test spec_symbol(cf, 2) == :O

    @test lookup(cf, :B) == 1
    @test lookup(cf, :O) == 2

    @test two_body_channel(cf, 1, 1) == 1
    @test two_body_channel(cf, 1, 2) == 2
    @test two_body_channel(cf, 2, 1) == 3
    @test two_body_channel(cf, 2, 2) == 4

    rng = two_body_range(cf, :B, :B)
    @test rng == two_body_range(cf, 1, 1)
    np2 = nfeatures_per_channel(cf, 2)
    @test length(rng) == np2
    @test first(rng) == 3

    ch = three_body_channel(cf, 1, 1, 1)
    @test ch >= 1
    @test three_body_channel(cf, 1, 1, 2) == three_body_channel(cf, 1, 2, 1)

    rng3 = three_body_range(cf, :B, :B, :B)
    tbfe = nfeatures_per_channel(cf, 3)
    @test length(rng3) == tbfe

    @test suggest_rcut(cf; shell=0.0) == max(cf.rcut2, cf.rcut3)

    names = feature_names(cf; show_value=false)
    @test length(names) == nfeatures(cf) - nbodyfeatures(cf, 1)  # no one-body names

    @test nfeatures_per_channel(cf, 2) == 7
    @test nfeatures_per_channel(cf, 3) == 49
    @test nchannels(cf, 2) == 4
    @test nchannels(cf, 3) == 6

    # Single element
    cf1 = CellFeature([:H])
    @test length(cf1.elements) == 1
    @test nchannels(cf1, 2) == 1
    @test nchannels(cf1, 3) == 1

    # Custom parameters
    cf2 = CellFeature([:C]; p2=[2, 3], rcut2=5.0, rcut3=4.0)
    @test nfeatures_per_channel(cf2, 2) == 2
    @test cf2.rcut2 == 5.0
    @test cf2.rcut3 == 4.0

    # Geometry sequence
    cf3 = CellFeature([:C]; p2=2:8, geometry_sequence=true)
    @test length(cf3.p2) == 7
    @test cf3.p2[1] ≈ 2.0 atol=0.1
    @test cf3.p2[end] ≈ 8.0 atol=0.1

    # Empty p3/q3
    cf4 = CellFeature([:C]; p3=[], q3=[])
    @test nbodyfeatures(cf4, 3) == 0
    @test nfeatures_per_channel(cf4, 3) == 0
    
    # Mixed numeric exponent types should promote safely
    cf_mixed = CellFeature([:C]; p2=[2, 3], p3=[2.5], q3=[1.5])
    @test eltype(cf_mixed.p2) <: AbstractFloat
    @test eltype(cf_mixed.p3) <: AbstractFloat
    @test eltype(cf_mixed.q3) <: AbstractFloat

    # + operator (matching params)
    a = CellFeature([:H]; p2=[2, 3], rcut2=3.0)
    b = CellFeature([:O]; p2=[2, 3], rcut2=3.0)
    c = a + b
    @test sort([c.elements...]) == [:H, :O]
    @test c.p2 == a.p2

    # == operator
    d = CellFeature([:C]; p2=2:8)
    e = CellFeature([:C]; p2=2:8)
    @test d == e
    f = CellFeature([:C]; p2=2:4)
    @test d != f

    # + operator error on mismatch
    g = CellFeature([:H]; p2=[2, 3], rcut2=3.0)
    h = CellFeature([:O]; p2=[2, 4], rcut2=3.0)
    @test_throws ArgumentError g + h
end
