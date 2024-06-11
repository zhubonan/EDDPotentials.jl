using EDDPotentials
using CellBase
using Test
using StatsBase

@testset "Preprocess" begin
    fpath = joinpath(datadir, "training/*.res")
    fpath = relpath(fpath, pwd())
    sc = EDDPotentials.StructureContainer([fpath])

    # Test indexing
    @test isa(sc[1], Cell)
    @test isa(sc[1:2], EDDPotentials.StructureContainer)
    @test length(sc) == 11
    labels = [x.metadata[:label] for x in sc.structures]
    @test isa(sc[[labels[1], labels[2]]], EDDPotentials.StructureContainer)
    @test isa(collect(sc)[1], CellBase.Cell)

    # test splitting
    sc1, sc2 = split(sc, 1, 2)
    @test length(sc1) == 1
    @test length(sc2) == 2

    sc3 = sc1 + sc2
    @test length(sc3) == length(sc1) + length(sc2)
    @test length(sc3) == 3

    sc1, sc2 = split(sc[1:10], 0.1, 0.2)
    @test length(sc1) == 1
    @test length(sc2) == 2

    sc_train, sc_test = EDDPotentials.train_test_split(sc, ratio_test=0.5)
    @test length(sc_train) == 6
    # Default
    fc = EDDPotentials.FeatureContainer(sc)
    @test length(fc) == 11
    fc_train, fc_test = EDDPotentials.train_test_split(fc, ratio_test=0.5)
    @test length(fc_train) == 6

    @test isa(fc[[labels[1], labels[2]]], EDDPotentials.FeatureContainer)
    @test isa(collect(fc)[1], Tuple)

    # Test standardisation
    fc1, fc2 = split(fc[1:10], 5, 3; shuffle=false, standardize=false)
    fc11, fc12 = EDDPotentials.standardize(fc1, fc2)
    @test mean(reduce(hcat, fc11.fvecs)[2, :]) ≈ 0.0 atol = 1e-8
    @test fc11.fvecs != fc1.fvecs
    @test !isapprox(mean(reduce(hcat, fc1.fvecs)[2, :]), 0, atol=1e-8)

    # Combine
    fc3 = fc1 + fc2
    @test length(fc3) == length(fc1) + length(fc2)

    @test fc11.fvecs != fc1.fvecs
    EDDPotentials.standardize!(fc1, fc2)
    @test fc.xt === nothing
    @test fc.yt === nothing
    @test fc1.fvecs[1] != fc.fvecs[1]
    @test !isnothing(fc1.xt)
    @test !isnothing(fc2.xt)
    @test fc1.xt === fc2.xt
    @test fc1.yt === fc2.yt
    @test mean(reduce(hcat, fc1.fvecs)[2, :]) ≈ 0.0 atol = 1e-8
    @test !isapprox(mean(reduce(hcat, fc2.fvecs)[2, :]), 0.0; atol=1e-8)

end
