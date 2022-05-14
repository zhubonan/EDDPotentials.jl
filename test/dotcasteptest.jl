using Test
using CellTools

@testset ".castep" begin
    datadir = joinpath(splitdir(@__FILE__)[1], "data")
    frames = CellTools.read_castep(joinpath(datadir, "8B-22-05-09-14-55-04-bfb10744-shake-8.castep"))
    @test length(frames) == 1
    @test length(frames[1].species) == 8
    @test length(frames[1].forces) == 24
    @test frames[1].forces[1] ≈ 0.77072
    @test frames[1].extras[:stress][1] ≈ -17.473519
    @test frames[1].energy ≈ -658.0427613302
end