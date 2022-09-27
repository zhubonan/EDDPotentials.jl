using EDDP
using CellBase
using Test

@testset "Tools" begin
    cell = Cell(Lattice(10, 10, 10), [:H], [[0., 0., 4]])
    cpar = cellpar(cell)
    spos = CellBase.scaled_positions(cell)
    # Test for rattling the cell
    EDDP.rattle_cell!(cell, 0.01)
    @test all(cpar .!= cellpar(cell))
    @test all(isapprox.(spos, CellBase.scaled_positions(cell), atol=1e-10))
end

include("test_preprocess.jl")
include("gptest.jl")
include("gtest.jl")
include("backproptest.jl")
include("dotcasteptest.jl")
include("gtest2.jl")