using EDDP
using CellBase
using Test

@testset "Tools" begin
    cell = Cell(Lattice(10, 10, 10), [:H], [[0.0, 0.0, 4]])
    cpar = cellpar(cell)
    spos = CellBase.scaled_positions(cell)
    # Test for rattling the cell
    EDDP.rattle_cell!(cell, 0.01)
    @test all(cpar .!= cellpar(cell))
    @test all(isapprox.(spos, CellBase.scaled_positions(cell), atol=1e-10))
end

include("utils.jl")

include("test_records.jl")
include("test_cellfeature.jl")
include("test_embedding.jl")
include("test_gradients.jl")
include("test_backprop.jl")
include("test_nninterface.jl")
include("test_calc.jl")
include("test_preprocess.jl")
include("test_dotcastep.jl")
include("test_train.jl")
include("test_builder.jl")
include("test_link.jl")
