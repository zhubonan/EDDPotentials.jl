using EDDP
using Test
using CellBase
using Flux
include("utils.jl")

@testset "Calc" begin
    cell = _h2_cell()
    cf = _generate_cf(cell)
    nnitf = EDDP.ManualFluxBackPropInterface(Chain(
        Dense(EDDP.nfeatures(cf)=>5), Dense(5=>1)
        ),
        length(cell)
        )
    calc = EDDP.NNCalc(cell, cf, nnitf)
end