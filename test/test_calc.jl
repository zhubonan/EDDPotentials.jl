using EDDP
using Test
using CellBase
using Flux
include("utils.jl")

@testset "Calc" begin
    cell = _h2_cell()
    cf = _generate_cf(cell)
    nnitf = EDDP.ManualFluxBackPropInterface(Chain(
        Dense(EDDP.nfeatures(cf;ignore_one_body=false)=>5), Dense(5=>1)
        ),
        length(cell)
        )
    global calc = EDDP.NNCalc(cell, cf, nnitf)
    nnitf.chain(calc.v)
    
    cell2 = deepcopy(cell)
    positions(cell2) .= 0.
    EDDP.copycell!(cell, cell2)
    @test EDDP.is_equal(cell, cell2)

    eng = EDDP.get_energy(calc)
    @test isa(eng, Float64)

    forces = EDDP.get_forces(calc)
    # Newton's second law
    @test all(isapprox.(sum(forces, dims=2), 0, atol=1e-10 )) 
end