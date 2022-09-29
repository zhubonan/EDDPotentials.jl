using EDDP
using EDDP: get_cell, get_forces, get_stress
using Test
using CellBase
using Flux
include("utils.jl")

@testset "Calc" begin
    cell = _h2_cell()

    cf = EDDP.CellFeature(
        EDDP.FeatureOptions(elements=unique(species(cell)), p2=[2], q3=[2, 3], p3=[2, 3])
    )

    nnitf = EDDP.ManualFluxBackPropInterface(Chain(
        Dense(rand(5, EDDP.nfeatures(cf;ignore_one_body=false))), Dense(rand(1, 5))
        ),
        length(cell)
        )
    calc = EDDP.NNCalc(cell, cf, nnitf)
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

    stress = EDDP.get_stress(calc)
    @test size(stress) == (3,3)
    @test any(stress .!== 0.)

    # Test against small displacements finite displacements
    ftmp = copy(EDDP.get_forces(calc))
    etmp = EDDP.get_energy(calc)
    amp = 1e-9
    positions(get_cell(calc))[1] += amp
    @test EDDP._need_calc(calc, true)
    tmp = (get_energy(calc) - etmp) / amp
    ftmp2 = get_forces(calc)
    @test isapprox(-tmp, ftmp[1], rtol=1e-5)
end