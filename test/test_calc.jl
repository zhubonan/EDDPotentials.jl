using EDDP
using EDDP: get_cell, get_forces, get_stress, get_energy
using Test
using CellBase
using Flux
include("utils.jl")

@testset "Calc" begin
    cell = _h2_cell()

    cf = EDDP.CellFeature(
        EDDP.FeatureOptions(elements=unique(species(cell)), p2=[2], q3=[2, 3], p3=[2, 3])
    )

    function _test_forces_fd(calc, amp=1e-7, rtol=1e-5)
        ftmp = copy(EDDP.get_forces(calc))
        etmp = EDDP.get_energy(calc)
        positions(get_cell(calc))[1] += amp
        @test EDDP._need_calc(calc, true)
        tmp = (get_energy(calc) - etmp) / amp
        @test -tmp ≈ ftmp[1] rtol=rtol
    end


    @testset "MBP" begin
        nnitf = EDDP.ManualFluxBackPropInterface(Chain(
            Dense(rand(5, EDDP.nfeatures(cf))), Dense(rand(1, 5))
            )
            )
        calc = EDDP.NNCalc(cell, cf, nnitf;core=nothing)
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
        _test_forces_fd(calc)
        global this_calc
        this_calc=calc
    end
    @testset "Linear" begin
        nnitf = EDDP.LinearInterface(rand(EDDP.nfeatures(cf)))
        calc = EDDP.NNCalc(cell, cf, nnitf;core=nothing)
        eng = get_energy(calc)
        forces = get_forces(calc)
        stress = get_stress(calc)

        @test isa(eng, Float64)

        @test all(isapprox.(sum(forces, dims=2), 0, atol=1e-10 )) 
        @test size(stress) == (3,3)
        @test any(stress .!== 0.)

        _test_forces_fd(calc)
    end
    
    @testset "VCFilter" begin
        nnitf = EDDP.LinearInterface(rand(EDDP.nfeatures(cf)))
        calc = EDDP.NNCalc(cell, cf, nnitf)
        filter =  EDDP.VariableCellCalc(calc)
        _test_forces_fd(filter)
        stress = get_stress(filter)
        @test size(stress) == (3,3)
        @test any(stress .!== 0.)

        # Sizes of the positions
        pos = get_positions(filter)
        @test size(pos) == (3, 5)
        pos[1] += 1e-9
        set_positions!(filter, pos)
        @test EDDP._need_calc(filter, false)
    end
end


@testset "Relax" begin
    cell = _h2_cell(10, 1.5)
    cf = EDDP.CellFeature(
        EDDP.FeatureOptions(elements=unique(species(cell)), rcut2=3.5, p2=[6, 12], p3=[], q3=[])
    )

    nnitf = EDDP.LinearInterface(rand(EDDP.nfeatures(cf)))
    # Attractive potential with -5f(x)^6 + f(x)^12
    EDDP.setparamvector!(nnitf, [0, -5, 1])
    calc = EDDP.NNCalc(cell, cf, nnitf)
    # Perform relaxation
    EDDP.optimise!(calc)

    # Expected distance
    rexp = (1 - (5/2/2^6)^(1/6)) * 3.5
    dd = distance_between(cell[1], cell[2])
    @test dd ≈ rexp atol=1e-6

    # Test recording trajectory
    cell = _h2_cell(10, 1.5)
    calc = EDDP.NNCalc(cell, cf, nnitf)
    global res, traj
    res, traj = EDDP.optimise!(calc, record_trajectory=true)
    @test res.g_converged

    dd = distance_between(cell[1], cell[2])
    @test dd ≈ rexp atol=1e-6

    @test length(traj) > 1
    @test :enthalpy in keys(traj[1].metadata)
end