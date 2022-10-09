using LinearAlgebra
using CellBase
using EDDP
using EDDP: get_energy, get_cell
using NLSolversBase
using Test
using Flux
#=
Something weird about these Tests
1. Using a smaller cell size seem to fail the test which is not the case with only one species?
2. Check the code when handling multiple features? 
=#

"Finite difference energy with displacements"
function _fd_energy(vc, p)
    posb = EDDP.get_positions(vc)
    EDDP.set_positions!(vc, reshape(p, 3, :))
    out = get_energy(vc)
    EDDP.set_positions!(vc, posb)
    out
end

function _fd_strain(calc, s)
    cell = EDDP.get_cell(calc)
    # This is a copy
    cm = EDDP.get_cellmat(cell)
    pos = EDDP.get_positions(cell)
    smat = diagm([1., 1., 1.])
    smat[:] .+= s
    set_cellmat!(cell, smat * cm;scale_positions=true)
    eng = get_energy(calc)
    set_cellmat!(cell, cm)
    CellBase.set_positions!(cell, pos)
    eng
end

function _fd_features(calc, p)
    cell = EDDP.get_cell(calc)
    pbak = copy(cell.positions)
    cell.positions[:] .= p 
    EDDP.update_feature_vector!(calc;rebuild_nl=true, gradients=true)
    cell.positions .= pbak
    copy(calc.force_buffer.fvec)
end

function _fd_features_strain(calc, s)
    cell = EDDP.get_cell(calc)
    # This is a copy
    cm = EDDP.get_cellmat(cell)
    pos = EDDP.get_positions(cell)
    smat = diagm([1., 1., 1.])
    smat[:] .+= s
    set_cellmat!(cell, smat * cm;scale_positions=true)
    EDDP.update_feature_vector!(calc;rebuild_nl=true, gradients=true)
    fv = copy(calc.force_buffer.fvec)
    set_cellmat!(cell, cm)
    CellBase.set_positions!(cell, pos)
    fv
end

include("utils.jl")

@testset "Force buffer" begin

    cell = _h2_cell()
    cf = _generate_cf(cell)
    fvec = vcat(EDDP.feature_vector(cf.two_body, cell), EDDP.feature_vector(cf.three_body, cell))
    fb = EDDP.ForceBuffer(fvec;ndims=3)

    fvec, gvec, stotv = EDDP.compute_fv_gv!(fb, cf.two_body, cf.three_body, cell)

    # Gradients
    gvec = fb.gvec
    stotv = fb.stotv

    function fv(cell, cf, pos)
        pb = get_positions(cell)
        cell.positions[:] .= vec(pos)
        fvec = vcat(EDDP.feature_vector(cf.two_body, cell), 
                   EDDP.feature_vector(cf.three_body, cell))
        set_positions!(cell, pb)
        fvec
    end

    function sv(cell, cf, s)
        cb = get_cellmat(cell)
        pb = get_positions(cell)
        smat = diagm([1., 1., 1.])
        smat[:] .+= s
        set_cellmat!(cell, smat * cb;scale_positions=true)
        fvec = vcat(EDDP.feature_vector(cf.two_body, cell), 
                   EDDP.feature_vector(cf.three_body, cell))
        set_cellmat!(cell, cb;scale_positions=true)
        set_positions!(cell, pb)
        fvec
    end


    p0 = cell.positions
    od = NLSolversBase.OnceDifferentiable( x -> fv(cell, cf, x), p0, fv(cell, cf, p0); 
                                          inplace=false)
    jac = reshape(NLSolversBase.jacobian!(od, p0), 3, 2, 3, 2)
    @test allclose(permutedims(gvec, [2,3,1,4]), jac;atol=1e-7)

    s0 = zeros(9)
    od = NLSolversBase.OnceDifferentiable( x -> sv(cell, cf, x), s0, sv(cell, cf, s0); 
                                          inplace=false)
    jac = NLSolversBase.jacobian!(od, s0)
    @test allclose(vec(jac), vec(permutedims(stotv, [3,4,1,2])), atol=1e-3)

end

@testset "Gradients" begin

global    calc = _get_calc()
    EDDP.calculate!(calc;forces=true)

    gtot = copy(calc.force_buffer.gvec)

    cell = EDDP.get_cell(calc)
    p0 = cell.positions[:]

 
    # Numerical differentiations
    od = NLSolversBase.OnceDifferentiable( x -> _fd_features(calc, x), p0, _fd_features(calc, p0); 
                                          inplace=false)
    jac = NLSolversBase.jacobian!(od, p0)

    # Check consistency with numerical differentiation
    gtot = permutedims(gtot, [2,3,1,4])
    jac = reshape(jac, size(gtot))

    _, n2, n2 = EDDP.feature_size(calc.cf)
    @test all(isapprox.(jac, gtot,  atol=1e-5))


    # ### Check for stress
    s0 = zeros(9)
    od = NLSolversBase.OnceDifferentiable( x -> _fd_features_strain(calc, x), s0, _fd_features_strain(calc, s0); inplace=false)
    sjac = reshape(NLSolversBase.jacobian!(od, s0),
                  sum(EDDP.feature_size(calc.cf)[2:end]), length(get_cell(calc)), 3,3)
    sd = calc.force_buffer.stotv
    sd = permutedims(sd, [3,4,1,2])
    # Accuracy of this numerical differentiation is not so good...
    @test allclose(sjac, sd, atol=1e-2)
end


@testset "Forces" begin

    calc = _get_calc()

    ntot = EDDP.nfeatures(calc.cf)    
    model = Chain(Dense(ones(1, ntot)))
    itf = EDDP.ManualFluxBackPropInterface(model)

    forces = copy(EDDP.get_forces(calc))
    stress = copy(EDDP.get_stress(calc))

    # Test the total force
    p0 = EDDP.get_positions(calc)
    od = OnceDifferentiable(x -> _fd_energy(calc, x), p0, _fd_energy(calc, p0))
    grad= NLSolversBase.gradient(od, p0)
    @test allclose(grad, -forces, atol=1e-6)

    # Test the total stress
    s0 = zeros(3,3)[:]
    od = OnceDifferentiable(x -> _fd_strain(calc, x), s0, _fd_strain(calc, s0), inplace=false)
    grad= NLSolversBase.gradient(od, s0) ./ volume(get_cell(calc))
    grad2 = grad
    @test allclose(grad, -vec(stress), atol=1e-4)

    # Test wrapper
    vc = EDDP.VariableCellCalc(calc)
    epos = EDDP.get_positions(vc)
    eforce = copy(EDDP.get_forces(vc))

    od = OnceDifferentiable(x -> _fd_energy(vc, x), epos, _fd_energy(vc, epos);inplace=false)
    grad= NLSolversBase.gradient(od, epos)
    @test allclose(grad, -eforce, atol=5e-3, rtol=1e-4)


    # Test using optimised routine
end

include("utils.jl")
calc = _get_calc()
calc

calc.energy_calculated = false
calc.forces_calculated = false
EDDP.calculate!(calc)
calc.force_buffer.forces
calc.force_buffer.stress

calc.energy_calculated = false
calc.forces_calculated = false
EDDP.calculate_old!(calc)