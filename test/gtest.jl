#=
Test analytical gradient vs numerical graidents
=#
using LinearAlgebra
using CellBase
using CellTools
using CellTools: get_energy, get_cell
using NLSolversBase
using Test
using Flux

function sv(s, cf)
    tmp = [0 0.1  
    0 1  
    0.1 0]
    Smat = diagm([1., 1., 1.])
    Smat[:] .+= s
    cell = Cell(Lattice(2, 2, 2), [:H, :H], tmp)
    frac = CellBase.get_scaled_positions(cell)
    cell.lattice.matrix .= Smat * cellmat(cell)
    CellBase.set_scaled_positions!(cell, frac)

    nfe = CellTools.nfeatures(cf[1])
    nat = nions(cell)
    nl = CellBase.NeighbourList(cell, 3.0;savevec=true)
    
    f = zeros(nfe, nat)
    g = zeros(nfe, nat, 3, nat)
    s = zeros(nfe, nat, 3, 3, nat)
    CellTools.feature_vector_and_gradients!(f, g, s, cf, cell;nl)
    f[:]
end

function sv_me(s, me)
    pos = [0 0.1  
    0 1  
    0.1 0]
    Smat = diagm([1., 1., 1.])
    Smat[:] .+= s

    cell = Cell(Lattice(2, 2, 2), [:H, :H], pos)
    set_cellmat!(cell, Smat * cellmat(cell))

    cf = CellTools.CellFeature([:H];p2=2:4, p3=0:-1, q3=0:-1)
    cw = CellTools.CellWorkSpace(cell;cf=cf, rcut=5.0, nmax=500)

    calc = CellTools.CellCalculator(cw, me)
    CellTools.calculate!(calc)
    CellTools.get_energy(calc)
end


function fv(pos, cf)
    tmp = [0 0.1  
    0 1  
    0.1 0]
    cell = Cell(Lattice(2, 2, 2), [:H, :H], tmp)
    cell.positions[:] .= pos
    nl = CellBase.NeighbourList(cell, 3.0;savevec=true)
    nfe = CellTools.nfeatures(cf[1])
    nat = nions(cell)

    f = zeros(nfe, nat)
    g = zeros(nfe, nat, 3, nat)
    s = zeros(nfe, nat, 3, 3, nat)
    CellTools.feature_vector_and_gradients!(f, g, s, cf, cell;nl)
    f[:]
end


@testset "Gradients" begin
    pos = [0 0.1  
        0 1  
        0.1 0]
    cell = Cell(Lattice(2, 2, 2), [:H, :H], pos)
    nl = CellBase.NeighbourList(cell, 3.0;savevec=true)
    cf = CellTools.CellFeature([:H];p2=2:4)

    twof = cf.two_body

    nfe = CellTools.nfeatures(twof[1])
    nat = nions(cell)

    fvecs = zeros(nfe, nat)
    gvecs = zeros(nfe, nat, 3, nat)
    svecs = zeros(nfe, nat, 3, 3, nat)
    CellTools.feature_vector_and_gradients!(fvecs, gvecs, svecs, twof, cell;nl)
    p0 = cell.positions[:]
    od = NLSolversBase.OnceDifferentiable( x -> fv(x, twof), p0, fv(p0, twof); inplace=false)

    jac = NLSolversBase.jacobian!(od, p0)
    # Check numberical diff is below tolerance
    @test maximum(abs.(vec(jac) .- vec(gvecs))) < 1e-8


    ### Check for stress


    s0 = zeros(9)
    od = NLSolversBase.OnceDifferentiable( x -> sv(x, twof), s0, sv(s0, twof); inplace=false)
    sjac = reshape(NLSolversBase.jacobian!(od, s0), sum(nfe), 2, 3,3)
    svtot = sum(svecs,dims=5)
    s1 = svtot[1, 1, :, :] 
    sj1 = sjac[1, 1, :, :]
    @test maximum(abs.(vec(s1) .- vec(sj1))) < 1e-8

    ## Tests for three body interactions

    threeof = cf.three_body
    nfe = CellTools.nfeatures(threeof[1])
    nat = nions(cell)

    fvecs = zeros(nfe, nat)
    gvecs = zeros(nfe, nat, 3, nat)
    svecs = zeros(nfe, nat, 3, 3, nat)
    CellTools.feature_vector_and_gradients!(fvecs, gvecs, svecs, threeof, cell;nl)
    gbuffer = zeros(3, nfe)
    gbuffer = zeros( nfe)
    p0 = copy(cell.positions[:])
    od = NLSolversBase.OnceDifferentiable(x -> fv(x, threeof), p0, fv(p0, threeof); inplace=false)
    jac = NLSolversBase.jacobian!(od, p0)
    # Check numberical diff is below tolerance
    @test maximum(abs.(vec(jac) .- vec(gvecs))) < 1e-8


    s0 = zeros(9)
    od = NLSolversBase.OnceDifferentiable(x -> sv(x, threeof), s0, sv(s0, threeof); inplace=false)
    sjac = reshape(NLSolversBase.jacobian!(od, s0), sum(nfe), 2, 3,3)
    svtot = sum(svecs,dims=5)
    @test maximum(abs.(vec(sjac) .- vec(svtot))) < 1e-6
end



@testset "Force and Stress" begin
    pos = [0 0.1  
    0 1  
    0.1 0]
    cell = Cell(Lattice(2, 2, 2), [:H, :H], pos)
    frac = CellBase.get_scaled_positions(cell) 
    cf = CellTools.CellFeature([:H];p2=2:4, p3=0:-1, q3=0:-1)
    cw = CellTools.CellWorkSpace(cell;cf=cf, rcut=5.0, nmax=500)
    ntot = CellTools.nfeatures(cw)

    model = Dense(ones(1, ntot))
    me = CellTools.ModelEnsemble(model=model)

    calc = CellTools.CellCalculator(cw, me)
    CellTools.calculate!(calc)
    eng = CellTools.get_energy(calc)
    stress = copy(CellTools.get_stress(calc))
    forces = copy(CellTools.get_forces(calc))

    # Check force consistency
    delta = 1e-9
    pos[1] = delta
    CellTools.calculate!(calc)

    function energy(pos)
        cell.positions .= pos
        CellTools.get_energy(calc)
    end

    p0 = CellTools.get_positions(calc)
    od = OnceDifferentiable(energy, p0, energy(p0))
    grad= NLSolversBase.gradient(od, p0)
    @test maximum(grad .+ forces) < 1e-7

    cell.positions .= p0


    orig_cell = copy(cellmat(get_cell(calc)))

    s0 = zeros(3, 3)[:]
    od = OnceDifferentiable(x -> sv_me(x, me), s0, sv_me(s0, me);inplace=false)
    grad= NLSolversBase.gradient(od, s0)
    st = stress * volume(get_cell(calc)) 
    @test maximum(abs.(grad .+ vec(st))) < 1e-4

    # Test for the wrapper 

    vc = CellTools.VariableLatticeFilter(calc)
    global eforce = CellTools.get_forces(vc)
    epos = CellTools.get_positions(vc)

    function eeng(p)
        CellTools.set_positions!(vc, reshape(p, 3, :))
        CellTools.get_energy(vc)
    end
    od = OnceDifferentiable(eeng, epos, eeng(epos);inplace=false)
    global grad= NLSolversBase.gradient(od, epos)

    @show grad[:]
    @show eforce[:]
end