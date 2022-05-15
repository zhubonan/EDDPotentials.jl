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
    if eltype(cf) <: CellTools.TwoBodyFeature
        CellTools.compute_two_body_fv_gv!(f, g, s, cf, cell;nl)
    else
        CellTools.compute_three_body_fv_gv!(f, g, s, cf, cell;nl)
    end
    f[:]
end

function sv_me(s, me, T=Float64)
    pos = [0 0.1  
    0 1  
    0.1 0]
    Smat = diagm([1., 1., 1.])
    Smat[:] .+= s

    cell = Cell(Lattice(2, 2, 2), [:H, :H], pos)
    set_cellmat!(cell, Smat * cellmat(cell))

    cf = CellTools.CellFeature([:H];p2=2:4, p3=0:-1, q3=0:-1)
    cw = CellTools.CellWorkSpace{T}(cell;cf=cf, rcut=5.0, nmax=500)

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
    if eltype(cf) <: CellTools.TwoBodyFeature
        CellTools.compute_two_body_fv_gv!(f, g, s, cf, cell;nl)
    else
        CellTools.compute_three_body_fv_gv!(f, g, s, cf, cell;nl)
    end
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
    CellTools.compute_two_body_fv_gv!(fvecs, gvecs, svecs, twof, cell;nl)
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
    CellTools.compute_three_body_fv_gv!(fvecs, gvecs, svecs, threeof, cell;nl)
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

    function allclose(A, B;tol=1e-7)
        maxdiv = maximum(abs.(A .- B))
        out = maxdiv < tol
        if !out
            @show maxdiv A B 
        end
        out
    end


    pos = [0 0.1  
    0 1  
    0.1 0]
    cell = Cell(Lattice(2, 2, 2), [:H, :H], pos)
    frac = CellBase.get_scaled_positions(cell) 
    cf = CellTools.CellFeature([:H];p2=2:4, p3=0:-1, q3=0:-1)
    cw = CellTools.CellWorkSpace{Float64}(cell;cf=cf, rcut=5.0, nmax=500)
    # Also test using Float32 for calculation
    cw32 = CellTools.CellWorkSpace{Float32}(cell;cf=cf, rcut=5.0, nmax=500)

    ntot = CellTools.nfeatures(cw)
    model = Chain(Dense(ones(1, ntot)))
    model32 = Chain(Dense(ones(Float32, 1, ntot)))

    me = CellTools.ModelEnsemble(model=model)
    me32 = CellTools.ModelEnsemble(model=model32)

    calc = CellTools.CellCalculator(cw, me)
    calc32 = CellTools.CellCalculator(cw32, me32)

    CellTools.calculate!(calc)
    CellTools.calculate!(calc32)
    eng = CellTools.get_energy(calc)
    eng32 = CellTools.get_energy(calc32)
    stress = copy(CellTools.get_stress(calc))
    forces = copy(CellTools.get_forces(calc))

    stress32 = copy(CellTools.get_stress(calc32))
    forces32 = copy(CellTools.get_forces(calc32))

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

    # Check the consistency of the neutron network derivatives
    g1 = Flux.gradient(x -> sum(model(x)), cw.v)[1]
    g2 = vcat(calc.g2, calc.g3)
    g3 = vcat(calc32.g2, calc32.g3)

    @test allclose(g1, g2)
    @test allclose(g3, g2)

    # Check the actual forces
    @test allclose(grad, -forces)
    @test allclose(grad, -forces32, tol=1e-6)
    @test allclose(forces, forces32, tol=1e-6)

    cell.positions .= p0


    orig_cell = copy(cellmat(get_cell(calc)))

    s0 = zeros(3, 3)[:]
    od = OnceDifferentiable(x -> sv_me(x, me), s0, sv_me(s0, me);inplace=false)
    grad= NLSolversBase.gradient(od, s0) ./ volume(get_cell(calc))
    st = stress  
    st32 = stress32 
    @test allclose(grad, -vec(st), tol=1e-5)
    @test allclose(grad, -vec(st32), tol=1e-5)
    @test allclose(st, st32, tol=1e-5)

    # Test for the wrapper 

    vc = CellTools.VariableLatticeFilter(calc)
    vc32 = CellTools.VariableLatticeFilter(calc32)

    eforce = CellTools.get_forces(vc)
    eforce32 = CellTools.get_forces(vc32)
    epos = CellTools.get_positions(vc)

    function eeng(p)
        CellTools.set_positions!(vc, reshape(p, 3, :))
        CellTools.get_energy(vc)
    end
    od = OnceDifferentiable(eeng, epos, eeng(epos);inplace=false)
    grad= NLSolversBase.gradient(od, epos)

    @test allclose(grad, -eforce, tol=1e-4)
    @test allclose(eforce32, eforce, tol=1e-5)
end