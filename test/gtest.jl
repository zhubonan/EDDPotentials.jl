#=
Test analytical gradient vs numerical graidents
=#
using LinearAlgebra
using CellBase
using CellTools
using NLSolversBase
using Test

function sv(s, cf)
    tmp = [0 0.1  
    0 1  
    0.1 0]
    Smat = diagm([1., 1., 1.])
    Smat[:] .+= s
    cell = Cell(Lattice(2, 2, 2), [:H, :H], tmp)
    frac = CellBase.get_scaled_positions(cell)
    cell.lattice.matrix .= cellmat(cell) * Smat
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
    p0 = copy(cell.positions[:])
    od = NLSolversBase.OnceDifferentiable(x -> fv(x, threeof), p0, fv(p0, threeof); inplace=false)
    jac = NLSolversBase.jacobian!(od, p0)
    # Check numberical diff is below tolerance
    @test maximum(abs.(vec(jac) .- vec(gvecs))) < 1e-8


    s0 = zeros(9)
    od = NLSolversBase.OnceDifferentiable(x -> sv(x, threeof), s0, sv(s0, threeof); inplace=false)
    sjac = reshape(NLSolversBase.jacobian!(od, s0), sum(nfe), 2, 3,3)
    svtot = sum(svecs,dims=5)
    s1 = svtot[1, 1, :, :] 
    sj1 = sjac[1, 1, :, :]
    @test maximum(abs.(vec(s1) .- vec(sj1))) < 1e-8
end