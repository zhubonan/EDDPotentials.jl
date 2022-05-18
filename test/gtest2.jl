using LinearAlgebra
using CellBase
using CellTools
using CellTools: get_energy, get_cell
using NLSolversBase
using Test
using Flux
#=
Something weird about these Tests
1. Using a smaller cell size seem to fail the test which is not the case with only one species?
2. Check the code when handling multiple features? 
=#

function genworkspace()
      pos = [0 0.1 0.3 
             0 1  3
             0.1 0 0.2]
    cell = Cell(Lattice(3.0, 2.0, 3.0), [:H, :H, :O], pos)
    cf = CellTools.CellFeature([:H, :O];p2=2:4, p3=2:4, q3=2:4, rcut3=4.0, rcut2=4.0)
    pop!(cf.three_body)
    pop!(cf.three_body)
    pop!(cf.three_body)
    CellTools.CellWorkSpace(cell;cf, rcut=4.0)
end

@testset "Gradients2" begin

    global wk = genworkspace()
    CellTools.update_feature_vector!(wk;rebuild_nl=true)

    g2 = wk.two_body_fbuffer.gvec
    g3 = wk.three_body_fbuffer.gvec
    global gtot = cat(g2, g3, dims=1)

    cell = CellTools.get_cell(wk)
    p0 = cell.positions[:]

    function update_v_with_pos(workspace, p)
        cell = CellTools.get_cell(workspace)
        pbak = copy(cell.positions)
        cell.positions[:] .= p 
        CellTools.update_feature_vector!(workspace;)
        cell.positions .= pbak
        workspace.v
    end
    od = NLSolversBase.OnceDifferentiable( x -> update_v_with_pos(wk, x), p0, update_v_with_pos(wk, p0); 
                                          inplace=false)
    # Numerical differentiations
    global jac = NLSolversBase.jacobian!(od, p0)
    jac = reshape(jac, size(gtot))
    # Two body part
    n2 = CellTools.nbodyfeatures(wk, 2)
    n3 = CellTools.nbodyfeatures(wk, 3)
    @test maximum(abs.(jac[1:n2, :, :, :] .- gtot[1:n2, :, :, :])) < 1e-6
    # Three body part
    @test maximum(abs.(jac[n2+1:end, :, :, :] .- gtot[n2+1:end, :, :, :])) < 1e-6


    # ### Check for stress


    # s0 = zeros(9)
    # od = NLSolversBase.OnceDifferentiable( x -> sv(x, twof), s0, sv(s0, twof); inplace=false)
    # sjac = reshape(NLSolversBase.jacobian!(od, s0), sum(nfe), 2, 3,3)
    # svtot = sum(svecs,dims=5)
    # s1 = svtot[1, 1, :, :] 
    # sj1 = sjac[1, 1, :, :]
    # @test maximum(abs.(vec(s1) .- vec(sj1))) < 1e-8

    # ## Tests for three body interactions

    # threeof = cf.three_body
    # nfe = CellTools.nfeatures(threeof[1])
    # nat = nions(cell)

    # fvecs = zeros(nfe, nat)
    # gvecs = zeros(nfe, nat, 3, nat)
    # svecs = zeros(nfe, nat, 3, 3, nat)
    # CellTools.compute_three_body_fv_gv!(fvecs, gvecs, svecs, threeof, cell;nl)
    # gbuffer = zeros(3, nfe)
    # gbuffer = zeros( nfe)
    # p0 = copy(cell.positions[:])
    # od = NLSolversBase.OnceDifferentiable(x -> fv(x, threeof), p0, fv(p0, threeof); inplace=false)
    # jac = NLSolversBase.jacobian!(od, p0)
    # # Check numberical diff is below tolerance
    # @test maximum(abs.(vec(jac) .- vec(gvecs))) < 1e-8


    # s0 = zeros(9)
    # od = NLSolversBase.OnceDifferentiable(x -> sv(x, threeof), s0, sv(s0, threeof); inplace=false)
    # sjac = reshape(NLSolversBase.jacobian!(od, s0), sum(nfe), 2, 3,3)
    # svtot = sum(svecs,dims=5)
    # @test maximum(abs.(vec(sjac) .- vec(svtot))) < 1e-6
end
