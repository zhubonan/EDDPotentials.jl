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

function allclose(A, B;tol=1e-7)
    maxdiv = maximum(abs.(A .- B))
    out = maxdiv < tol
    if !out
        @show maxdiv A[1:5] B[1:5] 
    end
    out
end 

function genworkspace()
      pos = [0 0.1 0.3 
             0 1  3
             0.1 0 0.2]
    cell = Cell(Lattice(3.15, 2.9, 3.1), [:H, :H, :O], pos)
    cf = CellTools.CellFeature([:H, :O];p2=2:4, p3=2:4, q3=2:4, rcut3=4.0, rcut2=4.0)
    CellTools.CellWorkSpace(cell;cf, rcut=CellTools.suggest_rcut(cf))
end

@testset "Gradients2" begin

    wk = genworkspace()
    CellTools.update_feature_vector!(wk;rebuild_nl=true)

    g2 = wk.two_body_fbuffer.gvec
    g3 = wk.three_body_fbuffer.gvec
    gtot = cat(g2, g3, dims=1)

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
    @test allclose(jac[1:n2, :, :, :], gtot[1:n2, :, :, :],  tol=1e-6)
    # Three body part
    @test allclose(jac[n2+1:end, :, :, :], gtot[n2+1:end, :, :, :],  tol=1e-6)


    # ### Check for stress

    function update_with_strain(workspace, s)
        cell = CellTools.get_cell(workspace)
        cm = CellTools.get_cellmat(cell)
        pos = CellTools.get_positions(cell)
        smat = diagm([1., 1., 1.])
        smat[:] .+= s
        set_cellmat!(cell, smat * cm)
        CellTools.update_feature_vector!(workspace;)
        set_cellmat!(cell, cm)
        CellBase.set_positions!(cell, pos)
        workspace.v
    end

    s0 = zeros(9)
    od = NLSolversBase.OnceDifferentiable( x -> update_with_strain(wk, x), s0, update_with_strain(wk, s0); inplace=false)
    global sjac = reshape(NLSolversBase.jacobian!(od, s0), CellTools.nfeatures(wk), 3, 3,3)
    s2 = wk.two_body_fbuffer.stotv
    s3 = wk.three_body_fbuffer.stotv
    sd = cat(s2, s3, dims=1)
    # Accuracy of this numerical differentiation is not so good...
    @test allclose(sjac, sd, tol=1e-2)
end


@testset "Forces" begin
    wk = genworkspace()
    ntot = CellTools.nfeatures(wk)    
    model = Chain(Dense(ones(1, ntot)))
    me = CellTools.ModelEnsemble(model=model)
    calc = CellTools.CellCalculator(wk, me)
    CellTools.calculate!(calc)
    stress = copy(CellTools.get_stress(calc))
    forces = copy(CellTools.get_forces(calc))

    function energy(calc, pos)
        bak = get_positions(calc)
        cell = get_cell(calc)
        cell.positions .= pos
        eng = get_energy(calc)
        cell.positions .= bak
        eng
    end

    function update_with_strain(calc, s)
        cell = CellTools.get_cell(calc)
        cm = CellTools.get_cellmat(cell)
        pos = CellTools.get_positions(cell)
        smat = diagm([1., 1., 1.])
        smat[:] .+= s
        set_cellmat!(cell, smat * cm)
        eng = get_energy(calc)
        set_cellmat!(cell, cm)
        CellBase.set_positions!(cell, pos)
        eng
    end

    # Test the total force
    p0 = CellTools.get_positions(calc)
    od = OnceDifferentiable(x -> energy(calc, x), p0, energy(calc, p0))
    grad= NLSolversBase.gradient(od, p0)
    @test allclose(grad, -forces, tol=1e-6)

    # Test the total stress
    s0 = zeros(3,3)[:]
    od = OnceDifferentiable(x -> update_with_strain(calc, x), s0, update_with_strain(calc, s0), inplace=false)
    grad= NLSolversBase.gradient(od, s0) ./ volume(get_cell(calc))
    @test allclose(grad, -vec(stress), tol=1e-5)

    # Test wrapper
    vc = CellTools.VariableLatticeFilter(calc)
    epos = CellTools.get_positions(vc)
    global eforce = CellTools.get_forces(vc)

    function eeng(vc, p)
        CellTools.set_positions!(vc, reshape(p, 3, :))
        get_energy(vc)
    end

    od = OnceDifferentiable(x -> eeng(vc, x), epos, eeng(vc, epos);inplace=false)
    global grad= NLSolversBase.gradient(od, epos)

    @test allclose(grad, -eforce, tol=5e-2)
end