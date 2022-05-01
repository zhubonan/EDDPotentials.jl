#=
Test analytical gradient vs numerical graidents
=#
using LinearAlgebra
using CellBase
using CellTools
using NLSolversBase

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

function fv(pos; cell=cell)
    tmp = [0 0.1  
       0 1  
       0.1 0]
    cell = Cell(Lattice(2, 2, 2), [:H, :H], tmp)
    cell.positions[:] .= pos
    nl = CellBase.NeighbourList(cell, 3.0;savevec=true)
    
    fvecs = zeros(nfe, nat)
    gvecs = zeros(nfe, nat, 3, nat)
    svecs = zeros(nfe, nat, 3, 3, nat)
    CellTools.feature_vector_and_gradients!(fvecs, gvecs, svecs, twof, cell;nl)
    fvecs[:]
end
p0 = cell.positions[:]
od = NLSolversBase.OnceDifferentiable(fv, p0, fv(p0); inplace=false)

jac = NLSolversBase.jacobian!(od, p0)
# Check numberical diff is below tolerance
@assert maximum(abs.(vec(jac) .- vec(gvecs))) < 1e-8


### Check for stress

function sv(s; cell=cell)
    tmp = [0 0.1  
       0 1  
       0.1 0]
    Smat = diagm([1., 1., 1.])
    Smat[:] .+= s
    cell = Cell(Lattice(2, 2, 2), [:H, :H], tmp)
    frac = CellBase.get_scaled_positions(cell)
    cell.lattice.matrix .= cellmat(cell) * Smat
    CellBase.set_scaled_positions!(cell, frac)

    nl = CellBase.NeighbourList(cell, 3.0;savevec=true)
    
    fvecs = zeros(nfe, nat)
    gvecs = zeros(nfe, nat, 3, nat)
    svecs = zeros(nfe, nat, 3, 3, nat)
    CellTools.feature_vector_and_gradients!(fvecs, gvecs, svecs, twof, cell;nl)
    fvecs[:]
end

s0 = zeros(9)
od = NLSolversBase.OnceDifferentiable(sv, s0, sv(s0); inplace=false)
sjac = reshape(NLSolversBase.jacobian!(od, s0), 3, 2, 3,3)
svtot = sum(svecs,dims=5)
s1 = svtot[1, 1, :, :] 
sj1 = sjac[1, 1, :, :]
@assert maximum(abs.(vec(s1) .- vec(sj1))) < 1e-8