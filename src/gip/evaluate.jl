#=
Support code for using the potentials for effcient energy/force/stress
calculations.
=#
import CellBase
export rebuild!, update!

"""
Combination of a cell, its associated neighbour list and feature vectors

This is to allow efficient re-calculation of the feature vectors without reallocating memory.
"""
struct CellWorkSpace{T, N, M}
    cell::T
    nl::N
    cf::M
    one_body
    two_body
    three_body
    v
end


function CellWorkSpace(cell::Cell;cf, rcut, nmax=100, savevec=false) 
    nl = NeighbourList(cell, rcut, nmax;savevec)
    us = unique(atomic_numbers(cell))
    one_body = zeros(length(us), nions(cell))
    two_body = feature_vector(cf.two_body, cell;nl=nl)
    three_body = feature_vector(cf.three_body, cell;nl=nl)
    CellWorkSpace(cell, nl, cf, one_body, two_body, three_body, vcat(one_body, two_body, three_body))
end

CellBase.rebuild!(cw::CellWorkSpace) = CellBase.rebuild!(cw.nl, cw.cell)
CellBase.update!(cw::CellWorkSpace) = CellBase.update!(cw.nl, cw.cell)

"""
    update_feature_vector!(wt::CellWorkSpace)

Update the feature vectors after atomic displacements.
"""
function update_feature_vector!(wt::CellWorkSpace;rebuild_nl=false)
    rebuild_nl ? rebuild!(wt) : update!(wt)

    one_body_vectors!(wt.one_body, wt.cell)
    feature_vector!(wt.two_body, wt.cf.two_body, wt.cell;wt.nl)
    feature_vector!(wt.three_body, wt.cf.three_body, wt.cell;wt.nl)
    # Block update
    i = 1
    l = size(wt.one_body, 1)
    wt.v[i: i+l-1, :] .= wt.one_body
    i += l
    l = size(wt.two_body, 1)
    wt.v[i: i+l-1, :] .= wt.two_body
    i += l
    l = size(wt.three_body, 1)
    wt.v[i: i+l-1, :] .= wt.three_body
    wt.v
end