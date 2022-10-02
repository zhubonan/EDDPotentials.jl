#=
Common functions for helping tests
=#
using EDDP
using CellBase

function _h2_cell(l=2.0, factor=1.0)
    tmp = Float64[0 0.1  
    0 1  
    0.1 0] .* factor
    Cell(Lattice(l, l, l), [:H, :H], tmp)
end

function _generate_cf(cell::Cell)
    cf = EDDP.CellFeature(
        EDDP.FeatureOptions(elements=unique(species(cell)), p2=[6, 12], q3=[2], p3=[2])
    )
    cf
end