#=
Building examples
=#
using CellBase

function _get_silicon()
    a = 0.543
    pos_frac = [
        0 0 0
        0.5 0.5 0
        0.25 0.75 0.25
        0.75 0.25 0.25
        0.5 0. 0.5
        0. 0.5 0.5
        0.25 0.25 0.75
        0.75 0.75 0.75
    ]
    pos_frac = collect(transpose(pos_frac))
    latt = Lattice(5.43, 5.43, 5.43, 90., 90., 90.)
    pos_abs = cellmat(latt) * pos_frac
    Cell(latt, repeat([:Si], 8), pos_abs)
end

"""
    quikbuild(name)

Return example structures - useful for testing...
"""
function quikbuild(name)
    if name == "silicon"
        return _get_silicon()
    end
end