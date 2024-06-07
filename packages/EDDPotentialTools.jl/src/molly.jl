#= Interface with Molly
=#

using StaticArrays
using .Molly
import .Molly

struct EDDPotentialInter{T}
    calc::T
end

function _geometry_system2calc!(calc, sys)
    cell = get_cell(calc)
    pos = zeros(3, length(sys.coords))
    for i in axes(pos, 2)
        pos[:, i] .= ustrip(sys.coords[i])
    end
    if isa(sys.boundary, Molly.CubicBoundary) ||
       isa(sys.boundary, Molly.RectangularBoundary)
        lattice = Lattice(sys.boundary[1], sys.boundary[2], sys.boundary[3])
    else
        b = sys.boundary
        lattice = Lattice(ustrip.(collect(hcat(b[1], b[2], b[3]))))
    end

    set_cellmat!(cell, cellmat(lattice))
    set_positions!(calc, pos)
end


function Molly.forces(inter::EDDPotentialInter, sys, neighbors=nothing)
    _geometry_system2calc!(inter.calc, sys)

    forces = get_forces(inter.calc)
    # Output forces
    [SVector{3}(col) * sys.force_units for col in eachcol(forces)]
end

function Molly.potential_energy(inter::EDDPotentialInter, sys, neighbours=nothing)
    _geometry_system2calc!(inter.calc, sys)
    get_energy(inter.calc) * sys.energy_units

end
