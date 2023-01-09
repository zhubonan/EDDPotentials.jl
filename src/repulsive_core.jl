
"""
Core repulsion energy
"""
function core_repulsion(r, c)
    if r < c
        return (c / r - 1)^12
    end
    return 0
end

"""
Core repulsion gradient
"""
function gcore_repulsion(r, c)
    if r < c
        return -12 * (c / r - 1)^11 / (r * r) * c
    end
    return 0
end

struct CoreReplusion{F,G}
    f::F
    g::G
    rcut::Float64
    a::Float64
end

"""
    CoreReplusion(rcut)

Construct a core repulsion with a specific cut off radius.
"""
function CoreReplusion(rcut=2.0)
    CoreReplusion(core_repulsion, gcore_repulsion, Float64(rcut), 1.0)
end
