
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


"""
Compute and update the hard core repulsion
"""
function _hard_core_update!(ecore, fcore, score, iat, jat, rij, vij, modvij, core)

    # forces
    # Note that we add the negative size here since F=-dE/dx
    gcore = -core.g(rij, core.rcut) * core.a
    if iat != jat
        #@inbounds for elm = 1:length(modvij)
        for elm = 1:length(modvij)
            # Newton's second law - only need to update this atom
            # as we also go through the same ij pair with ji 
            fcore[elm, iat] -= 2 * gcore * modvij[elm]
            #fcore[elm, jat] +=  gcore * modvij[elm]
        end
    end
    for elm1 = 1:3
        for elm2 = 1:3
            #@inbounds 
            score[elm1, elm2, iat] += vij[elm1] * modvij[elm2] * gcore
        end
    end
    ecore[iat] = core.f(rij, core.rcut) * core.a
end