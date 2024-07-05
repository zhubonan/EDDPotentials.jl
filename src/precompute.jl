#=
Precompute the powers of the transformed distances

Ideally, save the computation during the feature generation time??
=#
using CellBase
using LinearAlgebra


struct PowerPrecompute{T}
    powers::Vector{T}
    centres::Vector{Int}
    neighbours::Matrix{Int}
    distance_powers::Array{T, 3}
    vectors::Array{T, 3}
end

function PowerPrecompute{F}(cf, rcut, nl::NeighbourList)
    scaled_powers = []
    for f in cf.two_body
        append!(scaled_powers, f.p)
    end
    for f in cf.three_body
        append!(scaled_powers, f.p)
        append!(scaled_powers, f.q)
    end
    scaled_powers = unique(scaled_powers)
    # Assume they all have the same rcut
    rcut = cf.two_body[1].rcut
    f = cf.two_body[1].f
    nat = size(nl.neighbours, 2)

    # Precompute the distance powers
    npowers = length(scaled_powers)
    orig_dist_powers = zeros(eltype(nl.distances), npowers, size(nl.distance)...)
    for (i, p) in enumerate(scaled_powers)
        orig_dist_powers[i, :, :] = f.(nl.distances) .^ p
    end


    nneigh = size(nl.neighbours, 1)  # Maximum number of neighbours
    nextended = size(nl.ea.positions, 2)  # Number of extended points

    neighbours = zeros(Int, nneigh, nextended )
    vectors = zeros(SVector{F}, nneigh, nextended )
    distance_powers = zeros(Int, length(scaled_powers), nneigh, nextended )
    centres = zeros(Int, nextended)
    for iat in 1:nat
        i1 = 1
        for (jat, jextend, rij, vij) in CellBase.eachneighbourvector(nl, iat)
            i2 = 1
            centres[i1] = jextend
            for (kat, kextended, rik, vik) in CellBase.eachneighbourvector(nl, iat)
                neighbours[i2,i1] = kextended
                d = vik - vij
                vectors[i2, i1] = d
                # Record the precomputed powers
                dmod = f(sqrt(dot(d, d)), rcut)
                for (i3, p) in enumerate(scaled_powers)
                    distance_powers[i3, i2, i1] = dmod ^ p
                end
                i2 += 1
            end
        end
    end
    PowerPrecompute(scaled_powers, extended_neighbours, extended_neighbours_distances, extended_neighbours_vectors)
end