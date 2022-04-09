#=
Generalised potential
=#
using CellBase
using StaticArrays

abstract type AbstractNBodyFeature end

"""
Compunents for constructing the feature vector of two-body interactions
"""
struct TwoBodyFeature{T} <: AbstractNBodyFeature
    "Function of distance"
    f::T
    "Exponents"
    p::Vector{Float64}
    "Specie indices"
    sij_idx::Tuple{Int, Int}
    "Cut off distance"
    rcut::Float64
    np::Int
end


"Equation 7 in  the DDP paper"
fr(r::T, rcut::T) where {T} =  r <= rcut ? 2 * (1 - r / rcut) : zero(T)

TwoBodyFeature(f, p::Vector{Float64}, sij_idx::Tuple{Int, Int}, rcut::Float64) = TwoBodyFeature(f, p, sij_idx, rcut, length(p))
TwoBodyFeature(p::Vector{Float64}, sij_idx::Tuple{Int, Int}, rcut::Float64) = TwoBodyFeature(fr, p, sij_idx, rcut)

"""
Call the object to accumulate an existing feature vector
"""
function (f::TwoBodyFeature)(out::Vector, rij)
    for i in 1:nfeatures(f)
        out[i] += f.f(rij, f.rcut) ^ f.p[i]
    end
    out
end

function (f::TwoBodyFeature)(out::Vector, rij, si, sj)
    (si == f.sij_idx[1]) && (sj == f.sij_idx[2]) && f(out, rij)
    out
end


(f::TwoBodyFeature)(rij) = f(zeros(nfeatures(f)), rij)
nfeatures(f::TwoBodyFeature) = f.np


"""
Compunents for constructing the feature vector of three-body interactions
"""
struct ThreeBodyFeature{T} <: AbstractNBodyFeature
    "Function of distance"
    f::T
    "Exponents"
    p::Vector{Float64}
    q::Vector{Float64}
    "Specie indices"
    sijk_idx::Tuple{Int, Int, Int}
    "Cut off distance"
    rcut::Float64
    np::Int
    nq::Int
end

ThreeBodyFeature(f, p::Vector{Float64}, q::Vector{Float64}, sijk_idx::Tuple{Int, Int, Int}, rcut::Float64) = ThreeBodyFeature(f, p, q, sijk_idx, rcut, length(p), length(q))

nfeatures(f::ThreeBodyFeature) = f.np * f.nq

"""
    (f::ThreeBodyFeature)(out::Vector, rij, rik, rjk)

Accumulate an existing feature vector
"""
function (f::ThreeBodyFeature)(out::Vector, rij, rik, rjk)
    i = 1
    func = f.f
    rcut = f.rcut
    for m in 1:f.np
        for o in 1:f.nq  # Note that q is summed in the inner loop
            out[i] += (func(rij, rcut) ^ f.p[m]) * (func(rik, rcut) ^ f.p[m]) * (func(rjk, rcut) ^ f.q[o])
            i += 1
        end
    end
    out
end

"(f::ThreeBodyFeature)(out::Vector, rij, rik, rjk, si, sj, sk)"
function (f::ThreeBodyFeature)(out::Vector, rij, rik, rjk, si, sj, sk)
    (si == f.sijk_idx[1]) && (sj == f.sijk_idx[2]) && (sk == f.sijk_idx[3]) && f(out, rij, rik, rjk)
    out
end

"(f::ThreeBodyFeature)(rij) = f(zeros(nfeatures(f)), rij, rik, rjk)"
(f::ThreeBodyFeature)(rij) = f(zeros(nfeatures(f)), rij, rik, rjk)


"""
Map species types to integer indices
"""
struct SpeciesMap
    symbols::Vector{Symbol}
    indices::Vector{Int}
    unique::Vector{Symbol}
end


function SpeciesMap(symbols)
    us = unique(symbols)
    indices = zeros(Int, length(symbols))
    for (idx, sym) in enumerate(symbols)
        indices[idx] = findfirst(x -> x == sym, us)
    end
    SpeciesMap(symbols, indices, us)
end

"Get the mapped index for a given symbol"
index(smap::SpeciesMap, sym::Symbol) = findfirst(x -> x==sym, smap.us)
"Get the symbol index for a given integer index"
symbol(smap::SpeciesMap, sidx::Int) = smap.us[sidx]

"""
    interger_specie_index(cell::Cell)

Return an integer indexing array for the species
"""
function interger_specie_index(cell::Cell)
    sym = species(cell)
    us = unique(sym)
    out = zeros(Int, length(sym))
    for (idx, specie) in enumerate(sym)
        out[idx] = findfirst(x -> x == specie, us)
    end
    out, us
end


"""
Represent an array of points after expansion by periodic boundary
"""
struct ExtendedPointArray{T}
    "Original point indices"
    indices::Vector{Int}
    "Index of the shift"
    shiftidx::Vector{Int}
    "Shift vectors"
    shiftvecs::Vector{SVector{3, Float64}}
    "Positions"
    positions::Vector{T}
    "original Positions"
    orig_positions::Vector{T}
    "Index of of the all-zero shift vector"
    inoshift::Int
end

function Base.show(io::IO, s::ExtendedPointArray)
    print(io, "ExtendedPointArray of $(length(s.indices)) points from $(length(s.orig_positions)) points")
end

"""
    ExtendedPointArray(cell::Cell, rcut)

Constructed an ExtendedPointArray from a given structure
"""
function ExtendedPointArray(cell::Cell, rcut)

    ni = nions(cell)
    shifts = CellBase.shift_vectors(cellmat(lattice(cell)), rcut;safe=false)
    indices = zeros(Int, ni * length(shifts))
    shiftidx = zeros(Int, ni * length(shifts))
    pos_extended = zeros(eltype(positions(cell)), 3, ni * length(shifts))
    inoshift = findfirst(x -> all( x .== 0.), shifts)

    i = 1
    original_positions = sposarray(cell)
    for (idx, pos_orig) in enumerate(original_positions)   # Each original positions
        for (ishift, shiftvec) in enumerate(shifts)   # Each shift positions
            pos_extended[:, i] .= pos_orig .+ shiftvec
            indices[i] = idx
            shiftidx[i] = ishift
            i += 1
        end
    end
    ExtendedPointArray(indices, shiftidx, shifts, [SVector{3}(x) for x in eachcol(pos_extended)], original_positions, inoshift)
end


struct NeighbourList{T}
    ea::ExtendedPointArray{T}
    "Extended indice of the neighbours"
    extended_indices::Matrix{Int}
    "Original indice of the neighbours"
    orig_indices::Matrix{Int}
    "Distance to the neighbours"
    distance::Matrix{Float64}
    "Vector displacement to the neighbours"
    vectors::Array{Float64, 3}
    "Number of neighbours"
    nneigh::Vector{Int}
    "Maximum number of neighbours that can be stored"
    nmax::Int
    "Contains vector displacements or not"
    has_vectors::Bool
    rcut::Float64
end

"Number of ions in the original cell"
nions_orig(n::ExtendedPointArray) = length(n.orig_positions)
"Number of ions in the original cell"
nions_orig(n::NeighbourList) = nions_orig(n.ea) 

"Number of ions in the extended cell"
nions_extended(n::ExtendedPointArray) = length(n.positions)
"Number of ions in the extended cell"
nions_extended(n::NeighbourList) = nions_extended(n.ea)


function Base.show(io::IO, n::NeighbourList)
    print("NeighbourList of maximum sizs $(n.nmax) for $(nions_orig(n))/$(nions_extended(n)) atoms")
end

"""
    NeighbourList(ea::ExtendedPointArray, rcut, nmax=100; savevec=false)

Construct a NeighbourList from an extended point array for the points in the original cell
"""
function NeighbourList(ea::ExtendedPointArray, rcut, nmax=100; savevec=false)
    
    norig = length(ea.orig_positions)
    extended_indices = zeros(Int, nmax, norig)
    orig_indices = zeros(Int, nmax, norig)
    distance = fill(-1., nmax, norig)
    nneigh = zeros(Int, norig)

    # Save vectors or not
    savevec ? vectors = fill(-1., 3, norig, nmax) : vectors = fill(-1., 1, 1, 1)

    for (iorig, posi) in enumerate(ea.orig_positions)
        ineigh = 0
        for (j, posj) in enumerate(ea.positions)
            (ea.indices[j] == iorig) && (ea.shiftidx[j] == ea.inoshift) && continue
            dist = distance_between(posj, posi)
            # Store the information if the distance is smaller than the cut off
            if dist < rcut
                ineigh += 1
                if ineigh <= nmax
                    distance[ineigh, iorig] = dist
                    # Store the extended index
                    extended_indices[ineigh, iorig] = j
                    # Store the index of the point in the original cell
                    orig_indices[ineigh, iorig] = ea.indices[j]
                    savevec && (vectors[:, ineigh, iorig] .= posj .- posi)
                end
            end
        end
        # Store the total number of neighbours for this point
        if ineigh > nmax
            throw(ErrorException("Too many neighbours, please increase the value of nmax to at least $(ineigh)"))
        end
        nneigh[iorig] = ineigh
    end
    NeighbourList(
        ea,
        extended_indices,
        orig_indices,
        distance,
        vectors,
        nneigh,
        nmax,
        savevec,
        rcut,
    )
end


NeighbourList(cell::Cell, rcut, nmax=100;savevec=false) = NeighbourList(ExtendedPointArray(cell, rcut), rcut, nmax;savevec)

"Number of neighbours for a point"
num_neighbours(nl::NeighbourList, iorig) = nl.nneigh[iorig]


"Iterator interface for going through all neighbours"
struct NLIterator
    nl::NeighbourList
    iorig::Int
end

Base.length(nli::NLIterator) =  num_neighbours(nli.nl, nli.iorig)

function Base.iterate(nli::NLIterator, state=1)
    nl = nli.nl
    iorig = nli.iorig
    if state > nl.nneigh[nli.iorig] 
        return nothing
    end
    return (nl.orig_indices[state, iorig], nl.extended_indices[state, iorig], nl.distance[state, iorig]), state + 1
end

"""
Iterate the neighbours of a site in the original cell.
Returns a tuple of (original_index, extended_index, distance) for each iteration
"""
eachneighbour(nl::NeighbourList, iorig) = NLIterator(nl, iorig)

"""
    feature_vector(features::Vector{T}, cell::Cell) where T

Compute the feature vector for a give set of two body interactions
"""
function feature_vector(features::Vector{TwoBodyFeature{T}}, cell::Cell;nl=NeighbourList(cell, features[1].rcut)) where T
    # Feature vectors
    fvecs = [zeros(nfeatures(f)) for f in features]
    nat = natoms(cell)
    smap = SpeciesMap(species(cell))
    spidx = smap.indices
    for i = 1:nat
        for (j, jextend, rij) in eachneighbour(nl, i)
            # accumulate the feature vector
            for (nf, f) in enumerate(features)
                f(fvecs[nf], rij, spidx[i], spidx[j])
            end
        end
    end
    vcat(fvecs...)
end

"""
    feature_vector(features::Vector{T}, cell::Cell) where T

Compute the feature vector for a give set of two body interactions
"""
function feature_vector(features::Vector{ThreeBodyFeature{T}}, cell::Cell;nl=NeighbourList(cell, features[1].rcut)) where T
    # Feature vectors
    fvecs = Vector{Float64}[zeros(nfeatures(f)) for f in features]
    nat = natoms(cell)
    # Note - need to use twice the cut off to ensure distance between j-k is included
    smap = SpeciesMap(species(cell))
    spidx = smap.indices
    for i = 1:nat
        for (j, jextend, rij) in eachneighbour(nl, i)
            for (k, kextend, rik) in eachneighbour(nl, i)
                # Avoid double counting i j k is the same as i k j
                if k <= j 
                    continue
                end
                # Compute the distance between extended j and k
                rjk = distance_between(nl.ea.positions[jextend], nl.ea.positions[kextend])
                # accumulate the feature vector
                for (nf, f) in enumerate(features)
                    f(fvecs[nf], rij, rik, rjk, spidx[i], spidx[j], spidx[k])
                end
            end
        end
    end
    vcat(fvecs...)
end



"""
    two_body_feature_from_mapping(cell::Cell, p_mapping, rcut, func=fr)

Construct a vector containing the TwoBodyFeatures
"""
function two_body_feature_from_mapping(cell::Cell, p_mapping, rcut, func=fr)
    indx, us = interger_specie_index(cell)
    features = TwoBodyFeature{typeof(func)}[]
    for (i, map_pair) in enumerate(p_mapping)
        a, b = map_pair[1]
        p = map_pair[2]
        ii = findfirst(x -> x == a, us)
        jj = findfirst(x -> x == b, us)
        #Swap order if ii > jj
        if ii > jj
            ii, jj = jj, ii
        end
        push!(features, TwoBodyFeature(func, p, (ii, jj), Float64(rcut)))
    end

    # Check completeness
    all_ij = [f.sij_idx for f in features]
    for i in 1:length(us)
        for j in i:length(us)
            if !((i, j) in all_ij)
                @warn "Missing interaction between $(us[i]) and $(us[j])"
            end
        end
    end
    features
end

"""
    three_body_feature_from_mapping(cell::Cell, p_mapping, q_mapping, rcut, func=fr)

Construct a vector containing the TwoBodyFeatures
"""
function three_body_feature_from_mapping(cell::Cell, pq_mapping, rcut, func=fr;check=false)
    indx, us = interger_specie_index(cell)
    features = ThreeBodyFeature{typeof(func)}[]
    for (i, map_pair) in enumerate(pq_mapping)
        a, b, c = map_pair[1]
        p, q= map_pair[2]
        ii = findfirst(x -> x == a, us)
        jj = findfirst(x -> x == b, us)
        kk = findfirst(x -> x == c, us)
        #Swap order if ii > jj
        idx = tuple(sort([ii, jj, kk])...)
        push!(features, ThreeBodyFeature(func, p, q, idx, Float64(rcut)))
    end

    if check
        # Check for completeness
        all_ijk = [f.sijk_idx for f in features]
        for i in 1:length(us)
            for j in i:length(us)
                for k in i:length(us)
                    idx = tuple(sort([i, j, k])...)
                    if !(idx in all_ijk)
                        @warn "Missing interaction between $(us[i]), $(us[j]), and $(us[k])"
                    end
                end
            end
        end
    end
    features
end