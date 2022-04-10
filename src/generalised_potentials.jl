#=
Generalised potential
=#
using CellBase
using StaticArrays

abstract type AbstractNBodyFeature end

sortedtuple(iter) = Tuple(sort(collect(iter)))

"""
    permequal(A, i, j)

Check equivalence considering all permutations
"""
function permequal(A, i, j)
    (A[1] == i) && (A[2] ==j) && return true
    (A[2] == i) && (A[1] ==j) && return true
    false
end

"""
    permequal(A, i, j, k)

Check equivalence considering all permutations
"""
function permequal(A, i, j, k)
    (A[1] == i) && (A[2] ==j) && (A[3] == k) && return true
    (A[1] == i) && (A[3] ==j) && (A[2] == k) && return true
    (A[2] == i) && (A[1] ==j) && (A[3] == k) && return true
    (A[2] == i) && (A[3] ==j) && (A[1] == k) && return true
    (A[3] == i) && (A[2] ==j) && (A[1] == k) && return true
    (A[3] == i) && (A[1] ==j) && (A[2] == k) && return true
    false
end


"""
Compunents for constructing the feature vector of two-body interactions
"""
struct TwoBodyFeature{T} <: AbstractNBodyFeature
    "Function of distance"
    f::T
    "Exponents"
    p::Vector{Float64}
    "Specie indices"
    sij_idx::Tuple{Symbol, Symbol}
    "Cut off distance"
    rcut::Float64
    np::Int
end


"Equation 7 in  the DDP paper"
fr(r::T, rcut) where {T} =  r <= rcut ? 2 * (1 - r / rcut) : zero(T)

TwoBodyFeature(f, p::Vector{Float64}, sij_idx::Tuple{Symbol, Symbol}, rcut::Float64) = TwoBodyFeature(f, p, sortedtuple(sij_idx), rcut, length(p))
TwoBodyFeature(p::Vector{Float64}, sij_idx::Tuple{Symbol, Symbol}, rcut::Float64) = TwoBodyFeature(fr, p, sortedtuple(sij_idx), rcut)

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
    permequal(f.sij_idx, si, sj) && f(out, rij)
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
    sijk_idx::Tuple{Symbol, Symbol, Symbol}
    "Cut off distance"
    rcut::Float64
    np::Int
    nq::Int
end

ThreeBodyFeature(f, p::Vector{Float64}, q::Vector{Float64}, sijk_idx::Tuple{Symbol, Symbol, Symbol}, rcut::Float64) = ThreeBodyFeature(f, p, q, sortedtuple(sijk_idx), rcut, length(p), length(q))

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
    permequal(f.sijk_idx, si, sj, sk) && f(out, rij, rik, rjk)
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
    feature_vector(features::Vector{T}, cell::Cell) where T

Compute the feature vector for a give set of two body interactions
"""
function feature_vector(features::Vector{TwoBodyFeature{T}}, cell::Cell;nl=NeighbourList(cell, features[1].rcut)) where T
    # Feature vectors
    fvecs = [zeros(nfeatures(f)) for f in features]
    nat = natoms(cell)
    sym = species(cell)
    for i = 1:nat
        for (j, jextend, rij) in eachneighbour(nl, i)
            # accumulate the feature vector
            for (nf, f) in enumerate(features)
                f(fvecs[nf], rij, sym[i], sym[j])
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
    sym = species(cell)
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
                    f(fvecs[nf], rij, rik, rjk, sym[i], sym[j], sym[k])
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
        push!(features, TwoBodyFeature(func, p, (a, b), Float64(rcut)))
    end

    # Check completeness
    for i in us
        for j in us
            if !(any(x -> permequal(x.sij_idx, i, j), features))
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
        for i in us
            for j in us
                for k in us
                    if !(any(x -> permequal(x.sijk_idx, i, j, k), features))
                        @warn "Missing interaction between $(us[i]), $(us[j]), and $(us[k])"
                    end
                end
            end
        end
    end
    features
end


"""
Collection of Feature specifications and cell
"""
mutable struct CellFeature{T, N}
    elements::Vector{Symbol}
    two_body::Vector{TwoBodyFeature{T}}
    three_body::Vector{ThreeBodyFeature{N}}
end

"""
Construct feature specifications
"""
function CellFeature(elements; p2=2:8, p3=2:8, q3=2:8, rcut2=4.0, rcut3=3.0, f2=fr, f3=fr)
    
    # Sort the elements to ensure stability
    elements = sort(unique(elements))
    # Two body terms
    two_body_features = TwoBodyFeature{typeof(f2)}[]
    existing_comb = []
    for e1 in elements
        for e2 in elements
            if !(any(x -> permequal(x, e1, e2), existing_comb))
                push!(two_body_features, TwoBodyFeature(f2, collect(Float64, p2), (e1, e2), rcut2))
                push!(existing_comb, (e1, e2))
            end
        end
    end

    empty!(existing_comb)
    three_body_features = ThreeBodyFeature{typeof(f3)}[]
    for e1 in elements
        for e2 in elements
            for e3 in elements
                if !(any(x -> permequal(x, e1, e2, e3), existing_comb))
                    push!(three_body_features, ThreeBodyFeature(f3, collect(Float64, p3), collect(Float64, q3), (e1, e2, e3), rcut3))
                    push!(existing_comb, (e1, e2, e3))
                end
            end
        end
    end
    CellFeature(elements, two_body_features, three_body_features)
end

function nfeatures(c::CellFeature)
    n = length(c.elements)
    for f in c.two_body
        n += nfeatures(f)
    end
    for f in c.three_body
        n += nfeatures(f)
    end
    n
end


function feature_vector(cellf::CellFeature, cell::Cell)

    rcut = max(
        maximum(x -> x.rcut, cellf.two_body),
        maximum(x -> x.rcut, cellf.three_body)
    )
    nl = NeighbourList(cell, rcut)

    # One body vector is just a count of different speices
    numbers = atomic_numbers(cell)
    us = unique(numbers)
    sort!(us)
    one_body_vecs = Float64[count(x -> x == num, numbers) for num in us]
    # Concatenated two body vectors 
    two_body_vecs = feature_vector(cellf.two_body, cell;nl=nl)
    # Concatenated three body vectors 
    three_body_vecs = feature_vector(cellf.three_body, cell;nl=nl)
    vcat(one_body_vecs, two_body_vecs, three_body_vecs)
end