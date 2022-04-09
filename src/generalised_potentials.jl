#=
Generalised potential
=#
using CellBase

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
    cutoff::Float64
    np::Int
end


"Equation 7 in  the DDP paper"
fr(r::T, rcut::T) where {T} =  r <= rcut ? 2 * (1 - r / rcut) : zero(T)

TwoBodyFeature(f, p::Vector{Float64}, sij_idx::Tuple{Int, Int}, cutoff::Float64) = TwoBodyFeature(f, p, sij_idx, cutoff, length(p))
TwoBodyFeature(p::Vector{Float64}, sij_idx::Tuple{Int, Int}, cutoff::Float64) = TwoBodyFeature(fr, p, sij_idx, cutoff)

"""
Call the object to accumulate an existing feature vector
"""
function (f::TwoBodyFeature)(out::Vector, rij)
    for i in 1:nfeatures(f)
        out[i] += f.f(rij, f.cutoff) ^ f.p[i]
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
    cutoff::Float64
    np::Int
    nq::Int
end

ThreeBodyFeature(f, p::Vector{Float64}, q::Vector{Float64}, sijk_idx::Tuple{Int, Int, Int}, cutoff::Float64) = ThreeBodyFeature(f, p, q, sijk_idx, cutoff, length(p), length(q))

nfeatures(f::TwoBodyFeature) = f.np * f.nq

"""
    (f::ThreeBodyFeature)(out::Vector, rij, rik, rjk)

Accumulate an existing feature vector
"""
function (f::ThreeBodyFeature)(out::Vector, rij, rik, rjk)
    i = 1
    func = f.f
    cutoff = f.cutoff
    for m in 1:f.np
        for o in 1:f.nq
            out[i] += (func(rij, cutoff) ^ f.p[m]) * (func(rik, cutoff) ^ f.p[m]) * (func(rjk, cutoff) ^ f.q[o])
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
function feature_vector(features::Vector{TwoBodyFeature{T}}, cell::Cell) where T
    # Feature vectors
    fvecs = [zeros(nfeatures(f)) for f in features]
    pos = sposarray(cell)
    nat = natoms(cell)
    shifts = CellBase.shift_vectors(cellmat(lattice(cell)), maximum(f.cutoff for f in features);safe=true)
    spidx, smap = interger_specie_index(cell)
    for i = 1:nat
        for j = 1:nat
            i == j && continue
            for svec in shifts   # Shift vectors for j
                rij = distance_between(pos[i], pos[j], svec)
                # accumulate the feature vector
                for (nf, f) in enumerate(features)
                    f(fvecs[nf], rij, spidx[i], spidx[j])
                end
            end
        end
    end
    vcat(fvecs...)
end

"""
    feature_vector(features::Vector{T}, cell::Cell) where T

Compute the feature vector for a give set of two body interactions
"""
function feature_vector(features::Vector{ThreeBodyFeature{T}}, cell::Cell) where T
    # Feature vectors
    fvecs = [zeros(nfeatures(f)) for f in features]
    pos = sposarray(cell)
    nat = natoms(cell)
    # Note - need to use twice the cut off to ensure distance between j-k is included
    shifts = CellBase.shift_vectors(cellmat(lattice(cell)), maximum(f.cutoff for f in features) * 2;safe=true)
    spidx, smap = interger_specie_index(cell)
    for i = 1:nat
        for j = 1:nat
            i == j && continue
            for svecj in shifts   # Shift vectors for j
                posj = pos[j] .* svecj
                rij = distance_between(pos[i], posj)
                # accumulate the feature vector
                for sveck in shifts   # Shift vectors for k
                    rik = distance_between(pos[i], pos[k], sveck)
                    rjk = distance_between(posj, pos[k], sveck)
                    for (nf, f) in enumerate(features)
                        f(fvecs[nf], rij, rik, rjk, spidx[i], spidx[j], spidx[k])
                    end
                end
            end
        end
    end
    vcat(fvecs...)
end



"""
    two_body_feature_from_mapping(cell::Cell, p_mapping, cutoffs, func=fr)

Construct a vector containing the TwoBodyFeatures
"""
function two_body_feature_from_mapping(cell::Cell, p_mapping, cutoffs, func=fr)
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
        push!(features, TwoBodyFeature(func, p, (ii, jj), Float64(cutoffs[i])))
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
    three_body_feature_from_mapping(cell::Cell, p_mapping, q_mapping, cutoffs, func=fr)

Construct a vector containing the TwoBodyFeatures
"""
function three_body_feature_from_mapping(cell::Cell, pq_mapping, cutoffs, func=fr;check=false)
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
        push!(features, ThreeBodyFeature(func, p, q, idx, Float64(cutoffs[i])))
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