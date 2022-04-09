#=
Generalised potential
=#
using CellBase

"""
Compunents for constructing the feature vector
"""
struct TwoBodyFeature{T}
    "Function of distance"
    f::T
    "Exponents"
    p::Vector{Float64}
    "Specie indices"
    sij_idx::Tuple{Int, Int}
    "Cut off distance"
    cutoff::Float64
    nf::Int
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

nfeatures(f::TwoBodyFeature) = f.nf

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

Compute the feature vector for a give set of body interactions
"""
function feature_vector(features::Vector{T}, cell::Cell) where T
    # Feature vectors
    fvecs = [zeros(nfeatures(f)) for f in features]
    pos = sposarray(cell)
    nat = natoms(cell)
    shifts = CellBase.shift_vectors(cellmat(lattice(cell)), maximum(f.cutoff for f in features);safe=true)
    spidx, smap = interger_specie_index(cell)
    for i = 1:nat
        for j = i+1:nat
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