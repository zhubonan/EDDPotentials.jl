#=
Look up array for features
=#

"""
Lookup table for feature locations
"""
struct FeatureLookup
    two_body::Array{Int, 2}
    two_body_end::Array{Int, 2}
    three_body::Array{Int, 3}
    three_body_end::Array{Int, 3}
    ne::Int
    elements::Vector{Symbol}
end

function FeatureLookup(cf::CellFeature)
    elements = cf.elements
    ne = length(elements)
    two_body = zeros(Int, ne, ne)
    two_body_end = zeros(Int, ne, ne)
    three_body = zeros(Int, ne, ne, ne)
    three_body_end = zeros(Int, ne, ne, ne)
    # Skip the composition (1-body part)
    pos = ne + 1
    # Check for uniqueness 
    @assert length(unique([feat.sij_idx for feat in cf.two_body])) == length(cf.two_body)
    @assert length(unique([feat.sijk_idx for feat in cf.three_body])) == length(cf.three_body)

    for feat in cf.two_body
        i = findfirst(x -> x== feat.sij_idx[1], elements)
        j = findfirst(x -> x== feat.sij_idx[2], elements)
        two_body[i, j] = pos
        pos += feat.np
        two_body_end[i, j] = pos  - 1
    end 
    for feat in cf.three_body
        i = findfirst(x -> x== feat.sijk_idx[1], elements)
        j = findfirst(x -> x== feat.sijk_idx[2], elements)
        k = findfirst(x -> x== feat.sijk_idx[3], elements)
        three_body[i, j, k] = pos
        # Permutation equivalence
        three_body[i, k, j] = pos # This only works when i,k,j not equivalent to k,j,i
        pos += feat.np * feat.nq
        three_body_end[i, j, k] = pos - 1
        three_body_end[i, k, j] = pos - 1  # This only works when i,k,j not equivalent to k,j,i
    end
    @assert maximum(three_body_end) == nfeatures(cf)
    # Element descriptor offset
    FeatureLookup(
        two_body,
        two_body_end,
        three_body,
        three_body_end,
        ne,
        elements,
    )
end

function lookup(fl::FeatureLookup, elem1)
    findfirst( x -> x== elem1, fl.elements)
end

"""
    lookup(fl::FeatureLookup, elem1, elem2)

Lookup positions for a two body feature.
"""
function lookup(fl::FeatureLookup, elem1, elem2)
    i = findfirst( x -> x== elem1, fl.elements)
    j = findfirst( x -> x== elem2, fl.elements)
    return (fl.two_body[i, j], fl.two_body_end[i, j])
end

"""
    lookup(fl::FeatureLookup, elem1, elem2, elem3)

Lookup positions for a three body feature.
"""
function lookup(fl::FeatureLookup, elem1, elem2, elem3)
    i = findfirst( x -> x== elem1, fl.elements)
    j = findfirst( x -> x== elem2, fl.elements)
    k = findfirst( x -> x== elem3, fl.elements)
    return (fl.three_body[i, j, k], fl.three_body_end[i, j, k])
end

"""
    lookup(cf::CellFeature, args...)

Lookup position for features
"""
lookup(cf::CellFeature, i::Symbol) = lookup(FeatureLookup(cf), i)
lookup(cf::CellFeature, i::Symbol, j::Symbol) = lookup(FeatureLookup(cf), i, j)
lookup(cf::CellFeature, i::Symbol, j::Symbol, k::Symbol) = lookup(FeatureLookup(cf), i, j, k)

function lookup(cf::CellFeature, feat::TwoBodyFeature)
    pos = length(cf.elements) + 1
    for _feat in cf.two_body
        _feat == feat && return pos, pos + feat.np - 1
        pos += feat.np
    end
    throw(ErrorException("$(feat) is not found in the list of features"))
    return -1, -1
end

function lookup(cf::CellFeature, feat::ThreeBodyFeature)
    pos = length(cf.elements) + 1 + sum(_feat.np for _feat in cf.two_body)
    for _feat in cf.three_body
        _feat == feat && return pos, pos + feat.np * feat.nq - 1
        pos += feat.np * feat.nq
    end
    return -1, -1
end