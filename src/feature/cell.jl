
"""
    CellFeature

Collection of Feature specifications and cell.
"""
mutable struct CellFeature{T,G,V}
    elements::Vector{Symbol}
    two_body::T
    three_body::G
    constructor_kwargs::V
end

function CellFeature(elements, two_body, three_body)
    CellFeature(elements, two_body, three_body, nothing)
end

"""
    +(a::CellFeature, b::CellFeature)

Combine two `CellFeature` objects together. The features are simply concatenated in this case.
"""
function Base.:+(a::CellFeature, b::CellFeature)
    elements = sort(unique(vcat(a.elements, b.elements)))
    two_body = (a.two_body..., b.two_body...)
    three_body = (a.three_body..., b.three_body...)
    CellFeature(elements, two_body, three_body)
end

"""
    ==(A::T, B::T) where {T<:Union{AbstractNBodyFeature, CellFeature}}

Check equality between two `CellFeature` / `AbstractNBodyFeature` objects.
"""
function Base.:(==)(A::T, B::T) where {T<:Union{AbstractNBodyFeature,CellFeature}}
    for name in fieldnames(T)
        # Compare array equivalence
        if getproperty(A, name) != getproperty(B, name)
            return false
        end
    end
    return true
end

Base.show(io::IO, x::CellFeature) = Base.show(io, MIME("text/plain"), x)


"""
    CellFeature(elements;kwargs...)

Construct a `CellFeature` instance.

# Args
- `elements`: A vector of the elements to be included in the features.
- `p2`: A sequence of the two-body polynomial powers (``p``).
- `p3`: A sequence of the three-body polynomial powers (``p``).
- `q3`: A sequence of the three-body polynomial powers (``q``).
- `rcut2`: Cut off distance for two-body features.
- `rcut3`: Cut off distance for three-body features.
- `f2`: Distance function for two-body interactions.
- `f3`: Distance function for three-body interactions.
- `g2`: Gradient function for two-body interactions.
- `g3`: Gradient function for three-body interactions.
- `geometry_sequence`: Wether to covnert the `p2`, `p3`, `q3` as geometry sequence.

"""
function CellFeature(
    elements;
    p2=2:8,
    p3=p2,
    q3=p3,
    rcut2=4.0,
    rcut3=rcut2,
    f2=fr,
    g2=gfr,
    f3=fr,
    g3=gfr,
    geometry_sequence=false,
)
    cf_kwargs = (; p2, p3, q3, rcut2, rcut3, geometry_sequence)

    # Apply geomtry sequence for the powers
    if geometry_sequence
        p2 = genp(p2)
        p3 = genp(p3)
        q3 = genp(q3)
    end

    # Sort the elements to ensure stability
    elements = sort(unique(map(Symbol, elements)))
    # Two body terms
    two_body_features = []
    existing_comb = []
    for e1 in elements
        for e2 in elements
            if !(any(x -> permequal(x, e1, e2), existing_comb))
                push!(
                    two_body_features,
                    TwoBodyFeature(f2, g2, Tuple(collect(p2)), (e1, e2), rcut2),
                )
                push!(existing_comb, (e1, e2))
            end
        end
    end

    empty!(existing_comb)
    three_body_features = []
    for e1 in elements
        for e2 in elements
            for e3 in elements
                if !(any(x -> permequal(x, e1, e2, e3), existing_comb))
                    push!(
                        three_body_features,
                        ThreeBodyFeature(
                            f3,
                            g3,
                            Tuple(collect(p3)),
                            Tuple(collect(q3)),
                            (e1, e2, e3),
                            rcut3,
                        ),
                    )
                    push!(existing_comb, (e1, e2, e3))
                end
            end
        end
    end
    CellFeature(elements, Tuple(two_body_features), Tuple(three_body_features), cf_kwargs)
end

"""
    feature_names(cf::CellFeature)

Return the name for the features.
"""
function feature_names(cf::CellFeature; kwargs...)
    feature_names(cf.two_body..., cf.three_body...; kwargs...)
end

function Base.show(io::IO, z::MIME"text/plain", cf::CellFeature)
    println(io, "$(typeof(cf))")
    println(io, "  Elements:")
    println(io, "    $(cf.elements)")
    println(io, "  TwoBodyFeatures:")
    for tb in cf.two_body
        println(io, "    $(tb)")
    end
    println(io, "  ThreeBodyFeatures:")
    for tb in cf.three_body
        println(io, "    $(tb)")
    end
end

"""
    features(c::CellFeature)

Return the total number of features elements in a `CellFeature` object.
"""
function nfeatures(c::CellFeature)
    length(c.elements) + sum(nfeatures, c.two_body) + sum(nfeatures, c.three_body)
end

"""
    nbodyfeatures(c::CellFeature, nbody)

Return the number of N-body features
"""
function nbodyfeatures(c::CellFeature, nbody)
    if nbody == 1
        return length(c.elements)
    elseif nbody == 2
        return sum(nfeatures, c.two_body)
    elseif nbody == 3
        return sum(nfeatures, c.three_body)
    end
    return 0
end

"""
    feature_vector(cf::CellFeature, cell::Cell; nmax=500, skin=1.0)

Return a matrix of vectors describing the environment of each atom.
"""
function feature_vector(cf::CellFeature, cell::Cell; nmax=500, skin=1.0)
    # Infer rmax
    rcut = suggest_rcut(cf; shell=skin)
    nl = NeighbourList(cell, rcut, nmax; savevec=true)
    vecs = zeros(sum(feature_size(cf)), length(cell))
    n1 = feature_size(cf)[1]
    wk = get_workspace(cf.two_body, cf.three_body, nl, false; fvec=vecs)
    compute_fv!(wk, cf.two_body, cf.three_body, cell; nl, offset=n1)
    one_body_vectors!(vecs, cell, cf)
    vecs
end

"""
    one_body_vectors(cell::Cell, cf::CellFeature)

Construct one-body features for the structure.
The one-body feature is essentially an one-hot encoding of the specie labels 
"""
function one_body_vectors(cell::Cell, cf::CellFeature; offset=0)
    vecs = zeros(length(cf.elements), length(cell))
    one_body_vectors!(vecs, cell, cf)
end

"""
    one_body_vectors!(v, cell::Cell, cf::CellFeature)

Construct one-body features for the structure.
The one-body feature is essentially an one-hot encoding of the specie labels 
"""
function one_body_vectors!(v::AbstractMatrix, cell::Cell, cf::CellFeature; offset=0)
    symbols = species(cell)
    for (iat, s) in enumerate(symbols)
        for (ispec, sZ) in enumerate(cf.elements)
            if s == sZ
                v[ispec+offset, iat] = 1.0
            end
        end
    end
    v
end

"""
    feature_size(cf::CellFeature)

Return size of the feature vector for each body-order of a `CellFeature` object.
"""
function feature_size(cf::CellFeature)
    (length(cf.elements), sum(nfeatures, cf.two_body), sum(nfeatures, cf.three_body))
end

"""
    suggest_rcut(cf::CellFeature; shell=1.0)

Get a suggested cut off radius for NN list for a CellFeature.
"""
function suggest_rcut(cf::CellFeature; shell=1.0)
    r3 = maximum(x.rcut for x in cf.two_body)
    r2 = maximum(x.rcut for x in cf.three_body)
    max(r3, r2) + shell
end

"""
    suggest_rcut(features...; shell=1.0)

Get a suggested rcut for a collection of features.
"""
function suggest_rcut(features...; shell=1.0)
    maximum(x.rcut for x in features) + shell
end
