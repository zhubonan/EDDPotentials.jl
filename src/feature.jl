#=
Generalised potential
=#
import Base
using CellBase
using StaticArrays
using Parameters

abstract type AbstractNBodyFeature end

"""
Faster version of (^) by expanding more integer power into multiplications
"""
@inline function fast_pow(x, y::Int)
    y == -1 && return inv(x)
    y == 0 && return one(x)
    y == 1 && return x
    y == 2 && return x * x
    y == 3 && return x * x * x
    y == 4 && return x * x * x * x
    y == 5 && return x * x * x * x * x
    y == 6 && return x * x * x * x * x * x
    y == 7 && return x * x * x * x * x * x * x
    y == 8 && return x * x * x * x * x * x * x * x
    y == 9 && return x * x * x * x * x * x * x * x * x
    y == 10 && return x * x * x * x * x * x * x * x * x * x
    y == 11 && return x * x * x * x * x * x * x * x * x * x * x
    y == 12 && return x * x * x * x * x * x * x * x * x * x * x * x
    ^(x, y)
end

@inline fast_pow(x, y) = x^y

sortedtuple(iter) = Tuple(sort(collect(iter)))

"""
Generate p/q as a geometry series
"""
function genp(pmin, pmax, np)
    β = (pmax / pmin)^(1 / (np - 1))
    collect(round(pmin * β^(i - 1), digits=3) for i = 1:np)
end

"""
Generate p/q as geometry series
"""
function genp(p::Union{UnitRange,StepRange})
    genp(p[1], p[end], length(p))
end

function genp(p::Vector{T}) where {T}
    if isempty(p)
        return T[]
    end
    genp(p...)
end


"""
    permequal(A, i, j)

Check equivalence considering all permutations with the first element matched. 
"""
function permequal(A, i, j)
    (A[1] == i) && (A[2] == j) && return true
    false
end

"""
    permequal(A, i, j, k)

Check equivalence considering all permutations with the first element matched.
"""
function permequal(A, i, j, k)
    (A[1] == i) && (A[2] == j) && (A[3] == k) && return true
    (A[1] == i) && (A[3] == j) && (A[2] == k) && return true
    false
end


"""
TwoBodyFeature{T, M} <: AbstractNBodyFeature

Type for constructing the feature vector of the two-body interactions.
"""
struct TwoBodyFeature{T,M,P} <: AbstractNBodyFeature
    "Function of distance"
    f::T
    "df(r)/r"
    g::M
    "Exponents"
    p::P
    "Specie indices"
    sij_idx::Tuple{Symbol,Symbol}
    "Cut off distance"
    rcut::Float64
    np::Int
end


function Base.show(io::IO, ::MIME"text/plain", x::TwoBodyFeature)
    println(io, "$(typeof(x))")
    println(io, "  f: $(x.f)")
    println(io, "  g: $(x.g)")
    println(io, "  p: $(x.p)")
    println(io, "  specie: $(x.sij_idx[1])-$(x.sij_idx[2])")
    println(io, "  rcut: $(x.rcut)")
end


@doc raw"""
    fr(r, rcut)

Equation 7 in Pickard 2022 describing interactions with well-behaved cut offs:

```math
f(r)= \begin{cases} 
    2(1 - r / r_{rc}) & r \leq r_{rc} \\ 
    0 & r > r_{rc} 
    \end{cases}
```

"""
fr(r::T, rcut) where {T} = r <= rcut ? 2 * (1 - r / rcut) : zero(T)


@doc raw"""
    gfr(r, rcut)

Gradient of the Equation 7 in Pickard 2022 describing interactions with well-behaved cut offs:

```math
g(r)= \begin{cases} 
    -2 / r_{rc} & r \leq r_{rc} \\ 
    0 & r > r_{rc} 
    \end{cases}
```

"""
gfr(r::T, rcut) where {T} = r <= rcut ? -2 / rcut : zero(T)

TwoBodyFeature(f, g, p, sij_idx, rcut::Real) =
    TwoBodyFeature(f, g, collect(p), tuple(sij_idx[1], sij_idx[2]), rcut, length(p))
TwoBodyFeature(p, sij_idx, rcut::Real) = TwoBodyFeature(fr, gfr, p, sij_idx, rcut)

"""
    (f::TwoBodyFeature)(out::AbstractMatrix, rij, iat, istart=1)

Accumulate an existing matrix of the feature vectors

Args:
- out: Output matrix
- rji: distance between two atoms
- iat: starting index of the vector to be updated
- istart: starting index of the vector to be updated
"""
function (f::TwoBodyFeature)(out::AbstractMatrix, rij, iat, istart=1)
    val = f.f(rij, f.rcut)
    i = istart
    for j = 1:nfeatures(f)
        out[i, iat] += fast_pow(val, f.p[j])
        i += 1
    end
    out
end

"""
    withgradient!(e::Matrix, g::Vector, f::TwoBodyFeature, rij, iat, istart)

Calculate d(f(r)^p) / dr for each feature as well as the feature vector vector.

Args:
* e: The matrix storing the feature vectors of shape (nfe, nat)
* g: The matrix storing the gradient feature vectors of shape (nfe,), **for this particular pair of i and j**.
"""
function withgradient!(e, g, f::TwoBodyFeature, rij, iat, istart)
    val = f.f(rij, f.rcut)
    gval = f.g(rij, f.rcut)
    i = istart
    for j = 1:nfeatures(f)
        g[i] += f.p[j] * fast_pow(val, (f.p[j] - 1)) * gval  # Chain rule
        e[i, iat] += fast_pow(val, f.p[j])
        i += 1
    end
    e, g
end

function withgradient!(e, g, f::TwoBodyFeature, rij, si, sj, iat, istart=1)
    permequal(f.sij_idx, si, sj) && withgradient!(e, g, f, rij, iat, istart)
    e, g
end

function withgradient(f::TwoBodyFeature, rij)
    e = zeros(nfeatures(f), 1)
    g = zeros(nfeatures(f))
    withgradient!(e, g, f, rij, 1, 1)
end


function (f::TwoBodyFeature)(out::AbstractMatrix, rij, si, sj, iat, istart=1)
    permequal(f.sij_idx, si, sj) && f(out, rij, iat, istart)
    out
end


(f::TwoBodyFeature)(rij) = f(zeros(nfeatures(f), 1), rij, 1, 1)
nfeatures(f::TwoBodyFeature) = f.np


"""
    ThreeBodyFeature{T, M} <: AbstractNBodyFeature

Type for constructing the feature vector of the three-body interactions.
"""
struct ThreeBodyFeature{T,M,P,Q} <: AbstractNBodyFeature
    "Basis function"
    f::T
    "df(r)/r"
    g::M
    "Exponents for p"
    p::P
    "Exponents for q"
    q::Q
    "Specie indices"
    sijk_idx::Tuple{Symbol,Symbol,Symbol}
    "Cut off distance"
    rcut::Float64
    np::Int
    nq::Int
end

ThreeBodyFeature(f, g, p, q, sijk_idx, rcut::Float64) = ThreeBodyFeature(
    f,
    g,
    collect(p),
    collect(q),
    tuple(sijk_idx[1], sijk_idx[2], sijk_idx[3]),
    rcut,
    length(p),
    length(q),
)
ThreeBodyFeature(p, q, sijk_idx, rcut::Float64) =
    ThreeBodyFeature(fr, gfr, p, q, sijk_idx, rcut)


function Base.show(io::IO, ::MIME"text/plain", x::ThreeBodyFeature)
    println(io, "$(typeof(x))")
    println(io, "  f: $(x.f)")
    println(io, "  g: $(x.g)")
    println(io, "  p: $(x.p)")
    println(io, "  q: $(x.q)")
    println(io, "  specie: $(x.sijk_idx[1])-$(x.sijk_idx[2])-$(x.sijk_idx[3])")
    println(io, "  rcut: $(x.rcut)")
end

nfeatures(f::ThreeBodyFeature) = f.np * f.nq

"""
    (f::ThreeBodyFeature)(out::Vector, rij, rik, rjk)

Accumulate an existing feature vector
"""
function (f::ThreeBodyFeature)(out::AbstractMatrix, rij, rik, rjk, iat, istart=1)
    func = f.f
    rcut = f.rcut
    fij = func(rij, rcut)
    fik = func(rik, rcut)
    fjk = func(rjk, rcut)
    i = istart
    for m = 1:f.np
        ijkp = fast_pow(fij, f.p[m]) * fast_pow(fik, f.p[m])
        @inbounds for o = 1:f.nq  # Note that q is summed in the inner loop
            #out[i, iat] += (fij ^ f.p[m]) * (fik ^ f.p[m]) * (fjk ^ f.q[o])
            out[i, iat] += ijkp * fast_pow(fjk, f.q[o])
            i += 1
        end
    end
    out
end

"(f::ThreeBodyFeature)(out::Vector, rij, rik, rjk, si, sj, sk)"
function (f::ThreeBodyFeature)(
    out::AbstractMatrix,
    rij,
    rik,
    rjk,
    si,
    sj,
    sk,
    iat,
    istart=1,
)
    permequal(f.sijk_idx, si, sj, sk) && f(out, rij, rik, rjk, iat, istart)
    out
end

"(f::ThreeBodyFeature)(rij) = f(zeros(nfeatures(f)), rij, rik, rjk)"
(f::ThreeBodyFeature)(rij, rik, rjk) = f(zeros(nfeatures(f), 1), rij, rik, rjk, 1, 1)

function withgradient!(e, g, f::ThreeBodyFeature, rij, rik, rjk, si, sj, sk, iat, istart=1)
    permequal(f.sijk_idx, si, sj, sk) && withgradient!(e, g, f, rij, rik, rjk, iat, istart)
    e, g
end

function withgradient(f::ThreeBodyFeature, rij, rik, rjk)
    e = zeros(nfeatures(f), 1)
    g = zeros(3, nfeatures(f))
    withgradient!(e, g, f, rij, rik, rjk, 1, 1)
end

"""
Calculate df / drij, df /drik, df/drjk for each element of a ThreeBodyFeature

Args:
* e: The matrix storing the feature vectors of shape (nfe, nat)
* g: The matrix storing the gradient feature vectors of shape (3, nfe), against rij, rik and rjk
  **for this particular combination of i, j, k**.
"""
function withgradient!(
    e::Matrix,
    g::Matrix,
    f::ThreeBodyFeature,
    rij,
    rik,
    rjk,
    iat,
    istart,
)
    func = f.f
    rcut = f.rcut
    fij = func(rij, rcut)
    fik = func(rik, rcut)
    fjk = func(rjk, rcut)
    gij = f.g(rij, rcut)
    gik = f.g(rik, rcut)
    gjk = f.g(rjk, rcut)
    i = istart  # Index of the element
    for m = 1:f.np
        # Cache computed value
        ijkp = fast_pow(fij, f.p[m]) * fast_pow(fik, f.p[m])
        # Avoid NaN if fik is 0
        if fik != 0
            tmp = ijkp / fik  # fij ^ pm * fik * (pm - 1)
        else
            tmp = zero(ijkp)
        end
        for o = 1:f.nq  # Note that q is summed in the inner loop
            # Feature term
            e[i, iat] += ijkp * fast_pow(fjk, f.q[o])
            # Gradient - NOTE this can be optimised further...
            g[1, i] +=
                f.p[m] *
                fast_pow(fij, (f.p[m] - 1)) *
                fast_pow(fik, f.p[m]) *
                fast_pow(fjk, f.q[o]) *
                gij
            g[2, i] += tmp * f.p[m] * fast_pow(fjk, f.q[o]) * gik
            g[3, i] += ijkp * f.q[o] * fast_pow(fjk, (f.q[o] - 1)) * gjk
            i += 1
        end
    end
    e, g
end


"""
    feature_vector(features::Vector{T}, cell::Cell) where T

Compute the feature vector for a give set of two body interactions
"""
function feature_vector2!(
    fvecs,
    features::Tuple,
    cell::Cell;
    nl=NeighbourList(cell, features[1].rcut),
    offset=0,
)
    # Feature vectors
    nfe = map(nfeatures, features)
    nat = natoms(cell)
    sym = species(cell)
    # Set the feature vectors elements to be updated to zero
    fvecs[1+offset:sum(nfe)+offset, :] .= 0
    rcut = maximum(x -> x.rcut, features)
    for iat = 1:nat
        for (jat, jextend, rij) in eachneighbour(nl, iat)
            rij > rcut && continue
            # Accumulate feature vectors
            ist = 1 + offset
            for (ife, f) in enumerate(features)
                f(fvecs, rij, sym[iat], sym[jat], iat, ist)
                ist += nfe[ife]
            end
        end
    end
    fvecs
end

function feature_vector2(
    features::Tuple,
    cell::Cell;
    nmax=500,
    nl=NeighbourList(cell, maximum(f.rcut for f in features) + 1.0, nmax),
)
    # Feature vectors
    nfe = map(nfeatures, features)
    ni = nions(cell)
    fvecs = zeros(sum(nfe), ni)
    feature_vector2!(fvecs, features, cell; nl)
end

function feature_vector3(
    features::Tuple,
    cell::Cell;
    nmax=500,
    nl=NeighbourList(cell, maximum(f.rcut for f in features) + 1.0, nmax),
)
    # Feature vectors
    nfe = map(nfeatures, features)
    ni = nions(cell)
    fvecs = zeros(sum(nfe), ni)
    feature_vector3!(fvecs, features, cell; nl)
end


"""
    feature_vector(features::Vector{T}, cell::Cell) where T

Compute the feature vector for each atom a give set of three body interactions
"""
function feature_vector3!(
    fvecs,
    features::Tuple,
    cell::Cell;
    nl=NeighbourList(cell, features[1].rcut),
    offset=0,
)
    nat = natoms(cell)
    nfe = map(nfeatures, features)
    # Note - need to use twice the cut off to ensure distance between j-k is included
    sym = species(cell)
    # Set the feature vectors to be zero
    fvecs[1+offset:sum(nfe)+offset, :] .= 0
    rcut = maximum(x -> x.rcut, features)
    for iat = 1:nat
        for (jat, jextend, rij) in eachneighbour(nl, iat)
            rij > rcut && continue
            for (kat, kextend, rik) in eachneighbour(nl, iat)
                rik > rcut && continue
                # Avoid double counting i j k is the same as i k j
                if kextend <= jextend
                    continue
                end
                # Compute the distance between extended j and k
                rjk = distance_between(nl.ea.positions[jextend], nl.ea.positions[kextend])
                rjk > rcut && continue
                # accumulate the feature vector
                ist = 1 + offset
                for (ife, f) in enumerate(features)
                    f(fvecs, rij, rik, rjk, sym[iat], sym[jat], sym[kat], iat, ist)
                    ist += nfe[ife]
                end
            end
        end
    end
    fvecs
end

# """
#     feature_vector_new!(features::Vector{T}, cell::Cell) where T

# Compute the feature vector for each atom a give set of three body interactions.
# Improved version (only tested with single features)
# """
# function feature_vector_new!(fvecs, features::Vector{T}, cell::Cell;nl=NeighbourList(cell, features[1].rcut), offset=0) where {T<:ThreeBodyFeature}
#     nat = natoms(cell)
#     nfe = map(nfeatures, features) 
#     # Note - need to use twice the cut off to ensure distance between j-k is included
#     sym = species(cell)
#     # Set the feature vectors to be zero
#     fvecs[1+offset:sum(nfe) + offset, :] .= 0
#     rcut = maximum(x -> x.rcut, features)

#     # All values of P
#     npmax = maximum(length(x.p) for x in features)
#     # All values of q
#     nqmax = maximum(length(x.q) for x in features)

#     # Storage space for each p and q power
#     pij = zeros(npmax, length(features))
#     pik = zeros(npmax, length(features))
#     qjk = zeros(nqmax, length(features))
#     for iat = 1:nat
#         for (jat, jextend, rij) in eachneighbour(nl, iat)
#             rij > rcut && continue

#             for (i, feat) in enumerate(features)
#                 ftmp = feat.f(rij, feat.rcut)
#                 @inbounds for j in 1:length(feat.p)
#                     pij[j, i] = fast_pow(ftmp, feat.p[j])
#                 end
#             end

#             for (kat, kextend, rik) in eachneighbour(nl, iat)
#                 rik > rcut && continue

#                 # Avoid double counting i j k is the same as i k j
#                 if kextend <= jextend 
#                     continue
#                 end
#                 # Compute the distance between extended j and k
#                 rjk = distance_between(nl.ea.positions[jextend], nl.ea.positions[kextend])
#                 rjk > rcut && continue

#                 for (i, feat) in enumerate(features)
#                     ftmp = feat.f(rik, feat.rcut)
#                     @inbounds for j in 1:length(feat.p)
#                         pik[j, i] = fast_pow(ftmp, feat.p[j])
#                     end
#                 end

#                 for (i, feat) in enumerate(features)
#                     ftmp = feat.f(rjk, feat.rcut)
#                     # for j in axes(f.p)
#                     #     pjk[j, i] = fast_pow(ftmp, f.p[j])
#                     # end
#                     @inbounds for j in 1:length(feat.q)
#                         qjk[j, i] = fast_pow(ftmp, feat.q[j])
#                     end
#                 end

#                 # accumulate the feature vector
#                 ist = offset + 1
#                 for (ife, feat) in enumerate(features)
#                     if permequal(feat.sijk_idx, sym[iat], sym[jat], sym[kat])
#                         # Construct feature vector using p, q powers
#                         for m in 1:feat.np
#                             ijkp = pij[m, ife] * pik[m, ife]
#                             for o in 1:feat.nq
#                                 fvecs[ist, iat] += ijkp * qjk[o, ife]
#                                 ist += 1
#                             end
#                         end
#                     end
#                 end
#             end
#         end
#     end
#     fvecs
# end

"""
    two_body_feature_from_mapping(cell::Cell, p_mapping, rcut, func=fr)

Construct a vector containing the TwoBodyFeatures
"""
function two_body_feature_from_mapping(cell::Cell, p_mapping, rcut, func=fr, gfunc=gfr)
    us = unique(species(cell))
    features = TwoBodyFeature{typeof(func),typeof(gfunc)}[]
    for map_pair in p_mapping
        a, b = map_pair[1]
        p = map_pair[2]
        push!(features, TwoBodyFeature(func, gfunc, p, (a, b), Float64(rcut)))
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
function three_body_feature_from_mapping(
    cell::Cell,
    pq_mapping,
    rcut,
    func=fr,
    gfunc=gfr;
    check=false,
)
    us = unique(species(cell))
    features = ThreeBodyFeature{typeof(func),typeof(gfunc)}[]
    for map_pair in pq_mapping
        a, b, c = map_pair[1]
        p, q = map_pair[2]
        ii = findfirst(x -> x == a, us)
        jj = findfirst(x -> x == b, us)
        kk = findfirst(x -> x == c, us)
        #Swap order if ii > jj
        idx = tuple(sort([ii, jj, kk])...)
        push!(features, ThreeBodyFeature(func, gfunc, p, q, idx, Float64(rcut)))
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
                    TwoBodyFeature(f2, g2, collect(p2), (e1, e2), rcut2),
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
                            collect(p3),
                            collect(q3),
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
    feature_names(features...)

Return the name for the features.
"""
function feature_names(features...; show_value=true)
    names = String[]
    for feat in features
        if isa(feat, TwoBodyFeature)
            ftype = "2"
            pairs = join(string.(feat.sij_idx), "-")
            for (m, p) in enumerate(feat.p)
                if show_value
                    push!(names, "$(ftype)-$(pairs)-$p")
                else
                    push!(names, "$(ftype)-$(pairs)-$m")
                end

            end
        elseif isa(feat, ThreeBodyFeature)
            ftype = "3"
            pairs = join(string.(feat.sijk_idx), "-")
            for (m, p) in enumerate(feat.p)
                for (o, q) in enumerate(feat.q)
                    if show_value
                        push!(names, "$(ftype)-$(pairs)-$p-$q")
                    else
                        push!(names, "$(ftype)-$(pairs)-$m-$o")
                    end
                end
            end
        end
    end
    return names
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
    rcut = suggest_rcut(cf; offset=skin)
    nl = NeighbourList(cell, rcut, nmax)
    vecs = zeros(sum(feature_size(cf)), length(cell))
    n1 = feature_size(cf)[1]
    compute_fv!(vecs, cf.two_body, cf.three_body, cell; nl, offset=n1)
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
    suggest_rcut(cf::CellFeature; offset=1.0)

Get a suggested cut off radius for NN list for a CellFeature.
"""
function suggest_rcut(cf::CellFeature; offset=1.0)
    r3 = maximum(x.rcut for x in cf.two_body)
    r2 = maximum(x.rcut for x in cf.three_body)
    max(r3, r2) + offset
end

"""
    suggest_rcut(features...; offset=1.0)

Get a suggested rcut for a collection of features.
"""
function suggest_rcut(features...; offset=1.0)
    maximum(x.rcut for x in features) + offset
end


"""
    feature_vector!(fvecs, gvecs, features2, features3, cell;kwargs...)

Args:

- `nl`: If passed, using it as an existing `NeighbourList`.

Compute the feature vector for a given set of two and three body interactions, 
This is an optimised version for feature generation, but does not compute the gradients.

Returns the feature vector and the core repulsion energy if any.
"""
function feature_vector!(
    fvec,
    features2::Tuple,
    features3::Tuple,
    cell::Cell;
    nl=NeighbourList(
        cell,
        maximum(x.rcut for x in (features2..., features3...));
        savevec=true,
    ),
    offset=0,
    core=nothing,
)

    nfe3 = map(nfeatures, features3)
    lfe3 = length(features3)

    nfe2 = map(nfeatures, features2)
    totalfe2 = sum(nfe2)

    nat = natoms(cell)
    sym = species(cell)

    # Caches for three-body computation
    # The goal is to avoid ^ operator as much as possible
    npmax3 = maximum(length(x.p) for x in features3)
    # All values of q
    nqmax3 = maximum(length(x.q) for x in features3)

    pij = zeros(npmax3, lfe3)
    pij_1 = zeros(npmax3, lfe3)
    inv_fij = zeros(lfe3)

    pik = zeros(npmax3, lfe3)
    pik_1 = zeros(npmax3, lfe3)
    inv_fik = zeros(lfe3)

    qjk = zeros(nqmax3, lfe3)
    qjk_1 = zeros(nqmax3, lfe3)
    inv_fjk = zeros(lfe3)

    # Reset all gradients to zero
    fill!(fvec, 0)

    maxrcut = maximum(x -> x.rcut, (features3..., features2...))
    ecore = 0.0
    for iat = 1:nat
        for (jat, jextend, rij) in CellBase.eachneighbour(nl, iat)
            rij > maxrcut && continue

            if !isnothing(core)
                ecore += core.f(rij, core.rcut) * core.a
            end

            # Compute pij
            for (i, feat) in enumerate(features3)
                ftmp = feat.f(rij, feat.rcut)
                inv_fij[i] = 1.0 / ftmp
                @inbounds for j = 1:length(feat.p)
                    pij_1[j, i] = fast_pow(ftmp, feat.p[j] - 1)
                    pij[j, i] = pij_1[j, i] * ftmp
                end
            end

            # Update two body features
            i = 1 + offset
            for (ife, f) in enumerate(features2)
                if permequal(f.sij_idx, sym[iat], sym[jat])
                    fij = f.f(rij, f.rcut)
                    for m = 1:f.np
                        fvec[i, iat] += fast_pow(fij, f.p[m])
                        i += 1
                    end
                else
                    i += f.np
                end
            end

            # Skip three body update if no three-body feature is passed
            if length(features3) == 0
                continue
            end

            #### Update three body features and forces

            for (kat, kextend, rik) in CellBase.eachneighbour(nl, iat)
                rik > maxrcut && continue

                # Avoid double counting i j k is the same as i k j
                if kextend <= jextend
                    continue
                end

                # Compute the distance between extended j and k
                vjk = nl.ea.positions[kextend] .- nl.ea.positions[jextend]
                rjk = norm(vjk)
                rjk > maxrcut && continue

                # This is a valid pair - compute the distances

                # Compute pik
                for (i, feat) in enumerate(features3)
                    ftmp = feat.f(rik, feat.rcut)
                    inv_fik[i] = 1.0 / ftmp
                    @inbounds for j = 1:length(feat.p)
                        pik_1[j, i] = fast_pow(ftmp, feat.p[j] - 1)
                        pik[j, i] = pik_1[j, i] * ftmp
                    end
                end

                # Compute pjk
                for (i, feat) in enumerate(features3)
                    ftmp = feat.f(rjk, feat.rcut)
                    inv_fjk[i] = 1.0 / ftmp
                    @inbounds for j = 1:length(feat.q)
                        qjk_1[j, i] = fast_pow(ftmp, feat.q[j] - 1)
                        qjk[j, i] = qjk_1[j, i] * ftmp
                    end
                end

                i = totalfe2 + 1 + offset
                for (ife, f) in enumerate(features3)
                    if !permequal(f.sijk_idx, sym[iat], sym[jat], sym[kat])
                        i += nfe3[ife]
                        continue
                    end
                    # populate the buffer storing the gradients against rij, rik, rjk
                    for m = 1:f.np
                        # Cache computed value
                        ijkp = pij[m, ife] * pik[m, ife]
                        for o = 1:f.nq  # Note that q is summed in the inner loop
                            # Feature term
                            val = ijkp * qjk[o, ife]
                            fvec[i, iat] += val
                            i += 1
                        end
                    end
                end  # 3body-feature update loop 
            end # i,j,k pair
        end
    end
    fvec, ecore
end
