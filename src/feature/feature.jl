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
struct TwoBodyFeature{T,M,P<:Tuple} <: AbstractNBodyFeature
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
    TwoBodyFeature(f, g, Tuple(p), tuple(sij_idx[1], sij_idx[2]), rcut, length(p))
TwoBodyFeature(p, sij_idx, rcut::Real) = TwoBodyFeature(fr, gfr, Tuple(p), sij_idx, rcut)

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
struct ThreeBodyFeature{T,M,P<:Tuple,Q<:Tuple} <: AbstractNBodyFeature
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
    Tuple(p),
    Tuple(q),
    tuple(sijk_idx[1], sijk_idx[2], sijk_idx[3]),
    rcut,
    length(p),
    length(q),
)
ThreeBodyFeature(p, q, sijk_idx, rcut::Float64) =
    ThreeBodyFeature(fr, gfr, Tuple(p), Tuple(q), sijk_idx, rcut)


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

include("cell.jl")
include("lookup.jl")
include("compute.jl")