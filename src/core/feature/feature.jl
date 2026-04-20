#=
Generalised potential
=#
import Base
using CellBase
using StaticArrays
using Parameters

"""
    AbstractNBodyFeature

Abstract supertype for N-body atomic environment descriptors.
Concrete subtypes: [`TwoBodyFeature`](@ref), [`ThreeBodyFeature`](@ref).
"""
abstract type AbstractNBodyFeature end

"""
    fast_pow(x, y::Int) -> typeof(x)

Fast integer power function optimized for small exponents.

This function computes x^y more efficiently than the standard `^` operator
for integer exponents by expanding powers 1-12 into explicit multiplications.
This avoids the overhead of general exponentiation and is critical for
performance in feature computation where many powers are calculated.

# Arguments
- `x`: Base value (any numeric type)
- `y::Int`: Integer exponent

# Returns
- Result of x^y with the same type as x

# Examples
```julia
# Much faster than x^5 for small powers
result = fast_pow(2.0, 5)  # 32.0

# Used internally in feature computation
f_val = fr(rij, rcut)
feature = fast_pow(f_val, 3)  # f_val³
```

# Notes
- Optimized for exponents -1 to 12
- Falls back to standard `^` for larger exponents
- Critical for EDDP performance (features computed millions of times)
- Returns 1 for y=0, 1/x for y=-1

# See also
[`TwoBodyFeature`](@ref), [`ThreeBodyFeature`](@ref)
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
    genp(pmin, pmax, np) -> Vector{Float64}

Generate exponents as a geometric sequence.

Creates a geometric series of exponents from pmin to pmax with np points.
This is the recommended way to choose feature exponents as it provides
better sampling across different length scales compared to linear spacing.

# Arguments
- `pmin`: Minimum exponent value
- `pmax`: Maximum exponent value
- `np`: Number of exponent values to generate

# Returns
- Vector of np exponents forming a geometric sequence

# Examples
```julia
# Generate 7 exponents from 2 to 8
p = genp(2, 8, 7)
# Result: [2.0, 2.49, 3.11, 3.87, 4.82, 6.01, 8.0] (approximately)

# Use for two-body features
feat = TwoBodyFeature(genp(2, 8, 7), (:C, :C), 4.0)

# Use for three-body features
p = genp(2, 6, 5)
q = genp(2, 4, 3)
feat = ThreeBodyFeature(p, q, (:C, :C, :C), 4.0)
```

# Notes
- Geometric spacing provides better resolution at both short and long ranges
- The ratio β between consecutive values is constant: β = (pmax/pmin)^(1/(np-1))
- Values are rounded to 3 decimal places
- Recommended for EDDP feature generation over linear spacing

# See also
[`TwoBodyFeature`](@ref), [`ThreeBodyFeature`](@ref), [`CellFeature`](@ref)
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
    permequal(A, i, j) -> Bool

Check if an ordered element pair matches exactly.

For two-body features, checks if the element tuple A matches the pair (i, j).
This code treats the first species as the central atom species and the second
as the neighbor species, so `(i, j)` and `(j, i)` are different channels.

# Arguments
- `A`: Tuple of element symbols from feature specification
- `i`: First element symbol
- `j`: Second element symbol

# Returns
- `true` if A matches (i, j), `false` otherwise

# Examples
```julia
# Check if feature applies to an ordered pair
feat = TwoBodyFeature([2, 3], (:C, :C), 4.0)
permequal(feat.sij_idx, :C, :C)  # true
permequal(feat.sij_idx, :C, :H)  # false

# Used internally during feature computation
if permequal(feat.sij_idx, atom_i_type, atom_j_type)
    # Compute this feature
end
```

# See also
[`TwoBodyFeature`](@ref), [`ThreeBodyFeature`](@ref)
"""
function permequal(A, i, j)
    (A[1] == i) && (A[2] == j) && return true
    false
end

"""
    permequal(A, i, j, k) -> Bool

Check if an element triplet matches with a fixed center and exchangeable neighbors.

For three-body features centered on atom i, checks if the element tuple A
matches the triplet (i, j, k). The feature is ordered in the first index and
unordered in the second and third indices, so it checks both `(i, j, k)` and
`(i, k, j)`.

# Arguments
- `A`: Tuple of three element symbols from feature specification
- `i`: Central atom element symbol
- `j`: First neighbor element symbol
- `k`: Second neighbor element symbol

# Returns
- `true` if A matches (i, j, k) or (i, k, j), `false` otherwise

# Examples
```julia
# Three-body feature for C-C-C triplets
feat = ThreeBodyFeature([2, 3], [2, 3], (:C, :C, :C), 4.0)

# Both neighbor orderings match
permequal(feat.sijk_idx, :C, :C, :C)  # true

# For C-H-H feature
feat_ch2 = ThreeBodyFeature([2], [2], (:C, :H, :H), 4.0)
permequal(feat_ch2.sijk_idx, :C, :H, :H)  # true (C center, H neighbors)
permequal(feat_ch2.sijk_idx, :H, :C, :H)  # false (H cannot be center)
```

# Notes
- First element (i) is always the central atom
- Second and third elements (j, k) are exchangeable
- Used internally to match features during computation

# See also
[`ThreeBodyFeature`](@ref), [`permequal(A, i, j)`](@ref)
"""
function permequal(A, i, j, k)
    (A[1] == i) && (A[2] == j) && (A[3] == k) && return true
    (A[1] == i) && (A[3] == j) && (A[2] == k) && return true
    false
end


"""
    TwoBodyFeature{T,M,P} <: AbstractNBodyFeature

Specification for two-body (pairwise) atomic environment features.

Two-body features describe the local environment of an atom based on distances
to its neighbors. Each feature is computed as f(r)^p where f(r) is a basis
function (typically with a cutoff) and p is an exponent. These features capture
radial information about the atomic environment.

# Fields
- `f::T`: Basis function f(r) that depends on interatomic distance
- `g::M`: Gradient df(r)/dr for force calculations
- `p::P`: Tuple of exponents to apply to f(r)
- `sij_idx::Tuple{Symbol,Symbol}`: Pair of element symbols (e.g., (:C, :C))
- `rcut::Float64`: Cutoff radius in Angstroms
- `np::Int`: Number of features (length of p)

# Constructors

## Basic constructor
```julia
TwoBodyFeature(p, sij_idx, rcut)
```
Uses default basis functions `fr` and `gfr` (smooth cutoff functions).

## Custom basis functions
```julia
TwoBodyFeature(f, g, p, sij_idx, rcut)
```
Specify custom basis function f and its gradient g.

# Examples
```julia
# Simple C-C two-body features with exponents 2, 3, 4
feat = TwoBodyFeature([2, 3, 4], (:C, :C), 4.0)
println("Number of features: ", nfeatures(feat))  # 3

# Using geometry sequence for exponents
p = genp(2, 8, 7)  # 7 exponents from 2 to 8 (geometric series)
feat = TwoBodyFeature(p, (:C, :H), 5.0)

# Features are computed as f(r)^p for each p
# where f(r) = 2(1 - r/rcut) for r ≤ rcut, 0 otherwise
```

# Notes
- Features are only computed for atom pairs matching `sij_idx` element types
- The default `fr` function provides smooth decay to zero at cutoff
- Higher exponents (larger p) capture shorter-range interactions
- Number of features scales as n_elements² × length(p)

# See also
[`ThreeBodyFeature`](@ref), [`CellFeature`](@ref), [`fr`](@ref), [`genp`](@ref)
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
    ThreeBodyFeature{T,M,P,Q} <: AbstractNBodyFeature

Specification for three-body (triplet) atomic environment features.

Three-body features describe the local environment of an atom based on triplets
of atoms (i-j-k). Each feature is computed as f(r_ij)^p × f(r_ik)^p × f(r_jk)^q,
capturing angular and geometric information about the atomic environment beyond
simple radial distributions.

# Fields
- `f::T`: Basis function f(r) that depends on interatomic distance
- `g::M`: Gradient df(r)/dr for force calculations
- `p::P`: Tuple of exponents for i-j and i-k distances
- `q::Q`: Tuple of exponents for j-k distance
- `sijk_idx::Tuple{Symbol,Symbol,Symbol}`: Triplet of element symbols
- `rcut::Float64`: Cutoff radius in Angstroms
- `np::Int`: Number of p exponents
- `nq::Int`: Number of q exponents

# Constructors

## Basic constructor
```julia
ThreeBodyFeature(p, q, sijk_idx, rcut)
```
Uses default basis functions `fr` and `gfr`.

## Custom basis functions
```julia
ThreeBodyFeature(f, g, p, q, sijk_idx, rcut)
```
Specify custom basis function f and its gradient g.

# Examples
```julia
# Simple C-C-C three-body features
p = [2, 3, 4]  # Exponents for i-j and i-k pairs
q = [2, 3]     # Exponents for j-k pair
feat = ThreeBodyFeature(p, q, (:C, :C, :C), 4.0)
println("Number of features: ", nfeatures(feat))  # 3 × 2 = 6

# Using geometry sequence
p = genp(2, 6, 5)  # 5 exponents from 2 to 6
q = genp(2, 4, 3)  # 3 exponents from 2 to 4
feat = ThreeBodyFeature(p, q, (:C, :H, :H), 5.0)
```

# Notes
- Features computed as: f(r_ij)^p × f(r_ik)^p × f(r_jk)^q
- Total features = np × nq for each unique element triplet
- More expensive to compute than two-body features
- Captures bond angles and geometric arrangements
- Number of features scales as n_elements³ × np × nq

# See also
[`TwoBodyFeature`](@ref), [`CellFeature`](@ref), [`fr`](@ref), [`genp`](@ref)
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
include("embedded.jl")
