#=
Generalised potential
=#
import Base
using CellBase
using StaticArrays
using Parameters

"""
    FeatureProto{P,Q}

Lightweight container for two-body/three-body feature parameters used
by `RandomEmbeddedCellFeature`. Replaces `TwoBodyFeature`/`ThreeBodyFeature`.

# Fields
- `p::P`: Tuple of exponents
- `q::Q`: Tuple of exponents (empty tuple for two-body)
- `rcut::Float64`: Cutoff radius
- `np::Int`: Number of p exponents
- `nq::Int`: Number of q exponents (0 for two-body)
"""
struct FeatureProto{P,Q,F,G}
    p::P
    q::Q
    rcut::Float64
    np::Int
    nq::Int
    f::F
    g::G
end

nfeatures(f::FeatureProto) = f.np * max(f.nq, 1)

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

include("spline_table.jl")
include("cell.jl")
include("embedded.jl")
include("multi_embedded.jl")
