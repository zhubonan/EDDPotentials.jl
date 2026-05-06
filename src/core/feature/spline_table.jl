using StaticArrays

struct SplineTable{NE}
    coeffs::Array{Float64,3}
    rmin::Float64
    inv_dr::Float64
    nintervals::Int
end

function _solve_natural_spline(y::AbstractVector{Float64}, n::Int, dr::Float64)
    σ = zeros(n)
    n <= 2 && return σ
    rhs = zeros(n - 2)
    for i in 2:n-1
        rhs[i-1] = (6.0 / (dr * dr)) * (y[i-1] - 2y[i] + y[i+1])
    end
    m = n - 2
    d = fill(4.0, m)
    upper = fill(1.0, m - 1)
    rhs_copy = copy(rhs)
    for i in 2:m
        factor = 1.0 / d[i-1]
        d[i] -= factor * upper[i-1]
        rhs_copy[i] -= factor * rhs_copy[i-1]
    end
    σ[n-1] = rhs_copy[m] / d[m]
    for i in m-1:-1:1
        σ[i+1] = (rhs_copy[i] - upper[i] * σ[i+2]) / d[i]
    end
    return σ
end

@inline function _horner_coeffs(yi, yi1, σi, σi1, h)
    (
        yi,
        (yi1 - yi) - h * h * (2σi + σi1) / 6,
        σi * h * h / 2,
        (σi1 - σi) * h * h / 6,
    )
end

@inline function _interval_lookup(tbl, r)
    t_raw = (r - tbl.rmin) * tbl.inv_dr
    idx = unsafe_trunc(Int, t_raw) + 1
    idx = min(idx, tbl.nintervals)
    idx = max(idx, 1)
    t = t_raw - (idx - 1)
    t = min(max(t, 0.0), 1.0)
    return idx, t
end

function _build_spline_coeffs(exponents::NTuple{NE, T}, rcut::Float64, ngrid::Int;
                              f::Function) where {NE, T}
    dr = rcut / (ngrid - 1)
    rgrid = range(0.0, stop=rcut, length=ngrid)
    nintervals = ngrid - 1
    coeffs = zeros(NE, nintervals, 4)
    h = dr
    for k in 1:NE
        y = Float64[f(r, rcut)^exponents[k] for r in rgrid]
        σ = _solve_natural_spline(y, ngrid, dr)
        for i in 1:nintervals
            c0, c1, c2, c3 = _horner_coeffs(y[i], y[i+1], σ[i], σ[i+1], h)
            coeffs[k, i, 1] = c0
            coeffs[k, i, 2] = c1
            coeffs[k, i, 3] = c2
            coeffs[k, i, 4] = c3
        end
    end
    return coeffs, 0.0, 1.0 / dr, nintervals
end

function build_spline_table(exponents::NTuple{NE, T}, rcut::Float64, ngrid::Int;
                            f::Function) where {NE, T}
    NE == 0 && return SplineTable{NE}(zeros(0, 0, 4), 0.0, 1.0, 0)
    coeffs, rmin, inv_dr, nintervals = _build_spline_coeffs(exponents, rcut, ngrid; f)
    return SplineTable{NE}(coeffs, rmin, inv_dr, nintervals)
end

for N in 0:12
    exprs = [:(@inbounds tbl.coeffs[$i, idx, 1] +
              t * (tbl.coeffs[$i, idx, 2] +
               t * (tbl.coeffs[$i, idx, 3] +
                t * tbl.coeffs[$i, idx, 4]))) for i in 1:N]
    @eval @inline function evaluate_svector(tbl::SplineTable{$N}, r::Float64)
        idx, t = _interval_lookup(tbl, r)
        SVector{$N, Float64}(tuple($(exprs...)))
    end
    @eval @inline function evaluate_svector(tbl::SplineTable{$N}, r::Float32)
        idx, t = _interval_lookup(tbl, r)
        SVector{$N, Float32}(tuple($(exprs...)))
    end
end
