#=
Benchmark: Direct Evaluation vs Spline Interpolation for EDDP Feature Computation

This script compares three approaches for evaluating f(r)^p (the core operation
in EDDP feature generation):
  A. Direct evaluation (baseline) — fast_pow for Int exponents, ^ for Float64
  B. Interpolations.jl cubic splines
  C. Custom uniform-grid cubic interpolation (Horner's method)

The workload mimics the inner loops of compute_fv_gv_one! in gradient.jl,
particularly the _update_pij! function which is called O(N_neighbors²) per atom.
=#

using Pkg
# Ensure required packages are available
let deps = keys(Pkg.project().dependencies)
    missing_pkgs = String[]
    "BenchmarkTools" in deps || push!(missing_pkgs, "BenchmarkTools")
    "Interpolations" in deps || push!(missing_pkgs, "Interpolations")
    if !isempty(missing_pkgs)
        @info "Installing missing packages: $missing_pkgs"
        Pkg.add(missing_pkgs)
    end
end

using BenchmarkTools
using Interpolations
using Printf
using Statistics
using Random

# ============================================================================
# Section 1: Reproduce core functions from EDDPotentials (self-contained)
# ============================================================================

"Basis function: Eq. 7 in Pickard 2022"
@inline fr(r::T, rcut) where {T} = r <= rcut ? 2 * (1 - r / rcut) : zero(T)

"Gradient of basis function"
@inline gfr(r::T, rcut) where {T} = r <= rcut ? -2 / rcut : zero(T)

"Fast integer power via unrolled multiplications"
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
    return x^y
end

"Generate geometric sequence of exponents (production default)"
function genp(pmin, pmax, np)
    β = (pmax / pmin)^(1 / (np - 1))
    collect(round(pmin * β^(i - 1), digits=3) for i = 1:np)
end

# ============================================================================
# Section 2: Interpolation Implementations
# ============================================================================

# --- Method B: Interpolations.jl splines ---

"""
Build Interpolations.jl-based spline lookup for f(r)^p.

Returns (val_itp, grad_itp) where:
  val_itp(r, k) → fr(r, rcut)^exponents[k]
  grad_itp(r, k) → d/dr [fr(r, rcut)^exponents[k]]
"""
function build_interp_jl(exponents, rcut, ngrid)
    rgrid = range(0.0, stop=rcut, length=ngrid)
    nexp = length(exponents)

    # Build value table: A[i, k] = fr(rgrid[i], rcut)^exponents[k]
    A_val = zeros(ngrid, nexp)
    A_grad = zeros(ngrid, nexp)
    for k in 1:nexp
        for i in 1:ngrid
            r = rgrid[i]
            fval = fr(r, rcut)
            A_val[i, k] = fval^exponents[k]
            # Gradient: d/dr [f(r)^p] = p * f(r)^(p-1) * f'(r)
            if fval > 0 && r < rcut
                A_grad[i, k] = exponents[k] * fval^(exponents[k] - 1) * gfr(r, rcut)
            else
                A_grad[i, k] = 0.0
            end
        end
    end

    val_raw = interpolate(A_val, (BSpline(Cubic(Natural(OnGrid()))), NoInterp()))
    val_itp = scale(val_raw, rgrid, 1:nexp)

    grad_raw = interpolate(A_grad, (BSpline(Cubic(Natural(OnGrid()))), NoInterp()))
    grad_itp = scale(grad_raw, rgrid, 1:nexp)

    return val_itp, grad_itp
end

# --- Method C: Custom uniform-grid cubic interpolation ---

"""
Cubic spline coefficient table for uniform grid.

For interval i (r in [r_i, r_{i+1}]), with local parameter t = (r - r_i) / dr:
  f(r) ≈ coeffs[1,i,k] + t*(coeffs[2,i,k] + t*(coeffs[3,i,k] + t*coeffs[4,i,k]))
"""
struct CubicLookupTable
    coeffs::Array{Float64,3}   # (4, nintervals, nexp) — Horner coefficients
    rmin::Float64
    inv_dr::Float64            # 1/dr for fast interval lookup
    nintervals::Int
    nexp::Int
end

"""
Build custom cubic spline lookup table.

Uses natural cubic spline conditions (second derivative = 0 at boundaries).
"""
function build_cubic_table(exponents, rcut, ngrid)
    nexp = length(exponents)
    dr = rcut / (ngrid - 1)
    rgrid = range(0.0, stop=rcut, length=ngrid)
    nintervals = ngrid - 1

    # For each exponent, compute natural cubic spline coefficients
    coeffs = zeros(4, nintervals, nexp)

    for k in 1:nexp
        # Sample function values
        y = [fr(r, rcut)^exponents[k] for r in rgrid]

        # Solve for natural cubic spline second derivatives (tridiagonal system)
        # Natural BCs: σ[1] = σ[n] = 0
        n = ngrid
        σ = zeros(n)  # second derivatives at knots

        # Set up tridiagonal system for interior points
        # For uniform grid: σ[i-1] + 4σ[i] + σ[i+1] = (6/h²)(y[i-1] - 2y[i] + y[i+1])
        if n > 2
            rhs = zeros(n - 2)
            for i in 2:n-1
                rhs[i-1] = (6.0 / (dr * dr)) * (y[i-1] - 2y[i] + y[i+1])
            end

            # Solve tridiagonal: [4 1 0 ...] [σ2]   [rhs1]
            #                    [1 4 1 ...] [σ3] = [rhs2]
            #                    [... 1 4 1] [...]   [...]
            m = n - 2
            d = fill(4.0, m)
            upper = fill(1.0, m - 1)
            lower = fill(1.0, m - 1)

            # Forward elimination
            for i in 2:m
                factor = lower[i-1] / d[i-1]
                d[i] -= factor * upper[i-1]
                rhs[i] -= factor * rhs[i-1]
            end
            # Back substitution
            σ[n-1] = rhs[m] / d[m]
            for i in m-1:-1:1
                σ[i+1] = (rhs[i] - upper[i] * σ[i+2]) / d[i]
            end
        end

        # Convert to Horner coefficients for each interval
        # Standard cubic spline on interval [x_i, x_{i+1}] with t = (x - x_i)/h:
        #   S(t) = a + b*t + c*t² + d*t³
        # where:
        #   a = y[i]
        #   b = (y[i+1] - y[i]) - h/6 * (2σ[i] + σ[i+1]) * h  ... but with t normalized
        # We use t ∈ [0, 1], so h factors are absorbed:
        for i in 1:nintervals
            a = y[i]
            # With t = (r - r_i)/dr ∈ [0,1]:
            # S(t) = (1-t)*y[i] + t*y[i+1] + t*(1-t)*((1-t)*(-σ[i]*dr²/6 - (y[i+1]-y[i])/3 + ...) )
            # Simpler: convert standard spline form to Horner form with t ∈ [0,1]
            h = dr
            σi = σ[i]
            σi1 = σ[i+1]
            yi = y[i]
            yi1 = y[i+1]

            # Standard form: S(x) = σi/(6h)*(x_{i+1}-x)³ + σi1/(6h)*(x-x_i)³
            #                      + (yi/h - σi*h/6)*(x_{i+1}-x) + (yi1/h - σi1*h/6)*(x-x_i)
            # Let u = (x - x_i), so x_{i+1} - x = h - u:
            # S(u) = σi/(6h)*(h-u)³ + σi1/(6h)*u³ + (yi/h - σi*h/6)*(h-u) + (yi1/h - σi1*h/6)*u
            # Let t = u/h (t ∈ [0,1]), u = t*h:
            # S(t) = σi*h²/6*(1-t)³ + σi1*h²/6*t³ + yi*(1-t) + yi1*t - h²/6*(σi*(1-t) + σi1*t)  ... wait

            # Cleaner derivation. With t = (r - r_i)/h ∈ [0,1]:
            # S(t) = (1-t)*yi + t*yi1 + h²*t*(1-t)*((1-t)*(-σi/6) + t*(-σi1/6))
            #       ... no, let me use the standard result directly.
            # S(t) = yi*(1-t) + yi1*t + h²/6 * [σi*((1-t)³ - (1-t)) + σi1*(t³ - t)]
            # Expand:
            #   (1-t)³ - (1-t) = (1-t)*((1-t)² - 1) = (1-t)*(t² - 2t) = (1-t)*t*(t-2)
            #   t³ - t = t*(t²-1) = t*(t-1)*(t+1)
            # So:
            #   S(t) = yi + (yi1-yi)*t + h²/6 * [σi*(1-t)*t*(t-2) + σi1*t*(t-1)*(t+1)]
            #        = yi + (yi1-yi)*t + h²*t/6 * [σi*(1-t)*(t-2) + σi1*(t-1)*(t+1)]
            #        = yi + (yi1-yi)*t + h²*t/6 * [σi*(t-2-t²+2t) + σi1*(t²-1+t³-t)]  # wrong
            # Let me just expand directly as polynomial in t:
            # (1-t)³ = 1 - 3t + 3t² - t³
            # S(t) = yi - yi*t + yi1*t + h²/6*[σi*(1-3t+3t²-t³ - 1+t) + σi1*(t³-t)]
            #       = yi + (yi1-yi)*t + h²/6*[σi*(-2t+3t²-t³) + σi1*(t³-t)]
            #       = yi + (yi1-yi)*t + h²/6*[-2σi*t + 3σi*t² - σi*t³ + σi1*t³ - σi1*t]
            #       = yi + [(yi1-yi) - h²/6*(2σi+σi1)]*t + [h²/6*3σi]*t² + [h²/6*(σi1-σi)]*t³
            #       = yi + [(yi1-yi) - h²*(2σi+σi1)/6]*t + [σi*h²/2]*t² + [(σi1-σi)*h²/6]*t³

            c0 = yi
            c1 = (yi1 - yi) - h * h * (2σi + σi1) / 6
            c2 = σi * h * h / 2
            c3 = (σi1 - σi) * h * h / 6

            # Horner form: c0 + t*(c1 + t*(c2 + t*c3))
            coeffs[1, i, k] = c0
            coeffs[2, i, k] = c1
            coeffs[3, i, k] = c2
            coeffs[4, i, k] = c3
        end
    end

    return CubicLookupTable(coeffs, 0.0, 1.0 / dr, nintervals, nexp)
end

"Evaluate custom cubic lookup at distance r for exponent index k"
@inline function evaluate(tbl::CubicLookupTable, r::Float64, k::Int)
    # O(1) interval lookup for uniform grid
    t_raw = (r - tbl.rmin) * tbl.inv_dr
    idx = unsafe_trunc(Int, t_raw) + 1
    # Clamp to valid range
    idx = min(idx, tbl.nintervals)
    idx = max(idx, 1)
    t = t_raw - (idx - 1)
    # Clamp t to [0, 1]
    t = min(max(t, 0.0), 1.0)
    # Horner evaluation
    @inbounds c0 = tbl.coeffs[1, idx, k]
    @inbounds c1 = tbl.coeffs[2, idx, k]
    @inbounds c2 = tbl.coeffs[3, idx, k]
    @inbounds c3 = tbl.coeffs[4, idx, k]
    return c0 + t * (c1 + t * (c2 + t * c3))
end

"Evaluate all exponents at once for a given distance, storing in out[1:nexp]"
@inline function evaluate_all!(out::AbstractVector, tbl::CubicLookupTable, r::Float64)
    t_raw = (r - tbl.rmin) * tbl.inv_dr
    idx = unsafe_trunc(Int, t_raw) + 1
    idx = min(idx, tbl.nintervals)
    idx = max(idx, 1)
    t = t_raw - (idx - 1)
    t = min(max(t, 0.0), 1.0)
    @inbounds for k in 1:tbl.nexp
        c0 = tbl.coeffs[1, idx, k]
        c1 = tbl.coeffs[2, idx, k]
        c2 = tbl.coeffs[3, idx, k]
        c3 = tbl.coeffs[4, idx, k]
        out[k] = c0 + t * (c1 + t * (c2 + t * c3))
    end
end

# ============================================================================
# Section 3: Benchmark Kernels
# ============================================================================

# --- Scenario 1: Two-body kernel ---
# Mirrors gradient.jl:213-218
# For each neighbor, evaluate f(r)^p for all exponents, accumulate into fvec

function kernel_twobody_direct!(fvec, distances, exponents::Vector{Int}, rcut)
    for r in distances
        fval = fr(r, rcut)
        for (k, p) in enumerate(exponents)
            fvec[k] += fast_pow(fval, p)
        end
    end
end

function kernel_twobody_direct!(fvec, distances, exponents::Vector{Float64}, rcut)
    for r in distances
        fval = fr(r, rcut)
        for (k, p) in enumerate(exponents)
            fvec[k] += fval^p
        end
    end
end

function kernel_twobody_interp!(fvec, distances, val_itp, nexp)
    for r in distances
        for k in 1:nexp
            fvec[k] += val_itp(r, k)
        end
    end
end

function kernel_twobody_custom!(fvec, distances, tbl::CubicLookupTable)
    buf = zeros(tbl.nexp)
    for r in distances
        evaluate_all!(buf, tbl, r)
        for k in 1:tbl.nexp
            fvec[k] += buf[k]
        end
    end
end

# --- Scenario 2: Three-body _update_pij! kernel ---
# Mirrors gradient.jl:46-59
# For one distance, evaluate f(r)^p for all exponents into a pre-allocated array

function kernel_update_pij_direct!(pij, distances, exponents::Vector{Int}, rcut)
    for (j, r) in enumerate(distances)
        fval = fr(r, rcut)
        for (k, p) in enumerate(exponents)
            pij[k, j] = fast_pow(fval, p)
        end
    end
end

function kernel_update_pij_direct!(pij, distances, exponents::Vector{Float64}, rcut)
    for (j, r) in enumerate(distances)
        fval = fr(r, rcut)
        for (k, p) in enumerate(exponents)
            pij[k, j] = fval^p
        end
    end
end

function kernel_update_pij_interp!(pij, distances, val_itp, nexp)
    for (j, r) in enumerate(distances)
        for k in 1:nexp
            pij[k, j] = val_itp(r, k)
        end
    end
end

function kernel_update_pij_custom!(pij, distances, tbl::CubicLookupTable)
    buf = zeros(tbl.nexp)
    for (j, r) in enumerate(distances)
        evaluate_all!(buf, tbl, r)
        for k in 1:tbl.nexp
            pij[k, j] = buf[k]
        end
    end
end

# --- Scenario 3: With gradients ---
# Mirrors gradient.jl:220-227 and 298-305
# Evaluate both f(r)^p and d/dr[f(r)^p]

function kernel_with_grad_direct!(fvec, gvec, distances, exponents::Vector{Int}, rcut)
    for r in distances
        fval = fr(r, rcut)
        gval = gfr(r, rcut)
        for (k, p) in enumerate(exponents)
            val = fast_pow(fval, p)
            fvec[k] += val
            if val != 0.0
                gvec[k] += p * val / fval * gval
            end
        end
    end
end

function kernel_with_grad_direct!(fvec, gvec, distances, exponents::Vector{Float64}, rcut)
    for r in distances
        fval = fr(r, rcut)
        gval = gfr(r, rcut)
        for (k, p) in enumerate(exponents)
            val = fval^p
            fvec[k] += val
            if val != 0.0
                gvec[k] += p * val / fval * gval
            end
        end
    end
end

function kernel_with_grad_interp!(fvec, gvec, distances, val_itp, grad_itp, nexp)
    for r in distances
        for k in 1:nexp
            fvec[k] += val_itp(r, k)
            gvec[k] += grad_itp(r, k)
        end
    end
end

function kernel_with_grad_custom!(fvec, gvec, distances, val_tbl, grad_tbl)
    vbuf = zeros(val_tbl.nexp)
    gbuf = zeros(grad_tbl.nexp)
    for r in distances
        evaluate_all!(vbuf, val_tbl, r)
        evaluate_all!(gbuf, grad_tbl, r)
        for k in 1:val_tbl.nexp
            fvec[k] += vbuf[k]
            gvec[k] += gbuf[k]
        end
    end
end

# ============================================================================
# Section 4: Accuracy Measurement
# ============================================================================

function measure_accuracy(exponents, rcut, ngrid; ntest=10000)
    val_itp, grad_itp = build_interp_jl(exponents, rcut, ngrid)
    val_tbl = build_cubic_table(exponents, rcut, ngrid)

    # Build gradient table
    nexp = length(exponents)
    grad_vals = zeros(ngrid, nexp)
    rgrid = range(0.0, stop=rcut, length=ngrid)
    for k in 1:nexp
        for i in 1:ngrid
            r = rgrid[i]
            fval = fr(r, rcut)
            if fval > 0 && r < rcut
                grad_vals[i, k] = exponents[k] * fval^(exponents[k] - 1) * gfr(r, rcut)
            end
        end
    end
    grad_tbl = build_cubic_table_from_values(grad_vals, rcut)

    # Test at random points
    rng = MersenneTwister(42)
    test_r = rand(rng, ntest) .* (rcut - 0.01) .+ 0.005

    max_err_interp = 0.0
    max_err_custom = 0.0
    sum_sq_interp = 0.0
    sum_sq_custom = 0.0
    count = 0

    for r in test_r
        fval = fr(r, rcut)
        for (k, p) in enumerate(exponents)
            exact = fval^p
            interp_val = val_itp(r, k)
            custom_val = evaluate(val_tbl, r, k)

            err_i = abs(interp_val - exact)
            err_c = abs(custom_val - exact)

            max_err_interp = max(max_err_interp, err_i)
            max_err_custom = max(max_err_custom, err_c)
            sum_sq_interp += err_i^2
            sum_sq_custom += err_c^2
            count += 1
        end
    end

    rmse_interp = sqrt(sum_sq_interp / count)
    rmse_custom = sqrt(sum_sq_custom / count)

    return (;
        max_err_interp, rmse_interp,
        max_err_custom, rmse_custom,
    )
end

"Build cubic table directly from pre-computed sample values (ngrid × nexp matrix)"
function build_cubic_table_from_values(values::Matrix{Float64}, rcut)
    ngrid, nexp = size(values)
    dr = rcut / (ngrid - 1)
    nintervals = ngrid - 1
    coeffs = zeros(4, nintervals, nexp)

    for k in 1:nexp
        y = @view values[:, k]
        n = ngrid
        σ = zeros(n)

        if n > 2
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
        end

        h = dr
        for i in 1:nintervals
            yi = y[i]
            yi1 = y[i+1]
            σi = σ[i]
            σi1 = σ[i+1]
            coeffs[1, i, k] = yi
            coeffs[2, i, k] = (yi1 - yi) - h * h * (2σi + σi1) / 6
            coeffs[3, i, k] = σi * h * h / 2
            coeffs[4, i, k] = (σi1 - σi) * h * h / 6
        end
    end

    return CubicLookupTable(coeffs, 0.0, 1.0 / dr, nintervals, nexp)
end

# ============================================================================
# Section 5: Main Benchmark
# ============================================================================

function run_benchmarks()
    println("=" ^ 80)
    println("EDDP Feature Computation: Direct vs Spline Interpolation Benchmark")
    println("=" ^ 80)

    rcut = 6.0

    # Two exponent sets
    exponents_int = [2, 3, 4, 5, 6, 7, 8]
    exponents_float = genp(2, 10, 5)

    println("\nExponent sets:")
    println("  Integer: $exponents_int")
    println("  Float64 (genp): $exponents_float")
    println("  rcut = $rcut")

    # Workload parameters
    n_neighbors = 60
    n_atoms = 50

    # Generate realistic test distances
    rng = MersenneTwister(123)
    all_distances = [sort(rand(rng, n_neighbors) .* (rcut - 0.5) .+ 0.5) for _ in 1:n_atoms]

    grid_sizes = [100, 500, 1000, 5000]

    # ========================================================================
    # Part 1: Accuracy Analysis
    # ========================================================================
    println("\n" * "=" ^ 80)
    println("PART 1: ACCURACY ANALYSIS")
    println("=" ^ 80)

    for (label, exponents) in [("Integer", exponents_int), ("Float64", exponents_float)]
        println("\n--- Exponents: $label ---")
        @printf("%-10s  %15s  %15s  %15s  %15s\n",
            "Grid pts", "MaxErr(InterpJl)", "RMSE(InterpJl)", "MaxErr(Custom)", "RMSE(Custom)")
        @printf("%-10s  %15s  %15s  %15s  %15s\n",
            "-" ^ 10, "-" ^ 15, "-" ^ 15, "-" ^ 15, "-" ^ 15)

        for ngrid in grid_sizes
            acc = measure_accuracy(exponents, rcut, ngrid)
            @printf("%-10d  %15.2e  %15.2e  %15.2e  %15.2e\n",
                ngrid, acc.max_err_interp, acc.rmse_interp,
                acc.max_err_custom, acc.rmse_custom)
        end
    end

    # ========================================================================
    # Part 2: Performance Benchmarks
    # ========================================================================
    println("\n" * "=" ^ 80)
    println("PART 2: PERFORMANCE BENCHMARKS")
    println("=" ^ 80)

    # Use a good grid size for performance tests
    ngrid_perf = 1000

    for (label, exponents) in [("Integer", exponents_int), ("Float64", exponents_float)]
        nexp = length(exponents)

        println("\n" * "-" ^ 60)
        println("Exponents: $label ($nexp exponents), Grid: $ngrid_perf points")
        println("-" ^ 60)

        # Build interpolation objects
        val_itp, grad_itp = build_interp_jl(exponents, rcut, ngrid_perf)
        val_tbl = build_cubic_table(exponents, rcut, ngrid_perf)

        # Build gradient tables
        rgrid = range(0.0, stop=rcut, length=ngrid_perf)
        grad_vals = zeros(ngrid_perf, nexp)
        for k in 1:nexp
            for i in 1:ngrid_perf
                r = rgrid[i]
                fval = fr(r, rcut)
                if fval > 0 && r < rcut
                    grad_vals[i, k] = exponents[k] * fval^(exponents[k] - 1) * gfr(r, rcut)
                end
            end
        end
        grad_tbl = build_cubic_table_from_values(grad_vals, rcut)

        # --- Scenario 1: Two-body kernel ---
        println("\n  Scenario 1: Two-body kernel ($n_neighbors neighbors × $n_atoms atoms)")

        # Warmup + benchmark
        fvec = zeros(nexp)
        t_direct = @benchmark begin
            fill!($fvec, 0.0)
            for dists in $all_distances
                kernel_twobody_direct!($fvec, dists, $exponents, $rcut)
            end
        end

        fvec_i = zeros(nexp)
        t_interp = @benchmark begin
            fill!($fvec_i, 0.0)
            for dists in $all_distances
                kernel_twobody_interp!($fvec_i, dists, $val_itp, $nexp)
            end
        end

        fvec_c = zeros(nexp)
        t_custom = @benchmark begin
            fill!($fvec_c, 0.0)
            for dists in $all_distances
                kernel_twobody_custom!($fvec_c, dists, $val_tbl)
            end
        end

        med_d = median(t_direct).time / 1e6
        med_i = median(t_interp).time / 1e6
        med_c = median(t_custom).time / 1e6
        @printf("    Direct:         %8.3f ms\n", med_d)
        @printf("    Interp.jl:      %8.3f ms  (%.2f× vs direct)\n", med_i, med_d / med_i)
        @printf("    Custom cubic:   %8.3f ms  (%.2f× vs direct)\n", med_c, med_d / med_c)

        # --- Scenario 2: _update_pij! kernel ---
        # In the real code, _update_pij! is called once per neighbor in the outer
        # three-body loop. We simulate this by calling it for each atom's neighbor list.
        println("\n  Scenario 2: _update_pij! kernel ($n_neighbors distances × $n_atoms atoms)")

        pij = zeros(nexp, n_neighbors)
        t_direct2 = @benchmark begin
            for dists in $all_distances
                kernel_update_pij_direct!($pij, dists, $exponents, $rcut)
            end
        end setup=(fill!($pij, 0.0))
        t_interp2 = @benchmark begin
            for dists in $all_distances
                kernel_update_pij_interp!($pij, dists, $val_itp, $nexp)
            end
        end setup=(fill!($pij, 0.0))
        t_custom2 = @benchmark begin
            for dists in $all_distances
                kernel_update_pij_custom!($pij, dists, $val_tbl)
            end
        end setup=(fill!($pij, 0.0))

        med_d2 = median(t_direct2).time / 1e6
        med_i2 = median(t_interp2).time / 1e6
        med_c2 = median(t_custom2).time / 1e6
        @printf("    Direct:         %8.3f ms\n", med_d2)
        @printf("    Interp.jl:      %8.3f ms  (%.2f× vs direct)\n", med_i2, med_d2 / med_i2)
        @printf("    Custom cubic:   %8.3f ms  (%.2f× vs direct)\n", med_c2, med_d2 / med_c2)

        # --- Scenario 3: With gradients ---
        println("\n  Scenario 3: With gradients ($n_neighbors neighbors × $n_atoms atoms)")

        gvec = zeros(nexp)
        t_direct3 = @benchmark begin
            fill!($fvec, 0.0)
            fill!($gvec, 0.0)
            for dists in $all_distances
                kernel_with_grad_direct!($fvec, $gvec, dists, $exponents, $rcut)
            end
        end

        gvec_i = zeros(nexp)
        t_interp3 = @benchmark begin
            fill!($fvec_i, 0.0)
            fill!($gvec_i, 0.0)
            for dists in $all_distances
                kernel_with_grad_interp!($fvec_i, $gvec_i, dists, $val_itp, $grad_itp, $nexp)
            end
        end

        gvec_c = zeros(nexp)
        t_custom3 = @benchmark begin
            fill!($fvec_c, 0.0)
            fill!($gvec_c, 0.0)
            for dists in $all_distances
                kernel_with_grad_custom!($fvec_c, $gvec_c, dists, $val_tbl, $grad_tbl)
            end
        end

        med_d3 = median(t_direct3).time / 1e6
        med_i3 = median(t_interp3).time / 1e6
        med_c3 = median(t_custom3).time / 1e6
        @printf("    Direct:         %8.3f ms\n", med_d3)
        @printf("    Interp.jl:      %8.3f ms  (%.2f× vs direct)\n", med_i3, med_d3 / med_i3)
        @printf("    Custom cubic:   %8.3f ms  (%.2f× vs direct)\n", med_c3, med_d3 / med_c3)
    end

    # ========================================================================
    # Part 3: Grid Size vs Speed Tradeoff
    # ========================================================================
    println("\n" * "=" ^ 80)
    println("PART 3: GRID SIZE vs SPEED TRADEOFF (Float64 exponents, Scenario 2)")
    println("=" ^ 80)

    exponents = exponents_float
    nexp = length(exponents)

    # Baseline direct timing
    pij = zeros(nexp, n_neighbors)
    t_base = @benchmark begin
        for dists in $all_distances
            kernel_update_pij_direct!($pij, dists, $exponents, $rcut)
        end
    end setup=(fill!($pij, 0.0))
    med_base = median(t_base).time / 1e6

    @printf("\n%-10s  %10s  %10s  %12s  %12s  %12s  %12s\n",
        "Grid pts", "MaxErr(IJ)", "MaxErr(C)", "Time(IJ) ms", "Time(C) ms", "Speedup(IJ)", "Speedup(C)")
    @printf("%-10s  %10s  %10s  %12s  %12s  %12s  %12s\n",
        "-" ^ 10, "-" ^ 10, "-" ^ 10, "-" ^ 12, "-" ^ 12, "-" ^ 12, "-" ^ 12)

    for ngrid in grid_sizes
        acc = measure_accuracy(exponents, rcut, ngrid)

        val_itp, _ = build_interp_jl(exponents, rcut, ngrid)
        val_tbl = build_cubic_table(exponents, rcut, ngrid)

        t_i = @benchmark begin
            for dists in $all_distances
                kernel_update_pij_interp!($pij, dists, $val_itp, $nexp)
            end
        end setup=(fill!($pij, 0.0))
        t_c = @benchmark begin
            for dists in $all_distances
                kernel_update_pij_custom!($pij, dists, $val_tbl)
            end
        end setup=(fill!($pij, 0.0))

        med_i = median(t_i).time / 1e6
        med_c = median(t_c).time / 1e6

        @printf("%-10d  %10.2e  %10.2e  %12.3f  %12.3f  %12.2f×  %12.2f×\n",
            ngrid, acc.max_err_interp, acc.max_err_custom,
            med_i, med_c, med_base / med_i, med_base / med_c)
    end

    @printf("\nDirect baseline (Float64 exponents): %.3f ms\n", med_base)

    # ========================================================================
    # Part 4: Int vs Float64 Exponent Comparison
    # ========================================================================
    println("\n" * "=" ^ 80)
    println("PART 4: INTEGER vs FLOAT64 EXPONENT COST COMPARISON")
    println("=" ^ 80)
    println("\nScenario 2 (_update_pij! kernel), $n_neighbors distances × $n_atoms atoms")

    # Integer exponents — direct
    pij_int = zeros(length(exponents_int), n_neighbors)
    t_int_direct = @benchmark begin
        for dists in $all_distances
            kernel_update_pij_direct!($pij_int, dists, $exponents_int, $rcut)
        end
    end setup=(fill!($pij_int, 0.0))

    # Float64 exponents — direct
    pij_flt = zeros(length(exponents_float), n_neighbors)
    t_flt_direct = @benchmark begin
        for dists in $all_distances
            kernel_update_pij_direct!($pij_flt, dists, $exponents_float, $rcut)
        end
    end setup=(fill!($pij_flt, 0.0))

    # Float64 exponents — custom cubic (1000 grid)
    val_tbl_flt = build_cubic_table(exponents_float, rcut, 1000)
    t_flt_custom = @benchmark begin
        for dists in $all_distances
            kernel_update_pij_custom!($pij_flt, dists, $val_tbl_flt)
        end
    end setup=(fill!($pij_flt, 0.0))

    # Integer exponents — custom cubic (1000 grid) for reference
    val_tbl_int = build_cubic_table(Float64.(exponents_int), rcut, 1000)
    pij_int_c = zeros(length(exponents_int), n_neighbors)
    t_int_custom = @benchmark begin
        for dists in $all_distances
            kernel_update_pij_custom!($pij_int_c, dists, $val_tbl_int)
        end
    end setup=(fill!($pij_int_c, 0.0))

    med_id = median(t_int_direct).time / 1e6
    med_fd = median(t_flt_direct).time / 1e6
    med_fc = median(t_flt_custom).time / 1e6
    med_ic = median(t_int_custom).time / 1e6

    # Normalize to per-exponent cost
    nexp_int = length(exponents_int)
    nexp_flt = length(exponents_float)
    per_id = median(t_int_direct).time / (n_atoms * n_neighbors * nexp_int)  # ns per eval
    per_fd = median(t_flt_direct).time / (n_atoms * n_neighbors * nexp_flt)
    per_fc = median(t_flt_custom).time / (n_atoms * n_neighbors * nexp_flt)
    per_ic = median(t_int_custom).time / (n_atoms * n_neighbors * nexp_int)

    println("\n  Total time (all atoms, all neighbors, all exponents):")
    @printf("    Int  direct (fast_pow, %d exp):   %8.3f ms\n", nexp_int, med_id)
    @printf("    Int  custom cubic (%d exp):       %8.3f ms\n", nexp_int, med_ic)
    @printf("    Flt  direct (x^p, %d exp):        %8.3f ms\n", nexp_flt, med_fd)
    @printf("    Flt  custom cubic (%d exp):        %8.3f ms\n", nexp_flt, med_fc)

    println("\n  Per-evaluation cost (single r, single exponent):")
    @printf("    Int  direct (fast_pow):    %8.2f ns\n", per_id)
    @printf("    Int  custom cubic:         %8.2f ns\n", per_ic)
    @printf("    Flt  direct (x^p):         %8.2f ns\n", per_fd)
    @printf("    Flt  custom cubic:         %8.2f ns\n", per_fc)

    println("\n  Ratios:")
    @printf("    Float64 direct / Int direct:       %5.1f×  (cost of exp+log vs multiply)\n", per_fd / per_id)
    @printf("    Float64 direct / Flt custom:       %5.1f×  (speedup from spline)\n", per_fd / per_fc)
    @printf("    Float64 custom / Int direct:       %5.1f×  (spline vs fast_pow parity)\n", per_fc / per_id)

    println("\n" * "=" ^ 80)
    println("DONE")
    println("=" ^ 80)
end

# Run everything
run_benchmarks()
