#=
High-level optimization entry points.
=#
using LinearAlgebra

include("tpsd.jl")

"""
    opt_tpsd(vc::AbstractCalc; kwargs...) -> Bool

Optimize a structure using the native TPSD optimizer and return whether the
optimization converged.

This function preserves the legacy `Bool` return while delegating execution to
 the native optimizer framework.
"""
function opt_tpsd(
    vc::AbstractCalc;
    f_tol=1e-4,
    s_tol=1e-6,
    e_tol=1e-5,
    itermax=1000,
    trace=0,
    α_tol=1e-10,
    callback=nothing,
    trajectory=nothing,
)
    # Match legacy behavior: stress is ignored for fixed-cell optimizations.
    if !is_variable_cell(vc)
        s_tol = floatmax(Float64)
    end

    log_level = trace > 0 ? :standard : :minimal
    opt = NativeTPSD(;
        f_tol,
        s_tol,
        e_tol,
        itermax,
        α_tol,
        log_level,
        keep_trajectory=false,
    )

    wrapped_callback = isnothing(callback) && isnothing(trajectory) ? nothing : () -> begin
        isnothing(trajectory) || push!(trajectory, deepcopy(get_cell(vc)))
        isnothing(callback) || callback()
    end

    optimize!(opt, vc; callback=wrapped_callback)
    converged(opt, vc)
end

"""
    opt_fire(vc::AbstractCalc; kwargs...) -> NativeFIRE

Optimize a structure using the native FIRE optimizer and return the optimizer
state. This exposes detailed convergence/trajectory data for downstream tools.
"""
function opt_fire(
    vc::AbstractCalc;
    f_tol=1e-2,
    s_tol=1e-1,
    e_tol=1e-5,
    itermax=1000,
    maxstep=0.2,
    dt=0.1,
    dtmax=1.0,
    Nmin=5,
    finc=1.1,
    fdec=0.5,
    astart=0.1,
    fa=0.99,
    downhill_check=false,
    trace=false,
    verbose=trace,
    callback=nothing,
    keep_trajectory=false,
    trajectory=nothing,
)
    # Match legacy behavior: stress is ignored for fixed-cell optimizations.
    if !is_variable_cell(vc)
        s_tol = floatmax(Float64)
    end

    log_level = verbose ? :standard : :minimal
    opt = NativeFIRE(;
        dt,
        maxstep,
        dtmax,
        Nmin,
        finc,
        fdec,
        astart,
        fa,
        downhill_check,
        f_tol,
        s_tol,
        e_tol,
        itermax,
        log_level,
        keep_trajectory=keep_trajectory,
    )

    wrapped_callback = isnothing(callback) && isnothing(trajectory) ? nothing : () -> begin
        isnothing(trajectory) || push!(trajectory, deepcopy(get_cell(vc)))
        isnothing(callback) || callback()
    end

    optimize!(opt, vc; callback=wrapped_callback)
    opt
end
