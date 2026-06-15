#=
High-level optimization entry points.
=#
using LinearAlgebra

"""
    opt_tpsd(vc::AbstractCalc; kwargs...) -> NativeTPSD

Optimize a structure using the native TPSD optimizer and return the optimizer
state. Check convergence with `converged(opt, vc)`.
"""
function opt_tpsd(
    vc::AbstractCalc;
    f_tol=1e-4,
    s_tol=1e-6,
    e_tol=1e-5,
    convergence="force",
    require_residual_convergence=false,
    itermax=1000,
    trace=0,
    α_tol=1e-10,
    maxstep=0.2,
    max_cell_strain=0.1,
    relax_cell=false,
    ext_pressure=zeros(3, 3),
    cell_damping=1.0,
    callback=nothing,
    keep_trajectory=false,
    trajectory_every=1,
    trajectory=nothing,
    step_constraints=Any[],
)
    if !relax_cell
        s_tol = floatmax(Float64)
    end

    log_level = trace > 0 ? :standard : :minimal
    opt = NativeTPSD(;
        f_tol,
        s_tol,
        e_tol,
        convergence,
        require_residual_convergence,
        itermax,
        α_tol,
        maxstep,
        max_cell_strain,
        log_level,
        keep_trajectory=keep_trajectory,
        trajectory_every,
        relax_cell,
        ext_pressure,
        cell_damping,
        step_constraints=collect(step_constraints),
    )

    wrapped_callback = isnothing(callback) && isnothing(trajectory) ? nothing : () -> begin
        isnothing(trajectory) || push!(trajectory, deepcopy(get_cell(vc)))
        isnothing(callback) || callback()
    end

    optimize!(opt, vc; callback=wrapped_callback)
    opt
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
    convergence="force",
    require_residual_convergence=false,
    itermax=1000,
    maxstep=0.2,
    max_cell_strain=0.1,
    dt=0.1,
    dtmax=1.0,
    Nmin=5,
    finc=1.1,
    fdec=0.5,
    astart=0.1,
    fa=0.99,
    downhill_check=false,
    relax_cell=false,
    ext_pressure=zeros(3, 3),
    cell_damping=1.0,
    trace=false,
    verbose=trace,
    callback=nothing,
    keep_trajectory=false,
    trajectory_every=1,
    trajectory=nothing,
    step_constraints=Any[],
)
    if !relax_cell
        s_tol = floatmax(Float64)
    end

    log_level = verbose ? :standard : :minimal
    opt = NativeFIRE(;
        dt,
        maxstep,
        max_cell_strain,
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
        convergence,
        require_residual_convergence,
        itermax,
        log_level,
        keep_trajectory=keep_trajectory,
        trajectory_every,
        relax_cell,
        ext_pressure,
        cell_damping,
        step_constraints=collect(step_constraints),
    )

    wrapped_callback = isnothing(callback) && isnothing(trajectory) ? nothing : () -> begin
        isnothing(trajectory) || push!(trajectory, deepcopy(get_cell(vc)))
        isnothing(callback) || callback()
    end

    optimize!(opt, vc; callback=wrapped_callback)
    opt
end
