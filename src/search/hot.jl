#=
Hot-AIRSS protocol: alternating relaxation and molecular dynamics
for escaping local minima in crystal structure search.
=#

using Printf
using Configurations

# ─────────────────────────────────────────────────────────────
# Option and result types
# ─────────────────────────────────────────────────────────────

"""
    HotAIRSSOption <: EDDPotentialsOption

Configuration for the hot-AIRSS protocol.

Each cycle runs a fixed number of relaxation steps to quench toward a local
minimum, then runs MD steps at high temperature to let the structure explore
the energy landscape. A final relaxation ensures the result ends at a local
minimum.

# Fields

**Cycle control:**
- `cycles::Int` — number of relax–MD cycles (default 10)

**Relaxation phase:**
- `relax_itermax::Int` — max iterations per relaxation phase (default 200)
- `relax_f_tol::Float64` — force convergence tolerance in eV/Å (default 1e-2)
- `relax_s_tol::Float64` — stress convergence tolerance in internal eV/Å³
  units. Prefer `relax_s_tol_gpa` for user-facing configuration.
- `relax_s_tol_gpa::Float64` — stress convergence tolerance in GPa. Set to
  a nonnegative value to override `relax_s_tol` (default 0.1 GPa).
- `relax_e_tol::Float64` — energy convergence tolerance in eV (default 1e-6)
- `relax_convergence::String` — "step_energy" or "force" convergence mode.
- `relax_method::String` — "tpsd" or "fire" (default "tpsd")
- `relax_cell::Bool` — enable cell relaxation via optimizer's relax_cell parameter (default true)
- `relax_max_cell_strain::Float64` — cap on per-step cell strain norm (default 0.0125)
- `relax_cell_damping::Float64` — native optimizer cell-step gamma damping (default 1.0)
- `external_pressure::Vector{Float64}` — deprecated external pressure in
  internal eV/Å³ stress units (default [0.0]).
- `external_pressure_gpa::Union{Nothing,Float64}` — hydrostatic external
  pressure in GPa. Positive values are compressive and negative values are
  tensile; `nothing` leaves the deprecated `external_pressure` path in use.
- `symmetry::String` — "auto", "on", or "off" symmetry-aware relaxation mode.
- `symmetry_tolerance::Float64` — symmetry detection tolerance.

**MD phase:**
- `md_steps::Int` — MD steps per cycle (default 50)
- `md_temperature::Float64` — target temperature in K (default 1000.0)
- `md_dt::Float64` — timestep in fs (default 1.0)
- `md_integrator::String` — "langevin", "berendsen", or "nve" (default "langevin")
- `md_friction::Float64` — Langevin friction in 1/fs (default 0.01)
- `md_taut::Float64` — Berendsen coupling time in fs (default 50.0)
- `md_seed::Union{Int,Nothing}` — RNG seed for reproducibility (default nothing)

**Output:**
- `verbose::Bool` — print progress (default false)
"""
@option mutable struct HotAIRSSOption <: EDDPotentialsOption
    # Cycle control
    cycles::Int = 10
    # Relaxation phase
    relax_itermax::Int = 200
    relax_f_tol::Float64 = 1e-2
    relax_s_tol::Float64 = GPaToeVAng(0.1)
    relax_s_tol_gpa::Float64 = 0.1
    relax_e_tol::Float64 = 1e-6
    relax_convergence::String = "step_energy"
    relax_method::String = "tpsd"
    relax_cell::Bool = true
    relax_max_cell_strain::Float64 = 0.0125
    relax_cell_damping::Float64 = 1.0
    external_pressure::Vector{Float64} = [0.0]
    external_pressure_gpa::Union{Nothing,Float64} = nothing
    symmetry::String = "auto"
    symmetry_tolerance::Float64 = 1e-5
    # MD phase
    md_steps::Int = 50
    md_temperature::Float64 = 1000.0
    md_dt::Float64 = 1.0
    md_integrator::String = "langevin"
    md_friction::Float64 = 0.01
    md_taut::Float64 = 50.0
    md_seed::Union{Int,Nothing} = nothing
    # Output
    verbose::Bool = false
end

"""
    HotAIRSSResult

Result from a hot-AIRSS run.

# Fields
- `energy_history::Vector{Float64}` — potential energy at each phase boundary
  (relax, md, relax, md, ..., final-relax). Length = `2 * cycles + 1`.
- `relax_iterations::Vector{Int}` — iterations used in each relaxation phase.
  Length = `cycles + 1` (one per cycle + final).
- `cycles_completed::Int` — number of cycles actually executed.
- `converged::Bool` — whether the final relaxation converged.
- `final_energy::Float64` — final potential energy in eV.
- `final_fmax::Float64` — final max atomic force in eV/Å.
- `final_smax::Float64` — final max stress in GPa (0 if fixed cell).
"""
struct HotAIRSSResult
    energy_history::Vector{Float64}
    relax_iterations::Vector{Int}
    cycles_completed::Int
    converged::Bool
    final_energy::Float64
    final_fmax::Float64
    final_smax::Float64
end

# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

"""Create the MD integrator from options."""
function _make_integrator(opts::HotAIRSSOption)
    if opts.md_integrator == "langevin"
        Langevin(;
            friction=opts.md_friction,
            temperature=opts.md_temperature,
            seed=opts.md_seed,
        )
    elseif opts.md_integrator == "berendsen"
        BerendsenNVT(; temperature=opts.md_temperature, taut=opts.md_taut)
    elseif opts.md_integrator == "nve"
        VelocityVerlet()
    else
        throw(ArgumentError("Unknown MD integrator: $(opts.md_integrator)"))
    end
end

function _make_relax_calc(calc::AbstractCalc, opts::HotAIRSSOption)
    calc
end

"""Extract the source calculator for MD (unwraps MetaCalculator if needed)."""
function _raw_calc(calc::MetaCalculator)
    calc.source
end
_raw_calc(calc::AbstractCalc) = calc

_hot_relax_s_tol_internal(opts::HotAIRSSOption) =
    opts.relax_s_tol_gpa >= 0 ? GPaToeVAng(opts.relax_s_tol_gpa) : opts.relax_s_tol

function _hot_external_pressure_internal(opts::HotAIRSSOption)
    if opts.external_pressure_gpa !== nothing
        if opts.external_pressure != [0.0]
            throw(ArgumentError("Set only one of HotAIRSSOption.external_pressure_gpa or external_pressure. external_pressure uses internal eV/Å³ units."))
        end
        return Float64[GPaToeVAng(opts.external_pressure_gpa)]
    end
    if opts.external_pressure != [0.0]
        Base.depwarn("HotAIRSSOption.external_pressure is deprecated because it uses internal eV/Å³ units. Use external_pressure_gpa for hydrostatic pressure in GPa; positive values are compressive and negative values are tensile.", :external_pressure)
    end
    opts.external_pressure
end

_hot_external_stress_matrix(opts::HotAIRSSOption) =
    external_stress_matrix(_hot_external_pressure_internal(opts))

"""Run a single relaxation phase. Returns `(converged, iterations_used)`."""
function _run_relax!(relax_calc::AbstractCalc, opts::HotAIRSSOption)
    iter_count = Ref(0)
    cb = () -> (iter_count[] += 1)
    relax_opts = RelaxOption(;
        symmetry=opts.symmetry,
        symmetry_tolerance=opts.symmetry_tolerance,
        step_constraints=Any[],
    )
    step_constraints = Simulation._resolved_step_constraints(relax_calc, relax_opts)

    if opts.relax_method == "tpsd"
        opt = opt_tpsd(
            relax_calc;
            f_tol=opts.relax_f_tol,
            s_tol=_hot_relax_s_tol_internal(opts),
            e_tol=opts.relax_e_tol,
            convergence=opts.relax_convergence,
            itermax=opts.relax_itermax,
            trace=opts.verbose ? 1 : 0,
            callback=cb,
            relax_cell=opts.relax_cell,
            ext_pressure=_hot_external_stress_matrix(opts),
            max_cell_strain=opts.relax_max_cell_strain,
            cell_damping=opts.relax_cell_damping,
            step_constraints,
        )
        conv = converged(opt, relax_calc)
    elseif opts.relax_method == "fire"
        fire_opt = opt_fire(
            relax_calc;
            f_tol=opts.relax_f_tol,
            s_tol=_hot_relax_s_tol_internal(opts),
            e_tol=opts.relax_e_tol,
            convergence=opts.relax_convergence,
            itermax=opts.relax_itermax,
            verbose=opts.verbose,
            callback=cb,
            relax_cell=opts.relax_cell,
            ext_pressure=_hot_external_stress_matrix(opts),
            max_cell_strain=opts.relax_max_cell_strain,
            cell_damping=opts.relax_cell_damping,
            step_constraints,
        )
        conv = converged(fire_opt, relax_calc)
    else
        throw(ArgumentError("Unknown relax method: $(opts.relax_method)"))
    end

    conv, iter_count[]
end

# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

"""
    hot_airss!(calc::AbstractCalc; callback=nothing, kwargs...) -> HotAIRSSResult
    hot_airss!(calc::AbstractCalc, opts::HotAIRSSOption; callback=nothing) -> HotAIRSSResult

Run the hot-AIRSS protocol: alternating relaxation and MD.

Each cycle relaxes the structure for `relax_itermax` steps, then runs `md_steps`
of MD at `md_temperature`. A final relaxation is performed at the end.

The energy history records potential energy after each phase:
  [relax₁, md₁, relax₂, md₂, ..., relax_final]

# Arguments
- `calc` — calculator (typically `EDDPCalculator`). Cell relaxation is handled
  by passing `relax_cell=true` to the optimizer.

# Keyword arguments
See [`HotAIRSSOption`](@ref) for all options.

- `callback` — optional function called at each phase boundary with a NamedTuple:
  `(; phase::Symbol, cycle::Int, energy::Float64, step::Int)`.
  `phase` is `:relax`, `:md`, or `:final`.

# Example
```julia
calc = EDDPCalculator(cell, cf, nn)
result = hot_airss!(calc; cycles=5, md_temperature=800.0, md_steps=100)
println("Final energy: ", result.final_energy)
```
"""
function hot_airss!(calc::AbstractCalc; callback=nothing, kwargs...)
    opts = HotAIRSSOption(; kwargs...)
    hot_airss!(calc, opts; callback)
end

function hot_airss!(calc::AbstractCalc, opts::HotAIRSSOption; callback=nothing)
    # The raw calc is used for MD (fixed cell)
    md_calc = _raw_calc(calc)
    # Wrap for variable cell relaxation
    relax_calc = _make_relax_calc(calc, opts)
    # Create the integrator (reused across cycles)
    integrator = _make_integrator(opts)

    energy_history = Float64[]
    relax_iterations = Int[]

    for cycle in 1:opts.cycles
        # ── Relaxation phase ──
        conv, niter = _run_relax!(relax_calc, opts)

        e_relax = eddp_energy(relax_calc)
        push!(energy_history, e_relax)
        push!(relax_iterations, niter)

        if !isnothing(callback)
            callback((; phase=:relax, cycle, energy=e_relax, step=sum(relax_iterations)))
        end

        if opts.verbose
            @info @sprintf(
                "Cycle %d/%d  Relax  E=%12.6f  iter=%4d  conv=%s",
                cycle, opts.cycles, e_relax, niter, conv
            )
        end

        # ── MD phase ──
        runner = MDRunner(md_calc, integrator, opts.md_dt;
            temperature=opts.md_temperature,
        )
        run!(runner, opts.md_steps; trace=opts.verbose)

        e_md = eddp_energy(md_calc)
        push!(energy_history, e_md)

        if !isnothing(callback)
            total_steps = sum(relax_iterations) + cycle * opts.md_steps
            callback((; phase=:md, cycle, energy=e_md, step=total_steps))
        end

        if opts.verbose
            @info @sprintf(
                "Cycle %d/%d  MD     E=%12.6f  T=%8.1f K",
                cycle, opts.cycles, e_md, runner.state.temperature
            )
        end
    end

    # ── Final relaxation ──
    final_conv, niter = _run_relax!(relax_calc, opts)

    push!(relax_iterations, niter)
    e_final = eddp_energy(relax_calc)
    push!(energy_history, e_final)

    if !isnothing(callback)
        callback((; phase=:final, cycle=opts.cycles + 1, energy=e_final,
                   step=sum(relax_iterations)))
    end

    if opts.verbose
        @info @sprintf(
            "Final relax  E=%12.6f  iter=%4d  conv=%s",
            e_final, niter, final_conv
        )
    end

    # ── Final diagnostics ──
    forces = eddp_forces(md_calc)
    fmax = maximum(norm.(eachcol(forces)))
    smax = 0.0
    if opts.relax_cell
        stress = eddp_stress(md_calc) .- _hot_external_stress_matrix(opts)
        smax = maximum(abs.(eVAngToGPa.(stress)))
    end

    HotAIRSSResult(
        energy_history,
        relax_iterations,
        opts.cycles,
        final_conv,
        e_final,
        fmax,
        smax,
    )
end
