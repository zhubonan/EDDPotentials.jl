#=
Molecular dynamics simulation module

Implements NVE (Velocity Verlet) and NVT (Langevin, Berendsen) integrators
for running MD simulations with EDDP calculators.

Unit system: eV, Å, AMU (atomic mass units).
Internal time unit: √(AMU·Å²/eV) ≈ 10.18 fs.
=#

using Random
using LinearAlgebra
using Printf
using Unitful: ustrip
using ..Potential: get_cell, eddp_energy, eddp_forces, eddp_stress

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

# Boltzmann constant in eV/K
const KB = 8.617333262e-5

# Internal time unit: √(AMU·Å²/eV) ≈ 10.18 fs
const TIME_UNIT_FS = 10.17786336

# ─────────────────────────────────────────────────────────────
# Integrator types
# ─────────────────────────────────────────────────────────────

"""Abstract type for MD integrators."""
abstract type MDIntegrator end

"""NVE ensemble using Velocity Verlet algorithm."""
struct VelocityVerlet <: MDIntegrator end

"""
    Langevin{T} <: MDIntegrator

NVT ensemble using Langevin dynamics (Vanden-Eijnden & Ciccotti propagator).

Fields:
- `friction::T` — friction coefficient γ in inverse internal time units
- `temperature::T` — target temperature in Kelvin
- `rng::Random.MersenneTwister` — random number generator
"""
struct Langevin{T} <: MDIntegrator
    friction::T
    temperature::T
    rng::Random.MersenneTwister
end

"""
    Langevin(; friction, temperature, seed=nothing)

Create a Langevin integrator with given friction coefficient (in 1/fs) and
target temperature (in K).
"""
function Langevin(; friction, temperature, seed=nothing)
    T = float(promote_type(typeof(friction), typeof(temperature)))
    rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)
    # Convert friction from 1/fs to 1/time_unit
    Langevin{T}(friction * TIME_UNIT_FS, temperature, rng)
end

"""
    BerendsenNVT{T} <: MDIntegrator

NVT ensemble using Berendsen velocity rescaling thermostat.

Fields:
- `temperature::T` — target temperature in Kelvin
- `taut::T` — coupling time constant in internal time units
"""
struct BerendsenNVT{T} <: MDIntegrator
    temperature::T
    taut::T
end

"""
    BerendsenNVT(; temperature, taut)

Create a Berendsen NVT thermostat. `taut` is the coupling time in fs.
"""
function BerendsenNVT(; temperature, taut)
    T = float(promote_type(typeof(temperature), typeof(taut)))
    BerendsenNVT{T}(temperature, taut / TIME_UNIT_FS)
end

# ─────────────────────────────────────────────────────────────
# MD State
# ─────────────────────────────────────────────────────────────

"""
    mutable struct MDState{T}

Mutable state for an MD simulation: velocities, cached forces,
thermodynamic observables.
"""
mutable struct MDState{T}
    velocities::Matrix{T}   # 3×N velocity matrix (Å / time_unit)
    masses::Vector{T}       # N atomic masses (AMU)
    forces::Matrix{T}       # 3×N cached forces (eV/Å)
    kinetic_energy::T       # Current kinetic energy (eV)
    temperature::T          # Current instantaneous temperature (K)
    step::Int               # Step counter
    time::T                 # Elapsed simulation time (time_units)
    dt::T                   # Timestep (time_units)
end

# ─────────────────────────────────────────────────────────────
# MD Runner
# ─────────────────────────────────────────────────────────────

"""
    struct MDRunner

Top-level object for running MD simulations.

Usage:
```julia
calc = EDDPCalculator(cell, cf, nn)
runner = MDRunner(calc, VelocityVerlet(), 1.0; temperature=300.0)
run!(runner, 1000; trace=true)
```
"""
struct MDRunner{C<:AbstractCalc,I<:MDIntegrator,T}
    calc::C
    integrator::I
    state::MDState{T}
    trajectory::Vector{Cell}
    traj_velocities::Vector{Matrix{T}}   # Per-frame velocities
    traj_forces::Vector{Matrix{T}}       # Per-frame forces
    callback::Any   # Union{Nothing, Function}
    trajectory_interval::Int
    wrap_atoms::Bool
end

# ─────────────────────────────────────────────────────────────
# Velocity utilities
# ─────────────────────────────────────────────────────────────

"""
    maxwell_boltzmann_velocities(masses, temperature; rng=Random.default_rng())

Generate velocities from a Maxwell-Boltzmann distribution at the given temperature.

Returns a 3×N matrix. Each component of atom i is drawn from N(0, kB*T/m_i).
"""
function maxwell_boltzmann_velocities(
    masses::AbstractVector, temperature::Real; rng=Random.default_rng()
)
    N = length(masses)
    vel = Matrix{float(eltype(masses))}(undef, 3, N)
    for i in 1:N
        σ = sqrt(KB * temperature / masses[i])
        vel[1, i] = σ * randn(rng)
        vel[2, i] = σ * randn(rng)
        vel[3, i] = σ * randn(rng)
    end
    vel
end

"""
    zero_com_velocity!(velocities, masses)

Remove centre-of-mass velocity so the system has no net drift.
"""
function zero_com_velocity!(velocities::AbstractMatrix, masses::AbstractVector)
    total_mass = sum(masses)
    for d in 1:3
        v_com = dot(view(velocities, d, :), masses) / total_mass
        for i in axes(velocities, 2)
            velocities[d, i] -= v_com
        end
    end
    velocities
end

"""
    rescale_velocities!(velocities, masses, target_temp)

Scale velocities so the instantaneous temperature equals `target_temp`.
"""
function rescale_velocities!(
    velocities::AbstractMatrix, masses::AbstractVector, target_temp::Real
)
    T_curr = compute_temperature(velocities, masses)
    if T_curr > 0
        velocities .*= sqrt(target_temp / T_curr)
    end
    velocities
end

"""
    compute_kinetic_energy(velocities, masses) -> Float64

Compute kinetic energy: E_kin = 0.5 * Σ_i m_i * |v_i|²
"""
function compute_kinetic_energy(velocities::AbstractMatrix, masses::AbstractVector)
    ke = zero(eltype(velocities))
    for i in axes(velocities, 2)
        m = masses[i]
        ke += m * (velocities[1, i]^2 + velocities[2, i]^2 + velocities[3, i]^2)
    end
    0.5 * ke
end

"""
    compute_temperature(velocities, masses) -> Float64

Compute instantaneous temperature from equipartition:
T = 2 * E_kin / (3 * N * kB)
"""
function compute_temperature(velocities::AbstractMatrix, masses::AbstractVector)
    N = size(velocities, 2)
    ke = compute_kinetic_energy(velocities, masses)
    2 * ke / (3 * N * KB)
end

# ─────────────────────────────────────────────────────────────
# MDState constructors
# ─────────────────────────────────────────────────────────────

"""
    MDState(calc::AbstractCalc, dt_fs::Real;
            temperature=nothing, velocities=nothing)

Create MDState from a calculator. If `temperature` is given, initialises velocities
from a Maxwell-Boltzmann distribution. Alternatively, pass an explicit 3×N velocity
matrix. If neither is given, velocities default to zero.
"""
function MDState(
    calc::AbstractCalc,
    dt_fs::Real;
    temperature=nothing,
    velocities=nothing,
)
    cell = get_cell(calc)
    m = Float64.(ustrip.(masses(cell)))
    N = length(m)
    dt = dt_fs / TIME_UNIT_FS

    if !isnothing(velocities)
        vel = Float64.(velocities)
    elseif !isnothing(temperature)
        vel = maxwell_boltzmann_velocities(m, Float64(temperature))
        zero_com_velocity!(vel, m)
        rescale_velocities!(vel, m, Float64(temperature))
    else
        vel = zeros(Float64, 3, N)
    end

    # Get initial forces
    forces = eddp_forces(calc)
    ke = compute_kinetic_energy(vel, m)
    T = compute_temperature(vel, m)

    MDState{Float64}(vel, m, forces, ke, T, 0, 0.0, dt)
end

# ─────────────────────────────────────────────────────────────
# MDRunner constructors
# ─────────────────────────────────────────────────────────────

"""
    MDRunner(calc::AbstractCalc, integrator::MDIntegrator, dt_fs::Real;
             temperature=nothing, velocities=nothing,
             callback=nothing, trajectory_interval=1, wrap_atoms=true)

Create an MD simulation runner.

Arguments:
- `calc` — calculator (e.g. EDDPCalculator)
- `integrator` — MDIntegrator (VelocityVerlet(), Langevin(...), BerendsenNVT(...))
- `dt_fs` — timestep in femtoseconds

Keyword arguments:
- `temperature` — initial temperature for Maxwell-Boltzmann velocities (K)
- `velocities` — explicit initial velocities (3×N matrix, Å/time_unit)
- `callback` — function called each step: `callback(runner)`
- `trajectory_interval` — save a frame every N steps
- `wrap_atoms` — wrap atoms into the unit cell after each step
"""
function MDRunner(
    calc::AbstractCalc,
    integrator::MDIntegrator,
    dt_fs::Real;
    temperature=nothing,
    velocities=nothing,
    callback=nothing,
    trajectory_interval=1,
    wrap_atoms=true,
)
    state = MDState(calc, dt_fs; temperature, velocities)
    trajectory = Cell[]
    traj_velocities = Matrix{Float64}[]
    traj_forces = Matrix{Float64}[]
    MDRunner(calc, integrator, state, trajectory, traj_velocities, traj_forces, callback, trajectory_interval, wrap_atoms)
end

# ─────────────────────────────────────────────────────────────
# Observable helpers
# ─────────────────────────────────────────────────────────────

"""Return kinetic energy in eV."""
kinetic_energy(state::MDState) = state.kinetic_energy

"""Return instantaneous temperature in K."""
temperature(state::MDState) = state.temperature

"""Return total energy (potential + kinetic) in eV."""
function total_energy(calc::AbstractCalc, state::MDState)
    eddp_energy(calc) + state.kinetic_energy
end

"""Return instantaneous pressure in GPa (positive = compressive)."""
function instantaneous_pressure(calc::AbstractCalc, state::MDState)
    stress = eddp_stress(calc)
    # Use the same sign convention as get_pressure_gpa: tr(stress)/3
    # For the full virial pressure including kinetic contribution:
    # P_virial = (tr(stress) + 2*E_kin/V) / 3
    # where stress here includes the virial contribution from forces
    eVAngToGPa(tr(stress) / 3)
end

"""Return elapsed simulation time in femtoseconds."""
time_fs(state::MDState) = state.time * TIME_UNIT_FS

"""Return elapsed simulation time in picoseconds."""
time_ps(state::MDState) = state.time * TIME_UNIT_FS / 1000.0

# ─────────────────────────────────────────────────────────────
# Internal: update state observables
# ─────────────────────────────────────────────────────────────

function _update_observables!(state::MDState)
    state.kinetic_energy = compute_kinetic_energy(state.velocities, state.masses)
    state.temperature = compute_temperature(state.velocities, state.masses)
    state
end

# ─────────────────────────────────────────────────────────────
# Velocity Verlet step (NVE)
# ─────────────────────────────────────────────────────────────

function _step!(runner::MDRunner{C,<:VelocityVerlet}) where {C}
    state = runner.state
    calc = runner.calc
    dt = state.dt
    vel = state.velocities
    forces = state.forces
    m = state.masses

    # Position update: r += dt * v + 0.5 * dt^2 * F/m
    pos = get_positions(calc)
    for i in axes(pos, 2)
        mi = m[i]
        for d in 1:3
            pos[d, i] += dt * vel[d, i] + 0.5 * dt^2 * forces[d, i] / mi
        end
    end

    # Wrap atoms into cell
    runner.wrap_atoms && wrap!(get_cell(calc))

    set_positions!(calc, pos)

    # Get new forces
    new_forces = eddp_forces(calc)

    # Velocity update: v += 0.5 * dt * (F_old + F_new) / m
    for i in axes(vel, 2)
        mi = m[i]
        for d in 1:3
            vel[d, i] += 0.5 * dt * (forces[d, i] + new_forces[d, i]) / mi
        end
    end

    # Update cached forces and observables
    state.forces .= new_forces
    state.step += 1
    state.time += dt
    _update_observables!(state)
end

# ─────────────────────────────────────────────────────────────
# Langevin step (NVT)
# Reference: Vanden-Eijnden & Ciccotti, Chem. Phys. Lett. 429, 310 (2006)
#            ASE implementation ase/md/langevin.py
# ─────────────────────────────────────────────────────────────

# Precomputed Langevin coefficients
struct _LangevinCoeffs{T}
    c1::T            # dt/2 - dt²γ/8  (force drift, mass-independent)
    c2::T            # dtγ/2 - dt²γ²/8 (friction)
    c3::Vector{T}    # √dt·σ/2 - dt^(3/2)·γ·σ/8  (noise velocity, per atom)
    c5::Vector{T}    # dt^(3/2)·σ/(2√3)  (noise position, per atom)
    c4::T            # γ/2 · c5
end

function _langevin_coeffs(lg::Langevin{T}, dt::T, masses::AbstractVector) where {T}
    γ = lg.friction
    kB_T = KB * lg.temperature
    N = length(masses)

    sqrt_dt = sqrt(dt)
    dt15 = dt * sqrt_dt

    # c1 is mass-independent (same for all atoms)
    c1 = dt / 2 - dt^2 * γ / 8
    c2 = dt * γ / 2 - dt^2 * γ^2 / 8

    c3 = Vector{T}(undef, N)
    c5 = Vector{T}(undef, N)

    for i in 1:N
        mi = masses[i]
        σ = sqrt(2 * kB_T * γ / mi)
        c3[i] = sqrt_dt * σ / 2 - dt15 * γ * σ / 8
        c5[i] = dt15 * σ / (2 * sqrt(3))
    end

    c4 = γ / 2   # multiplied by c5 per-atom later

    _LangevinCoeffs{T}(c1, c2, c3, c5, c4)
end

function _step!(runner::MDRunner{C,<:Langevin}) where {C}
    state = runner.state
    calc = runner.calc
    lg = runner.integrator
    dt = state.dt
    vel = state.velocities
    forces = state.forces
    m = state.masses
    N = length(m)

    coeffs = _langevin_coeffs(lg, dt, m)

    # Generate noise
    xi = randn(lg.rng, 3, N)
    eta = randn(lg.rng, 3, N)

    # Random contributions (with COM correction)
    rnd_pos = Matrix{Float64}(undef, 3, N)
    rnd_vel = Matrix{Float64}(undef, 3, N)
    for i in 1:N
        for d in 1:3
            rnd_pos[d, i] = coeffs.c5[i] * eta[d, i]
            rnd_vel[d, i] = coeffs.c3[i] * xi[d, i] - coeffs.c4 * coeffs.c5[i] * eta[d, i]
        end
    end

    # Remove COM from random contributions
    for d in 1:3
        rnd_pos_avg = sum(view(rnd_pos, d, :)) / N
        rnd_vel_avg = sum(rnd_vel[d, i] * m[i] for i in 1:N) / (sum(m))
        for i in 1:N
            rnd_pos[d, i] -= rnd_pos_avg
            rnd_vel[d, i] -= rnd_vel_avg
        end
    end

    # First half-step in velocity
    for i in 1:N
        for d in 1:3
            vel[d, i] += coeffs.c1 * forces[d, i] / m[i] - coeffs.c2 * vel[d, i] + rnd_vel[d, i]
        end
    end

    # Full step in positions
    pos = get_positions(calc)
    for i in 1:N
        for d in 1:3
            pos[d, i] += dt * vel[d, i] + rnd_pos[d, i]
        end
    end

    runner.wrap_atoms && wrap!(get_cell(calc))
    set_positions!(calc, pos)

    # New forces
    new_forces = eddp_forces(calc)

    # Second half-step in velocity
    for i in 1:N
        for d in 1:3
            vel[d, i] += coeffs.c1 * new_forces[d, i] / m[i] - coeffs.c2 * vel[d, i] + rnd_vel[d, i]
        end
    end

    state.forces .= new_forces
    state.step += 1
    state.time += dt
    _update_observables!(state)
end

# ─────────────────────────────────────────────────────────────
# Berendsen NVT step
# ─────────────────────────────────────────────────────────────

function _step!(runner::MDRunner{C,<:BerendsenNVT}) where {C}
    state = runner.state
    bt = runner.integrator

    # Apply Berendsen velocity rescaling
    _update_observables!(state)
    T_curr = state.temperature
    if T_curr > 0
        λ = sqrt(1.0 + (bt.temperature / T_curr - 1.0) * state.dt / bt.taut)
        # Clamp for stability
        λ = clamp(λ, 0.9, 1.1)
        state.velocities .*= λ
    end

    # Standard Velocity Verlet step
    calc = runner.calc
    dt = state.dt
    vel = state.velocities
    forces = state.forces
    m = state.masses

    # Position update
    pos = get_positions(calc)
    for i in axes(pos, 2)
        mi = m[i]
        for d in 1:3
            pos[d, i] += dt * vel[d, i] + 0.5 * dt^2 * forces[d, i] / mi
        end
    end

    runner.wrap_atoms && wrap!(get_cell(calc))
    set_positions!(calc, pos)

    new_forces = eddp_forces(calc)

    # Velocity update
    for i in axes(vel, 2)
        mi = m[i]
        for d in 1:3
            vel[d, i] += 0.5 * dt * (forces[d, i] + new_forces[d, i]) / mi
        end
    end

    state.forces .= new_forces
    state.step += 1
    state.time += dt
    _update_observables!(state)
end

# ─────────────────────────────────────────────────────────────
# Run loop
# ─────────────────────────────────────────────────────────────

"""
    step!(runner::MDRunner)

Advance the simulation by one timestep.
"""
function step!(runner::MDRunner)
    _step!(runner)

    # Save trajectory frame with per-frame velocities and forces
    if runner.trajectory_interval > 0 && runner.state.step % runner.trajectory_interval == 0
        push!(runner.trajectory, deepcopy(get_cell(runner.calc)))
        push!(runner.traj_velocities, copy(runner.state.velocities))
        push!(runner.traj_forces, copy(runner.state.forces))
    end

    # Callback
    runner.callback isa Function && runner.callback(runner)
end

"""
    run!(runner::MDRunner, nsteps::Int; trace=false)

Run MD simulation for `nsteps` steps.

If `trace=true`, prints step, time (fs), total energy (eV), temperature (K),
and pressure (GPa) at each step.
"""
function run!(runner::MDRunner, nsteps::Int; trace=false)
    if trace
        @info @sprintf(
            "%-6s %10s %12s %12s %12s",
            "Step", "Time(fs)", "E_total(eV)", "Temp(K)", "Press(GPa)"
        )
    end

    for _ in 1:nsteps
        step!(runner)

        if trace
            s = runner.state
            e_tot = total_energy(runner.calc, s)
            press = instantaneous_pressure(runner.calc, s)
            @info @sprintf(
                "%-6d %10.2f %12.6f %12.2f %12.4f",
                s.step, time_fs(s), e_tot, s.temperature, press
            )
        end
    end

    runner
end

# ─────────────────────────────────────────────────────────────
# Trajectory I/O (extxyz format)
# ─────────────────────────────────────────────────────────────

"""
    _write_extxyz_frame!(io, cell, velocities, forces)

Write a single frame in extended XYZ format with per-atom velocity and force columns.
"""
function _write_extxyz_frame!(io::IO, cell::Cell, velocities, forces)
    N = nions(cell)
    mat = cellmat(lattice(cell))

    # Number of atoms
    println(io, N)

    # Comment line: Lattice + Properties
    ax, ay, az = mat[:, 1]
    bx, by, bz = mat[:, 2]
    cx, cy, cz = mat[:, 3]

    # Properties: species:S:1:pos:R:3:vel:R:3:forces:R:3
    println(
        io,
        @sprintf(
            "Lattice=\"%g %g %g %g %g %g %g %g %g\" Properties=species:S:1:pos:R:3:vel:R:3:forces:R:3",
            ax, ay, az, bx, by, bz, cx, cy, cz
        )
    )

    # Atom lines
    sp = species(cell)
    pos = positions(cell)
    for i in 1:N
        println(
            io,
            @sprintf(
                "%s %g %g %g %g %g %g %g %g %g",
                sp[i],
                pos[1, i], pos[2, i], pos[3, i],
                velocities[1, i], velocities[2, i], velocities[3, i],
                forces[1, i], forces[2, i], forces[3, i]
            )
        )
    end
end

"""
    save_trajectory(filename::String, runner::MDRunner)

Save the trajectory to an extxyz file with per-frame velocities and forces.
"""
function save_trajectory(filename::String, runner::MDRunner)
    open(filename, "w") do io
        for (frame, vel, frc) in zip(runner.trajectory, runner.traj_velocities, runner.traj_forces)
            _write_extxyz_frame!(io, frame, vel, frc)
        end
    end
end

"""
    save_frame(filename::String, calc::AbstractCalc, state::MDState)

Append a single frame to an extxyz file with velocities and forces.
"""
function save_frame(filename::String, calc::AbstractCalc, state::MDState)
    open(filename, "a") do io
        _write_extxyz_frame!(io, get_cell(calc), state.velocities, state.forces)
    end
end

"""
    save_frame(filename::String, runner::MDRunner)

Append the current frame to an extxyz file.
"""
function save_frame(filename::String, runner::MDRunner)
    save_frame(filename, runner.calc, runner.state)
end
