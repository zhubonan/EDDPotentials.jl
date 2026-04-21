#=
Tests for the molecular dynamics module
=#
using EDDPotentials
using CellBase
using Test
using Random
using LinearAlgebra

# ─── Helper: simple mock calculator for unit tests ───

"""
Harmonic oscillator calculator: E = Σ_i 0.5 * k * |r_i - r0_i|²
Forces: F = -k * (r - r0)
"""
struct HarmonicCalc{T} <: EDDPotentials.AbstractCalc
    cell::Cell{T}
    k::T
    r0::Matrix{T}
end

HarmonicCalc(cell::Cell; k=1.0) = HarmonicCalc(cell, Float64(k), copy(positions(cell)))

EDDPotentials.get_cell(hc::HarmonicCalc) = hc.cell

function EDDPotentials.eddp_energy(hc::HarmonicCalc; kwargs...)
    pos = positions(hc.cell)
    0.5 * hc.k * sum(abs2, pos .- hc.r0)
end

function EDDPotentials.eddp_forces(hc::HarmonicCalc; kwargs...)
    -hc.k .* (positions(hc.cell) .- hc.r0)
end

function EDDPotentials.eddp_stress(hc::HarmonicCalc; kwargs...)
    zeros(3, 3)
end

function EDDPotentials.calculate!(hc::HarmonicCalc; kwargs...)
    hc
end

# ─── Velocity utilities ───

@testset "MD Velocity Utilities" begin
    m = [1.0, 2.0, 3.0]
    T_target = 300.0

    @testset "Maxwell-Boltzmann velocities" begin
        rng = MersenneTwister(42)
        vel = EDDPotentials.maxwell_boltzmann_velocities(m, T_target; rng)
        @test size(vel) == (3, 3)

        # Check temperature is approximately correct
        T_meas = EDDPotentials.compute_temperature(vel, m)
        # With only 3 atoms the variance is high, so use loose tolerance
        @test abs(T_meas - T_target) / T_target < 0.8

        # After rescaling, should be exact
        EDDPotentials.rescale_velocities!(vel, m, T_target)
        T_rescaled = EDDPotentials.compute_temperature(vel, m)
        @test abs(T_rescaled - T_target) < 1e-6
    end

    @testset "Zero COM velocity" begin
        vel = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
        EDDPotentials.zero_com_velocity!(vel, m)
        # COM velocity should be zero
        total_mass = sum(m)
        for d in 1:3
            v_com = dot(view(vel, d, :), m) / total_mass
            @test abs(v_com) < 1e-12
        end
    end

    @testset "Kinetic energy" begin
        vel = [1.0 0.0; 0.0 1.0; 0.0 0.0]
        m = [2.0, 3.0]
        ke = EDDPotentials.compute_kinetic_energy(vel, m)
        # KE = 0.5 * 2 * 1 + 0.5 * 3 * 1 = 2.5
        @test ke ≈ 2.5
    end

    @testset "Temperature" begin
        vel = [1.0 0.0; 0.0 1.0; 0.0 0.0]
        m = [1.0, 1.0]
        T = EDDPotentials.compute_temperature(vel, m)
        # KE = 0.5*(1+1) = 1.0, T = 2*KE/(3*2*KB)
        @test T ≈ 2 * 1.0 / (3 * 2 * EDDPotentials.KB)
    end
end

# ─── NVE energy conservation ───

@testset "MD NVE (Velocity Verlet)" begin
    @testset "Energy conservation with harmonic potential" begin
        # Two atoms in a harmonic well (need ≥2 for COM zeroing to not kill all velocity)
        pos = Float64[0.5 1.5; 0.5 1.5; 0.5 1.5]
        cell = Cell(Lattice(20.0, 20.0, 20.0), [:H, :H], pos)
        hc = HarmonicCalc(cell; k=0.5)

        # Use low temperature for tighter energy conservation test
        dt_fs = 0.5  # 0.5 fs timestep
        runner = MDRunner(hc, VelocityVerlet(), dt_fs; temperature=50.0)

        e_initial = total_energy(hc, runner.state)
        e_values = [e_initial]

        for _ in 1:200
            step!(runner)
            push!(e_values, total_energy(hc, runner.state))
        end

        # Energy should be conserved to high precision
        e_max = maximum(e_values)
        e_min = minimum(e_values)
        de_rel = (e_max - e_min) / abs(e_initial)
        @test de_rel < 1e-3
    end

    @testset "Multi-atom NVE" begin
        pos = Float64[0.0 2.0 1.0; 0.0 2.0 1.0; 0.0 2.0 1.0]
        cell = Cell(Lattice(10.0, 10.0, 10.0), [:H, :H, :He], pos)
        hc = HarmonicCalc(cell; k=0.5)

        runner = MDRunner(hc, VelocityVerlet(), 0.5; temperature=500.0)
        e_initial = total_energy(hc, runner.state)

        run!(runner, 100)

        e_final = total_energy(hc, runner.state)
        @test abs(e_final - e_initial) / abs(e_initial) < 1e-4
    end
end

# ─── NVT Berendsen ───

@testset "MD NVT Berendsen" begin
    T_target = 500.0
    pos = Float64[0.0 1.0 2.0 3.0; 0.0 1.0 2.0 3.0; 0.0 1.0 2.0 3.0]
    cell = Cell(Lattice(15.0, 15.0, 15.0), [:H, :H, :He, :He], pos)
    hc = HarmonicCalc(cell; k=1.0)

    bt = BerendsenNVT(; temperature=T_target, taut=50.0)
    runner = MDRunner(hc, bt, 0.5; temperature=100.0)

    temps = Float64[]
    for _ in 1:1000
        step!(runner)
        push!(temps, runner.state.temperature)
    end

    # Average temperature over second half should approach target
    avg_T = sum(temps[501:end]) / length(temps[501:end])
    @test abs(avg_T - T_target) / T_target < 0.3
end

# ─── NVT Langevin ───

@testset "MD NVT Langevin" begin
    T_target = 400.0
    pos = Float64[0.0 1.0 2.0 3.0; 0.0 1.0 2.0 3.0; 0.0 1.0 2.0 3.0]
    cell = Cell(Lattice(15.0, 15.0, 15.0), [:H, :H, :He, :He], pos)
    hc = HarmonicCalc(cell; k=1.0)

    lg = Langevin(; friction=0.01, temperature=T_target, seed=123)
    runner = MDRunner(hc, lg, 0.5; temperature=100.0)

    temps = Float64[]
    for _ in 1:1000
        step!(runner)
        push!(temps, runner.state.temperature)
    end

    # Average temperature over second half should approach target
    avg_T = sum(temps[501:end]) / length(temps[501:end])
    @test abs(avg_T - T_target) / T_target < 0.4
end

# ─── Trajectory and IO ───

@testset "MD Trajectory" begin
    @testset "In-memory trajectory" begin
        pos = Float64[0.5 1.5; 0.5 1.5; 0.5 1.5]
        cell = Cell(Lattice(20.0, 20.0, 20.0), [:H, :H], pos)
        hc = HarmonicCalc(cell; k=1.0)

        runner = MDRunner(hc, VelocityVerlet(), 0.5; temperature=300.0, trajectory_interval=10)
        run!(runner, 50)

        @test length(runner.trajectory) == 5
        # Positions should differ between frames
        if length(runner.trajectory) >= 2
            @test !all(runner.trajectory[1].positions .≈ runner.trajectory[end].positions)
        end
    end

    @testset "Extxyz writer" begin
        pos = Float64[0.0 1.0; 0.0 1.0; 0.0 1.0]
        cell = Cell(Lattice(10.0, 10.0, 10.0), [:H, :He], pos)
        hc = HarmonicCalc(cell; k=1.0)

        runner = MDRunner(hc, VelocityVerlet(), 0.5; temperature=300.0)
        run!(runner, 5)

        tmpfile = tempname() * ".extxyz"
        try
            save_frame(tmpfile, runner)
            save_frame(tmpfile, runner)

            # Read back and check
            lines = readlines(tmpfile)
            @test length(lines) == 8  # 2 frames × (1 natoms + 1 comment + 2 atom) lines
            @test strip(lines[1]) == "2"  # natoms
            @test strip(lines[5]) == "2"  # second frame natoms
        finally
            rm(tmpfile; force=true)
        end
    end
end

# ─── Integration with NNCalc ───

@testset "MD with NNCalc" begin
    using Flux
    isdefined(Main, :_h2_cell) || include("utils.jl")
    calc = _get_lco_calc()

    @testset "NVE short run" begin
        runner = MDRunner(calc, VelocityVerlet(), 0.5; temperature=300.0)
        e_initial = total_energy(calc, runner.state)

        run!(runner, 20)

        e_final = total_energy(calc, runner.state)
        # Allow some tolerance for NN-based calculator
        de_rel = abs(e_final - e_initial) / abs(e_initial)
        @test de_rel < 0.05
    end

    @testset "NVT Langevin short run" begin
        lg = Langevin(; friction=0.005, temperature=300.0, seed=42)
        runner = MDRunner(calc, lg, 0.5; temperature=300.0)
        run!(runner, 20)

        # Should not crash and temperature should be finite
        @test isfinite(runner.state.temperature)
        @test runner.state.step == 20
    end
end

# ─── Callback mechanism ───

@testset "MD Callback" begin
    pos = Float64[0.5 1.5; 0.5 1.5; 0.5 1.5]
    cell = Cell(Lattice(20.0, 20.0, 20.0), [:H, :H], pos)
    hc = HarmonicCalc(cell; k=1.0)

    step_log = Int[]
    runner = MDRunner(
        hc, VelocityVerlet(), 0.5;
        temperature=300.0,
        callback=r -> push!(step_log, r.state.step),
    )

    run!(runner, 10)
    @test step_log == collect(1:10)
end
