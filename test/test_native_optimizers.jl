using Test
using EDDPotentials
using CellBase
using LinearAlgebra
using Unitful: ustrip
import EDDPotentials.Potential: get_cell, eddp_energy, eddp_forces, eddp_stress, calculate!, GPaToeVAng

# Import test utilities
isdefined(Main, :_h2_cell) || include("utils.jl")
const lj_like_calc = EDDPotentials.lj_like_calc

mutable struct StressOnlyCalc <: AbstractCalc
    cell::Cell
    stress::Matrix{Float64}
end

get_cell(calc::StressOnlyCalc) = calc.cell
eddp_energy(calc::StressOnlyCalc; kwargs...) = 0.0
eddp_forces(calc::StressOnlyCalc; kwargs...) = zeros(3, length(calc.cell))
eddp_stress(calc::StressOnlyCalc; kwargs...) = copy(calc.stress)
calculate!(calc::StressOnlyCalc; kwargs...) = calc

mutable struct UphillCellCalc <: AbstractCalc
    cell::Cell
end

get_cell(calc::UphillCellCalc) = calc.cell
eddp_energy(calc::UphillCellCalc; kwargs...) = sum(get_cellmat(calc)[1:3, 1:3])
eddp_forces(calc::UphillCellCalc; kwargs...) = zeros(3, length(calc.cell))
eddp_stress(calc::UphillCellCalc; kwargs...) = Matrix{Float64}(I, 3, 3)
calculate!(calc::UphillCellCalc; kwargs...) = calc

@testset "Native Optimizers" begin

    @testset "Step Constraints" begin
        @testset "ZeroTranslationConstraint" begin
            cell = _h2o_cell(10.0, 1.0)
            calc = lj_like_calc(Cell(Lattice(10, 10, 10), [:H, :H], get_positions(cell)[:, 1:2]); α=0.5, a=2, rc=8.0)
            opt = NativeTPSD(log_level=:minimal)
            dx = [1.0, 2.0, 3.0, -2.0, 1.0, 0.5]

            apply_step_constraints!([ZeroTranslationConstraint()], opt, calc, dx)
            ionic = reshape(dx, 3, 2)
            @test norm(sum(ionic; dims=2)) < 1e-12
        end

        @testset "IonicConstraint" begin
            cell = _h2_cell(10.0, 1.0)
            calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)
            pos0 = copy(get_positions(calc))
            opt = NativeTPSD(f_tol=1e-3, maxstep=0.1, log_level=:minimal, step_constraints=[IonicConstraint([1])])

            step!(opt, calc)
            pos1 = get_positions(calc)
            @test pos1[:, 1] ≈ pos0[:, 1]
            @test norm(pos1[:, 2] - pos0[:, 2]) > 0
        end

        @testset "FixCartesian" begin
            cell = _h2_cell(10.0, 1.0)
            calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)
            opt = NativeTPSD(log_level=:minimal)
            dx = [1.0, 2.0, 3.0, -2.0, 1.0, 0.5]

            apply_step_constraints!([FixCartesian([1]; mask=(true, false, true))], opt, calc, dx)
            ionic = reshape(dx, 3, 2)
            @test ionic[1, 1] == 0.0
            @test ionic[2, 1] == 2.0
            @test ionic[3, 1] == 0.0
            @test ionic[:, 2] ≈ [-2.0, 1.0, 0.5]
        end

        @testset "FixScaled nonorthogonal step and force projection" begin
            cmat = [3.0 0.4 0.1; 0.0 2.5 0.3; 0.0 0.0 4.0]
            cell = Cell(Lattice(cmat), [:H, :H], cmat * [0.1 0.7; 0.2 0.3; 0.3 0.4])
            calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)
            opt = NativeTPSD(log_level=:minimal)
            scaled_step = [0.2, -0.1, 0.3]
            dxmat = zeros(3, 2)
            dxmat[:, 1] .= cmat * scaled_step
            dxmat[:, 2] .= [0.4, -0.2, 0.1]
            dx = vec(dxmat)

            apply_step_constraints!([FixScaled(1; mask=(true, false, true))], opt, calc, dx)
            projected_step = cmat \ reshape(dx, 3, 2)[:, 1]
            @test projected_step[1] ≈ 0.0 atol=1e-12
            @test projected_step[2] ≈ scaled_step[2] atol=1e-12
            @test projected_step[3] ≈ 0.0 atol=1e-12

            F = [1.0 0.2; -0.3 0.7; 0.5 -0.4]
            apply_force_constraints!([FixScaled(1; mask=(true, false, true))], opt, calc, F)
            scaled_force = transpose(cmat) * F[:, 1]
            @test scaled_force[1] ≈ 0.0 atol=1e-12
            @test scaled_force[3] ≈ 0.0 atol=1e-12
        end

        @testset "FixRelative and FixRigidGroup" begin
            cell = Cell(Lattice(10.0, 10.0, 10.0), [:H, :H, :H],
                [0.0 0.2 1.0; 0.0 1.0 1.2; 0.0 0.0 0.4])
            calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)
            opt = NativeTPSD(log_level=:minimal)

            dx = collect(1.0:9.0)
            apply_step_constraints!([FixRelative([(1, 2)]; mask=(true, false, true))], opt, calc, dx)
            ionic = reshape(dx, 3, 3)
            @test ionic[1, 1] - ionic[1, 2] ≈ 0.0 atol=1e-12
            @test ionic[3, 1] - ionic[3, 2] ≈ 0.0 atol=1e-12
            @test !(ionic[2, 1] - ionic[2, 2] ≈ 0.0)

            dx2 = collect(1.0:9.0)
            apply_step_constraints!([FixRigidGroup([1, 2, 3])], opt, calc, dx2)
            ionic2 = reshape(dx2, 3, 3)
            @test ionic2[:, 1] ≈ ionic2[:, 2] atol=1e-12
            @test ionic2[:, 1] ≈ ionic2[:, 3] atol=1e-12
        end

        @testset "FixLinearIonic redundant rows" begin
            cell = _h2_cell(10.0, 1.0)
            calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)
            opt = NativeTPSD(log_level=:minimal)
            A = zeros(2, length(get_positions(calc)))
            A[:, 1] .= 1.0
            dx = ones(length(get_positions(calc)))

            apply_step_constraints!([FixLinearIonic(A)], opt, calc, dx)
            @test dx[1] ≈ 0.0 atol=1e-12
        end

        @testset "FixVolumeConstraint" begin
            cell = _h2_cell(10.0, 1.0)
            calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)
            opt = NativeTPSD(relax_cell=true, log_level=:minimal, step_constraints=[FixVolumeConstraint()])
            dx = zeros(length(get_positions(calc)) + 9)
            dx[(length(get_positions(calc)) + 1):end] .= vec(0.2 .* Matrix(I, 3, 3))
            vol0 = ustrip(volume(get_cell(calc)))

            apply_step_constraints!([FixVolumeConstraint()], opt, calc, dx)
            cell_old = get_cellmat(calc)[1:3, 1:3]
            cell_new = cell_old .+ reshape(dx[(length(get_positions(calc)) + 1):end], 3, 3)
            @test det(cell_new) ≈ vol0 atol=1e-8
        end

        @testset "FixCellConstraint Voigt mask" begin
            cell = _h2_cell(10.0, 1.0)
            calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)
            H = get_cellmat(calc)[1:3, 1:3]
            D, nat = size(get_positions(calc))
            opt = NativeTPSD(relax_cell=true, log_level=:minimal)

            strain = [
                0.10 0.06 0.05
                0.06 0.20 0.04
                0.05 0.04 0.30
            ]
            dx = zeros(D * nat + 9)
            reshape(view(dx, D * nat + 1:length(dx)), 3, 3) .= strain * H
            apply_step_constraints!([FixCellConstraint()], opt, calc, dx)
            @test norm(dx[D * nat + 1:end]) < 1e-12

            dx = zeros(D * nat + 9)
            reshape(view(dx, D * nat + 1:length(dx)), 3, 3) .= strain * H
            apply_step_constraints!([FixCellConstraint(mask=(false, false, true, false, false, false))], opt, calc, dx)
            projected = reshape(dx[D * nat + 1:end], 3, 3) / H
            @test projected[1, 1] ≈ strain[1, 1] atol=1e-12
            @test projected[2, 2] ≈ strain[2, 2] atol=1e-12
            @test projected[3, 3] ≈ 0.0 atol=1e-12
            @test projected[2, 3] ≈ strain[2, 3] atol=1e-12

            dx = zeros(D * nat + 9)
            reshape(view(dx, D * nat + 1:length(dx)), 3, 3) .= strain * H
            apply_step_constraints!([FixCellConstraint(mask=(false, false, false, true, true, true))], opt, calc, dx)
            projected = reshape(dx[D * nat + 1:end], 3, 3) / H
            @test projected[1, 1] ≈ strain[1, 1] atol=1e-12
            @test projected[2, 2] ≈ strain[2, 2] atol=1e-12
            @test projected[3, 3] ≈ strain[3, 3] atol=1e-12
            @test projected[2, 3] ≈ 0.0 atol=1e-12
            @test projected[1, 3] ≈ 0.0 atol=1e-12
            @test projected[1, 2] ≈ 0.0 atol=1e-12
        end

        @testset "FixCellConstraint and FixVolumeConstraint ordering" begin
            cell = _h2_cell(10.0, 1.0)
            calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)
            D, nat = size(get_positions(calc))
            opt = NativeTPSD(relax_cell=true, log_level=:minimal)
            dx = zeros(D * nat + 9)
            dx[D * nat + 1:end] .= vec(0.2 .* Matrix(I, 3, 3))

            apply_step_constraints!([FixCellConstraint(), FixVolumeConstraint()], opt, calc, dx)
            @test norm(dx[D * nat + 1:end]) < 1e-12
        end
    end

    @testset "NativeTPSD" begin
        @testset "Basic Functionality" begin
            # Create a simple H2 cell displaced from equilibrium
            cell = _h2_cell(10.0, 1.0)  # Lattice = 10, displacement factor = 1.0
            calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)

            # Create TPSD optimizer
            opt = NativeTPSD(f_tol=1e-3, itermax=500, log_level=:minimal)

            # Initialize
            initialize!(opt, calc)
            @test opt.iteration == 0
            @test length(opt.x_prev) > 0

            # Run a few steps
            for i in 1:10
                step!(opt, calc)
            end

            @test opt.iteration == 10
            @test opt.α > 0  # Step size should be positive

            # Check forces are decreasing
            F = eddp_forces(calc)
            fmax = maximum(norm.(eachcol(F)))
            @test fmax < 0.1  # Should have made progress
        end

        @testset "Convergence - Fixed Cell" begin
            # Create H2 cell — atoms on attractive side of minimum
            cell = _h2_cell(10.0, 2.5)
            calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)

            # Run optimization
            opt = NativeTPSD(f_tol=1e-3, itermax=500, log_level=:minimal)

            iter = 0
            while !converged(opt, calc) && iter < 500
                step!(opt, calc)
                iter += 1
            end

            # Check convergence
            @test converged(opt, calc)
            @test opt.iteration < 500

            # Check forces small
            F = eddp_forces(calc)
            fmax = maximum(norm.(eachcol(F)))
            @test fmax < 1e-3

            # Check relaxed — atoms should be within cutoff region
            final_cell = get_cell(calc)
            pos = get_positions(final_cell)
            d = norm(pos[:, 1] - pos[:, 2])
            @test d > 4.0 && d < 6.0  # should relax to near rc/2 ≈ 4.0
        end

        @testset "Reset Functionality" begin
            cell = _h2_cell(10.0, 1.0)
            calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)

            opt = NativeTPSD(f_tol=1e-3, log_level=:minimal)

            # Run a few steps
            for i in 1:5
                step!(opt, calc)
            end
            @test opt.iteration == 5

            # Reset
            reset!(opt)
            @test opt.iteration == 0
            @test opt.α == 1.0
        end

        @testset "Converges with fully fixed raw forces" begin
            cell = _h2_cell(10.0, 1.0)
            calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)
            raw_fmax = maximum(norm.(eachcol(eddp_forces(calc))))
            opt = NativeTPSD(f_tol=1e-12, itermax=5, log_level=:minimal, step_constraints=[FixAtoms([1, 2])])

            step!(opt, calc)
            @test raw_fmax > 1e-8
            @test converged(opt, calc)
        end
    end

    @testset "NativeFIRE" begin
        @testset "Basic Functionality" begin
            # Create H2 cell
            cell = _h2_cell(10.0, 1.0)
            calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)

            # Create FIRE optimizer
            opt = NativeFIRE(f_tol=1e-3, maxstep=0.5, log_level=:minimal)

            # Initialize
            initialize!(opt, calc)
            @test opt.iteration == 0
            @test length(opt.vel) > 0
            @test all(opt.vel .== 0)  # Velocity starts at zero

            # Run a few steps
            for i in 1:10
                step!(opt, calc)
            end

            @test opt.iteration == 10
            @test opt.dt > 0  # Timestep should be positive
        end

        @testset "Convergence - Fixed Cell" begin
            # Create H2 cell — atoms on attractive side of minimum
            cell = _h2_cell(10.0, 2.5)
            calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)

            # Run optimization
            opt = NativeFIRE(f_tol=1e-3, maxstep=0.5, itermax=500, log_level=:minimal)

            iter = 0
            while !converged(opt, calc) && iter < 500
                step!(opt, calc)
                iter += 1
            end

            # Check convergence
            @test converged(opt, calc)
            @test opt.iteration < 500

            # Check forces small
            F = eddp_forces(calc)
            fmax = maximum(norm.(eachcol(F)))
            @test fmax < 1e-3

            # Check relaxed — atoms should be within cutoff region
            final_cell = get_cell(calc)
            pos = get_positions(final_cell)
            d = norm(pos[:, 1] - pos[:, 2])
            @test d > 4.0 && d < 6.0  # near rc/2 ≈ 4.0
        end

        @testset "Velocity Reset" begin
            # Test that velocity builds up during optimization
            cell = _h2_cell(10.0, 1.0)
            calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)

            opt = NativeFIRE(f_tol=1e-3, maxstep=0.2, log_level=:minimal)

            # Run until velocity builds up
            for i in 1:10
                step!(opt, calc)
            end

            # Velocity should be non-zero after moving
            vel_norm_before = norm(opt.vel)
            @test vel_norm_before > 0

            # Nsteps should have incremented
            @test opt.Nsteps >= 0
        end

        @testset "Downhill Check" begin
            # Test with downhill energy checking enabled
            cell = _h2_cell(10.0, 1.0)
            calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)

            opt = NativeFIRE(
                f_tol=1e-3,
                maxstep=0.5,
                downhill_check=true,
                log_level=:minimal
            )

            # Run optimization
            iter = 0
            while !converged(opt, calc) && iter < 500
                step!(opt, calc)
                iter += 1
            end

            # Should still converge with downhill check
            @test converged(opt, calc)
            @test opt.iteration < 500
        end

        @testset "Downhill Check Rejects Uphill Trial" begin
            cell = _h2_cell(10.0, 1.0)
            calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)
            opt = NativeFIRE(
                f_tol=1e-8,
                maxstep=0.5,
                itermax=80,
                downhill_check=true,
                log_level=:minimal,
            )

            energies = [EDDPotentials.eddp_energy(calc)]
            for _ = 1:80
                step!(opt, calc)
                push!(energies, EDDPotentials.eddp_energy(calc))
            end

            @test all(diff(energies) .<= 1e-10)
        end

        @testset "Downhill Check Restores Variable Cell Exactly" begin
            cell = _h2_cell(10.0, 1.0)
            calc = UphillCellCalc(cell)
            H0 = copy(get_cellmat(calc))
            pos0 = copy(get_positions(calc))
            opt = NativeFIRE(
                dt=0.2,
                maxstep=0.5,
                maxstrain=0.05,
                relax_cell=true,
                cell_damping=1.0,
                downhill_check=true,
                log_level=:minimal,
            )

            step!(opt, calc)

            @test get_cellmat(calc) ≈ H0 atol=1e-12
            @test get_positions(calc) ≈ pos0 atol=1e-12
            @test norm(opt.vel) == 0.0
            @test opt.dt ≈ 0.1
        end

        @testset "First Step Keeps Initial dt" begin
            cell = _h2_cell(10.0, 1.0)
            calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)
            opt = NativeFIRE(dt=0.1, maxstep=0.5, log_level=:minimal)

            step!(opt, calc)

            @test opt.dt ≈ 0.1
        end

        @testset "Reset Functionality" begin
            cell = _h2_cell(10.0, 1.0)
            calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)

            opt = NativeFIRE(f_tol=1e-3, log_level=:minimal)

            # Run a few steps
            for i in 1:5
                step!(opt, calc)
            end
            @test opt.iteration == 5

            # Reset
            dt_init = opt.dt_init
            reset!(opt)
            @test opt.iteration == 0
            @test opt.dt == dt_init
            @test all(opt.vel .== 0)
            @test opt.Nsteps == 0
        end

        @testset "FixAtoms" begin
            cell = _h2_cell(10.0, 1.0)
            calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)
            pos0 = copy(get_positions(calc))
            opt = NativeFIRE(f_tol=1e-3, maxstep=0.1, log_level=:minimal, step_constraints=[FixAtoms([1])])

            step!(opt, calc)
            pos1 = get_positions(calc)
            @test pos1[:, 1] ≈ pos0[:, 1]
            @test norm(pos1[:, 2] - pos0[:, 2]) > 0
        end
    end

    @testset "Comparison TPSD vs FIRE" begin
        # Test both algorithms converge to same result
        @testset "Same Final Structure" begin
            # Create two identical cells on attractive side of minimum
            cell1 = _h2_cell(10.0, 2.5)
            cell2 = _h2_cell(10.0, 2.5)

            calc1 = lj_like_calc(cell1; α=0.5, a=2, rc=8.0)
            calc2 = lj_like_calc(cell2; α=0.5, a=2, rc=8.0)

            # Optimize with TPSD
            opt_tpsd = NativeTPSD(f_tol=1e-3, itermax=500, log_level=:minimal)
            while !converged(opt_tpsd, calc1) && opt_tpsd.iteration < 500
                step!(opt_tpsd, calc1)
            end

            # Optimize with FIRE
            opt_fire = NativeFIRE(f_tol=1e-3, maxstep=0.5, itermax=500, log_level=:minimal)
            while !converged(opt_fire, calc2) && opt_fire.iteration < 500
                step!(opt_fire, calc2)
            end

            # Both should converge
            @test converged(opt_tpsd, calc1)
            @test converged(opt_fire, calc2)

            # Final interatomic distances should be very close
            d1 = norm(get_positions(get_cell(calc1))[:, 1] - get_positions(get_cell(calc1))[:, 2])
            d2 = norm(get_positions(get_cell(calc2))[:, 1] - get_positions(get_cell(calc2))[:, 2])
            @test abs(d1 - d2) < 1e-4

            println("  TPSD iterations: $(opt_tpsd.iteration)")
            println("  FIRE iterations: $(opt_fire.iteration)")
        end
    end

    @testset "opt_fire function" begin
        @testset "Basic Usage" begin
            cell = _h2_cell(10.0, 1.0)
            calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)

            # Use opt_fire function
            result = opt_fire(calc; f_tol=1e-3, maxstep=0.5, verbose=false)

            # Check result
            @test result isa NativeFIRE
            @test converged(result, calc)

            # Check forces
            F = eddp_forces(calc)
            fmax = maximum(norm.(eachcol(F)))
            @test fmax < 1e-3
        end

        @testset "With Trajectory" begin
            cell = _h2_cell(10.0, 1.0)
            calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)

            # Run with trajectory
            result = opt_fire(
                calc;
                f_tol=1e-3,
                maxstep=0.2,
                keep_trajectory=true,
                verbose=false
            )

            # Check trajectory was stored
            @test length(result.trajectory) > 0
            @test result.trajectory[1].iteration == 1
        end
    end

    @testset "relax! step_constraints passthrough" begin
        cell = _h2_cell(10.0, 1.0)
        calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)
        pos0 = copy(get_positions(calc))
        result = relax!(calc;
            multi=false,
            relax_cell=false,
            method="tpsd",
            iterations=3,
            maxstep=0.1,
            step_constraints=[FixAtoms([1])],
        )
        @test result.relax.options.step_constraints[1] isa FixAtoms
        @test get_positions(calc)[:, 1] ≈ pos0[:, 1]
    end

    @testset "RelaxOption optimizer kwargs and aliases" begin
        opts = RelaxOption(
            method="fire",
            stress_threshold=0.1,
            external_pressure=[10.0],
            optimizer_kwargs=Dict{String,Any}("dt" => 0.05, "downhill_check" => true),
        )
        kwargs = EDDPotentials.Simulation._optimizer_kwargs(opts)
        @test kwargs.s_tol ≈ GPaToeVAng(0.1)
        @test kwargs.ext_pressure ≈ GPaToeVAng(10.0) .* Matrix{Float64}(I, 3, 3)
        @test kwargs.dt == 0.05
        @test kwargs.downhill_check == true

        opts6 = RelaxOption(method="tpsd", external_pressure=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        kwargs6 = EDDPotentials.Simulation._optimizer_kwargs(opts6)
        @test kwargs6.ext_pressure ≈ GPaToeVAng.([1.0 2.0 3.0; 2.0 4.0 5.0; 3.0 5.0 6.0])
    end

    @testset "relax! external_pressure uses GPa" begin
        cell = _h2_cell(10.0, 1.0)
        stress = GPaToeVAng(10.0) .* Matrix{Float64}(I, 3, 3)
        calc = StressOnlyCalc(cell, stress)
        result = relax!(calc;
            multi=false,
            relax_cell=true,
            method="tpsd",
            external_pressure=[10.0],
            stress_threshold=0.1,
            force_threshold=1e-12,
            iterations=1,
        )
        @test result.converged
        @test result.smax ≈ 0.0 atol=1e-12
    end

    @testset "relax! optimizer kwargs passthrough" begin
        cell = _h2_cell(10.0, 1.0)
        calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)
        result = relax!(calc;
            multi=false,
            relax_cell=false,
            method="fire",
            itermax=3,
            f_tol=1e-3,
            maxstep=0.1,
            dt=0.05,
            downhill_check=true,
        )
        @test result.relax.options.iterations == 3
        @test result.relax.options.force_threshold == 1e-3
        @test result.relax.options.optimizer_kwargs["dt"] == 0.05
        @test result.relax.options.optimizer_kwargs["downhill_check"] == true
    end

    @testset "Variable Cell" begin
        @testset "TPSD with relax_cell" begin
            cell = _h2_cell(10.0, 1.0)
            base_calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)

            opt = NativeTPSD(f_tol=1e-3, s_tol=1e-2, itermax=500, log_level=:minimal,
                relax_cell=true, cell_damping=0.01)

            for i in 1:20
                step!(opt, base_calc)
            end

            @test opt.iteration == 20
        end

        @testset "TPSD with selective cell constraint" begin
            cell = _h2_cell(10.0, 1.0)
            base_calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)
            H0 = get_cellmat(base_calc)[1:3, 1:3]
            opt = NativeTPSD(f_tol=1e-3, s_tol=1e-2, log_level=:minimal,
                relax_cell=true, cell_damping=0.01,
                step_constraints=[FixCellConstraint(mask=(false, false, true, false, false, false))])

            step!(opt, base_calc)
            H1 = get_cellmat(base_calc)[1:3, 1:3]
            strain = (H1 - H0) / H0
            @test strain[3, 3] ≈ 0.0 atol=1e-12
        end

        @testset "FIRE with relax_cell" begin
            cell = _h2_cell(10.0, 1.0)
            base_calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)

            opt = NativeFIRE(f_tol=1e-3, s_tol=1e-2, maxstep=0.5, itermax=500, log_level=:minimal,
                relax_cell=true, cell_damping=0.01)

            for i in 1:20
                step!(opt, base_calc)
            end

            @test opt.iteration == 20
        end

        @testset "FIRE with selective cell constraint" begin
            cell = _h2_cell(10.0, 1.0)
            base_calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)
            H0 = get_cellmat(base_calc)[1:3, 1:3]
            opt = NativeFIRE(f_tol=1e-3, s_tol=1e-2, maxstep=0.5, log_level=:minimal,
                relax_cell=true, cell_damping=0.01,
                step_constraints=[FixCellConstraint(mask=(false, false, true, false, false, false))])

            step!(opt, base_calc)
            H1 = get_cellmat(base_calc)[1:3, 1:3]
            strain = (H1 - H0) / H0
            @test strain[3, 3] ≈ 0.0 atol=1e-12
        end

        @testset "Projected stress convergence leaves raw stress unchanged" begin
            cell = _h2_cell(10.0, 1.0)
            raw_stress = diagm([0.0, 0.0, 10.0])
            calc = StressOnlyCalc(cell, raw_stress)
            opt = NativeTPSD(f_tol=1e-12, s_tol=1e-6, log_level=:minimal,
                relax_cell=true,
                step_constraints=[FixCellConstraint(mask=(false, false, true, false, false, false))])

            step!(opt, calc)
            @test eddp_stress(calc) ≈ raw_stress
            @test converged(opt, calc)
        end
    end
end
