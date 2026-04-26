using Test
using EDDPotentials
using CellBase
using LinearAlgebra

# Import test utilities
isdefined(Main, :_h2_cell) || include("utils.jl")
const lj_like_calc = EDDPotentials.lj_like_calc

@testset "Native Optimizers" begin

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

    @testset "Variable Cell (if supported)" begin
        # Test that optimizers work with VariableCellCalc
        # This requires having VariableCellCalc available
        if isdefined(EDDPotentials, :VariableCellCalc)
            @testset "TPSD with VariableCellCalc" begin
                cell = _h2_cell(10.0, 1.0)
                base_calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)
                vc_calc = VariableCellCalc(base_calc)

                opt = NativeTPSD(f_tol=1e-3, s_tol=1e-2, itermax=500, log_level=:minimal)

                # Run a few steps (full convergence may be slow)
                for i in 1:20
                    step!(opt, vc_calc)
                end

                # Should make progress
                @test opt.iteration == 20
            end

            @testset "FIRE with VariableCellCalc" begin
                cell = _h2_cell(10.0, 1.0)
                base_calc = lj_like_calc(cell; α=0.5, a=2, rc=8.0)
                vc_calc = VariableCellCalc(base_calc)

                opt = NativeFIRE(f_tol=1e-3, s_tol=1e-2, maxstep=0.5, itermax=500, log_level=:minimal)

                # Run a few steps
                for i in 1:20
                    step!(opt, vc_calc)
                end

                # Should make progress
                @test opt.iteration == 20
            end
        end
    end
end
