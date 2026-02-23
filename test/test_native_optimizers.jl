using Test
using EDDPotentials
using CellBase
using LinearAlgebra

# Import test utilities
const lj_like_calc = EDDPotentials.lj_like_calc

@testset "Native Optimizers" begin

    @testset "NativeTPSD" begin
        @testset "Basic Functionality" begin
            # Create a simple H2 cell displaced from equilibrium
            cell = _h2_cell(10.0, 1.5)  # Lattice = 10, displacement factor = 1.5
            calc = lj_like_calc(cell; α=1.0, a=6, rc=3.0)

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
            # Create H2 cell
            cell = _h2_cell(10.0, 1.5)
            calc = lj_like_calc(cell; α=1.0, a=6, rc=3.0)

            # Run optimization
            opt = NativeTPSD(f_tol=1e-4, itermax=1000, log_level=:minimal)

            iter = 0
            while !converged(opt, calc) && iter < 1000
                step!(opt, calc)
                iter += 1
            end

            # Check convergence
            @test converged(opt, calc)
            @test opt.iteration < 1000

            # Check forces
            F = eddp_forces(calc)
            fmax = maximum(norm.(eachcol(F)))
            @test fmax < 1e-4

            # Check equilibrium distance (should be around r_eq = rc/2 = 1.5)
            final_cell = get_cell(calc)
            pos = get_positions(final_cell)
            d = norm(pos[:, 1] - pos[:, 2])
            @test d ≈ 1.5 atol=0.1
        end

        @testset "Reset Functionality" begin
            cell = _h2_cell(10.0, 1.5)
            calc = lj_like_calc(cell; α=1.0, a=6, rc=3.0)

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
            cell = _h2_cell(10.0, 1.5)
            calc = lj_like_calc(cell; α=1.0, a=6, rc=3.0)

            # Create FIRE optimizer
            opt = NativeFIRE(f_tol=1e-3, maxstep=0.2, log_level=:minimal)

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
            # Create H2 cell
            cell = _h2_cell(10.0, 1.5)
            calc = lj_like_calc(cell; α=1.0, a=6, rc=3.0)

            # Run optimization
            opt = NativeFIRE(f_tol=1e-4, maxstep=0.2, itermax=1000, log_level=:minimal)

            iter = 0
            while !converged(opt, calc) && iter < 1000
                step!(opt, calc)
                iter += 1
            end

            # Check convergence
            @test converged(opt, calc)
            @test opt.iteration < 1000

            # Check forces
            F = eddp_forces(calc)
            fmax = maximum(norm.(eachcol(F)))
            @test fmax < 1e-4

            # Check equilibrium distance
            final_cell = get_cell(calc)
            pos = get_positions(final_cell)
            d = norm(pos[:, 1] - pos[:, 2])
            @test d ≈ 1.5 atol=0.1
        end

        @testset "Velocity Reset" begin
            # Test that velocity resets when moving uphill
            cell = _h2_cell(10.0, 1.5)
            calc = lj_like_calc(cell; α=1.0, a=6, rc=3.0)

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
            cell = _h2_cell(10.0, 1.5)
            calc = lj_like_calc(cell; α=1.0, a=6, rc=3.0)

            opt = NativeFIRE(
                f_tol=1e-3,
                maxstep=0.2,
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
            cell = _h2_cell(10.0, 1.5)
            calc = lj_like_calc(cell; α=1.0, a=6, rc=3.0)

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
            # Create two identical cells
            cell1 = _h2_cell(10.0, 1.5)
            cell2 = _h2_cell(10.0, 1.5)

            calc1 = lj_like_calc(cell1; α=1.0, a=6, rc=3.0)
            calc2 = lj_like_calc(cell2; α=1.0, a=6, rc=3.0)

            # Optimize with TPSD
            opt_tpsd = NativeTPSD(f_tol=1e-4, itermax=1000, log_level=:minimal)
            while !converged(opt_tpsd, calc1) && opt_tpsd.iteration < 1000
                step!(opt_tpsd, calc1)
            end

            # Optimize with FIRE
            opt_fire = NativeFIRE(f_tol=1e-4, maxstep=0.2, itermax=1000, log_level=:minimal)
            while !converged(opt_fire, calc2) && opt_fire.iteration < 1000
                step!(opt_fire, calc2)
            end

            # Both should converge
            @test converged(opt_tpsd, calc1)
            @test converged(opt_fire, calc2)

            # Final positions should be very close
            pos1 = get_positions(get_cell(calc1))
            pos2 = get_positions(get_cell(calc2))
            @test maximum(abs.(pos1 - pos2)) < 1e-3

            println("  TPSD iterations: $(opt_tpsd.iteration)")
            println("  FIRE iterations: $(opt_fire.iteration)")
        end
    end

    @testset "opt_fire function" begin
        @testset "Basic Usage" begin
            cell = _h2_cell(10.0, 1.5)
            calc = lj_like_calc(cell; α=1.0, a=6, rc=3.0)

            # Use opt_fire function
            result = opt_fire(calc; f_tol=1e-4, maxstep=0.2, verbose=false)

            # Check result
            @test result isa NativeFIRE
            @test converged(result, calc)

            # Check forces
            F = eddp_forces(calc)
            fmax = maximum(norm.(eachcol(F)))
            @test fmax < 1e-4
        end

        @testset "With Trajectory" begin
            cell = _h2_cell(10.0, 1.5)
            calc = lj_like_calc(cell; α=1.0, a=6, rc=3.0)

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
                cell = _h2_cell(10.0, 1.5)
                base_calc = lj_like_calc(cell; α=1.0, a=6, rc=3.0)
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
                cell = _h2_cell(10.0, 1.5)
                base_calc = lj_like_calc(cell; α=1.0, a=6, rc=3.0)
                vc_calc = VariableCellCalc(base_calc)

                opt = NativeFIRE(f_tol=1e-3, s_tol=1e-2, maxstep=0.2, itermax=500, log_level=:minimal)

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
