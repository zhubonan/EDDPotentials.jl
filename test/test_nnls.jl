using EDDPotentials.Potential.NNLS:
    NNLSWorkspace, load!, solve!, nnls, QP, QPWorkspace,
    primal_infeasibility, dual_infeasibility, stationarity_violation,
    slackness_violation, check_optimality_conditions
using Test, LinearAlgebra

@testset "NNLS" begin
    @testset "Known solutions" begin
        @test nnls([1.0 0 0; 0 1 0; 0 0 1], [1.0, -2, 3]) ≈ [1, 0, 3] atol=1e-12
        @test nnls([1.0 0; 0 1], [-3.0, -5]) ≈ [0, 0] atol=1e-12
    end

    @testset "Matches backslash for non-negative solution" begin
        A, x_true = randn(10, 4), rand(4)
        x_nnls = nnls(A, A * x_true)
        @test x_nnls ≈ x_true atol=1e-8
    end

    @testset "Workspace reuse" begin
        w = NNLSWorkspace([1.0 0; 0 1], [3.0, 0])
        @test solve!(w) ≈ [3, 0] atol=1e-12
        solve!(w, [2.0 0; 0 2], [0, 5.0])
        @test w.x ≈ [0, 2.5] atol=1e-12
    end

    @testset "QP box constraint" begin
        qp = QP([1.0 0; 0 1.0], [-1.0, -1], Matrix([-1.0 0; 0 -1.0]), [0.0, 0])
        z, λ = solve!(QPWorkspace(qp))
        @test all(z .>= -1e-8)
        @test primal_infeasibility(qp, z, λ) <= 1e-8
        @test dual_infeasibility(qp, z, λ) <= 1e-8
    end

    @testset "QP inequality active" begin
        qp = QP([1.0 0; 0 1.0], [-2.0, -2], reshape([1.0, 1.0], 1, 2), [1.0])
        z, λ = solve!(QPWorkspace(qp))
        @test z[1] + z[2] <= 1 + 1e-6
        @test check_optimality_conditions(qp, z, λ) < 1e-4
    end
end
