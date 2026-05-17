using EDDPotentials
using CellBase, Test, LinearAlgebra

@testset "Model structures" begin
    @testset "Pair structures" begin
        cells = model_structures(1.0:1.0:3.0, :Si, :Si, 5.0)
        @test length(cells) == 3
        for c in cells
            @test length(c) == 2
            @test species(c) == [:Si, :Si]
        end
        d = norm(cells[1].positions[:, 2] - cells[1].positions[:, 1])
        @test d ≈ 1.0 atol=1e-10
    end

    @testset "Triplet structures" begin
        cells = model_structures(2.0:1.0:3.0, :Si, :O, :Si, 5.0)
        @test length(cells) == 2
        @test length(cells[1]) == 3
    end

    @testset "High-level combinatorial" begin
        combs, structs = model_structures([:Si, :O]; order=2, npts=5)
        @test length(combs) == 3  # Si-Si, Si-O, O-O
        @test all(s -> length(s) == 5, structs)
    end
end
