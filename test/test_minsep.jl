using EDDPotentials: get_specie_pairs, compute_specie_separations, gather_minsep_stats
using CellBase, Test

@testset "Minsep" begin
    @testset "get_specie_pairs" begin
        p = get_specie_pairs([:H,:O])
        @test Set(p) == Set([:H=>:H, :H=>:O, :O=>:O])
        @test any(==(:O=>:H), get_specie_pairs([:H,:O]; permute=true))
        @test get_specie_pairs([:H]) == [:H=>:H]
    end

    @testset "compute_specie_separations" begin
        cell = Cell(Lattice(5,5,5), [:H,:H], [0 1.8; 0 0.0; 0 0])
        seps = compute_specie_separations(cell; rcut=6)
        @test seps[:H=>:H].minsep ≈ 1.8 atol=1e-10
        @test seps[:H=>:H] === seps[:H=>:H]
    end

    @testset "multi-species" begin
        cell = Cell(Lattice(5,5,5), [:H,:H,:O], [0 0.1 1; 0 1 1; 0.1 0 1])
        seps = compute_specie_separations(cell; rcut=6)
        @test seps[:H=>:O] === seps[:O=>:H]
        @test all(v -> v.minsep < 999.0, values(seps))
    end

    @testset "gather_minsep_stats" begin
        c1 = Cell(Lattice(5,5,5), [:H,:H], [0 1.8; 0 0; 0 0])
        c2 = Cell(Lattice(5,5,5), [:H,:H], [0 2.5; 0 0; 0 0])
        ms, _ = gather_minsep_stats([c1, c2]; rcut=6)
        @test minimum(ms[:H=>:H]) ≈ 1.8 atol=1e-10
    end
end
