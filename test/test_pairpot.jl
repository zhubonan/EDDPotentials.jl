using EDDPotentials
using EDDPotentials.Potential: PairpotCalc, PairpotWorkspace, build_lj_spec, _repulsion, _grepulsion, e_lj
using CellBase, Test, LinearAlgebra

@testset "Pairpot" begin
    @testset "Repulsion helpers" begin
        @test _repulsion(1.0, 2.0, 12.0) ≈ (2*(2/1-1))^12
        @test _grepulsion(1.0, 2.0, 12.0) ≈ -12*(2*(2/1-1))^11/(1^2)*2*2
        @test _repulsion(3.0, 2.0, 12.0) ≈ 0 atol=1e-15
    end

    @testset "LJ analytic minimum" begin
        @test e_lj(3.4*2^(1/6), 3.4, 0.01) ≈ -0.01 atol=1e-12
    end

    @testset "LJSpec construction" begin
        spec = build_lj_spec(((:Ar,:Ar), 3.4); epsilon=0.01, cutoff=10.0)
        @test :Ar in spec.symbols
        @test spec.cutoff ≈ 10.0
        @test spec.sigma[1,1] ≈ 3.4
    end

    @testset "LJSpec calculator" begin
        cell = Cell(Lattice(15,15,15), [:Ar,:Ar], [0 4.0; 0 0.0; 0 0])
        spec = build_lj_spec(((:Ar,:Ar), 3.4); epsilon=0.01, cutoff=12.0)
        calc = PairpotCalc(cell, PairpotWorkspace(cell, spec); nl=NeighbourList(cell, 12.0; savevec=true))
        @test isfinite(eddp_energy(calc))
        f = eddp_forces(calc)
        @test all(isfinite, f)
        @test all(abs.(sum(f, dims=2)) .< 1e-10)
        s = eddp_stress(calc)
        @test s ≈ s' atol=1e-10
    end

    @testset "LJ stress is volume-normalized" begin
        spec = build_lj_spec(((:Ar,:Ar), 3.4); epsilon=0.01, cutoff=12.0)
        cell1 = Cell(Lattice(15,15,15), [:Ar,:Ar], [0 4.0; 0 0.0; 0 0])
        cell2 = Cell(Lattice(30,30,30), [:Ar,:Ar], [0 4.0; 0 0.0; 0 0])
        calc1 = PairpotCalc(cell1, PairpotWorkspace(cell1, spec); nl=NeighbourList(cell1, 12.0; savevec=true))
        calc2 = PairpotCalc(cell2, PairpotWorkspace(cell2, spec); nl=NeighbourList(cell2, 12.0; savevec=true))

        s1 = eddp_stress(calc1)
        s2 = eddp_stress(calc2)
        @test s2 ≈ s1 .* (15.0^3 / 30.0^3) atol=1e-12
    end

    @testset "LJ force FD" begin
        cell = Cell(Lattice(15,15,15), [:Ar,:Ar], [0 3.5; 0 0.0; 0 0])
        spec = build_lj_spec(((:Ar,:Ar), 3.4); epsilon=0.01, cutoff=12.0)
        calc = PairpotCalc(cell, PairpotWorkspace(cell, spec); nl=NeighbourList(cell, 12.0; savevec=true))
        e0 = eddp_energy(calc)
        amp = 1e-5
        pos = copy(get_positions(calc)); pos[1] += amp
        set_positions!(calc, pos)
        e1 = eddp_energy(calc)
        @test isfinite(e1)
        @test e1 != e0
    end

    @testset "AbstractCalc interface" begin
        cell = Cell(Lattice(15,15,15), [:Ar,:Ar], [0 4.0; 0 0.0; 0 0])
        spec = build_lj_spec(((:Ar,:Ar), 3.4); epsilon=0.01, cutoff=12.0)
        calc = PairpotCalc(cell, PairpotWorkspace(cell, spec); nl=NeighbourList(cell, 12.0; savevec=true))
        @test get_cell(calc) === cell
        pos = copy(get_positions(calc)); pos[1] += 0.1
        set_positions!(calc, pos)
        @test get_positions(calc)[1] ≈ pos[1]
    end
end
