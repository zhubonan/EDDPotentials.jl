using Test
using EDDPotentials
using CellBase

@testset "Core Module Namespace" begin
    @test isdefined(EDDPotentials, :EDDPotentialsCore)

    core = EDDPotentials.EDDPotentialsCore
    @test core.EDDPCalculator === EDDPotentials.EDDPCalculator
    @test core.CellFeature === EDDPotentials.CellFeature

    cell = Cell(Lattice(6.0, 6.0, 6.0), [:H, :H], [0.0 0.0; 0.0 2.0; 0.0 0.0])
    calc = EDDPotentials.lj_like_calc(cell; α=1.0, a=6, rc=3.0)

    @test core.eddp_energy(calc) ≈ EDDPotentials.eddp_energy(calc)
    @test core.eddp_forces(calc) ≈ EDDPotentials.eddp_forces(calc)
end
