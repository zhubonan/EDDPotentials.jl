using Test
using EDDPotentials
using CellBase

@testset "Potential Module Namespace" begin
    @test isdefined(EDDPotentials, :Potential)
    @test !isdefined(EDDPotentials, :EDDPotentialsCore)
    @test :relax! in names(EDDPotentials)
    @test :read_castep in names(EDDPotentials)
    @test :read_res in names(EDDPotentials)
    @test :write_res in names(EDDPotentials)
    @test :read_cell in names(EDDPotentials)
    @test :write_xyz in names(EDDPotentials)
    @test :read_stru in names(EDDPotentials)
    @test :write_stru in names(EDDPotentials)

    core = EDDPotentials.Potential
    @test core.EDDPCalculator === EDDPotentials.EDDPCalculator
    @test core.CellFeature === EDDPotentials.CellFeature
    @test core.eddp_energy === EDDPotentials.eddp_energy
    @test core.eddp_forces === EDDPotentials.eddp_forces
end
