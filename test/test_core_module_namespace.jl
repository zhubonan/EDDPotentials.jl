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

    top_exports = Set(names(EDDPotentials; all=false))
    for module_name in (:Potential, :Training, :Simulation, :Search, :Workflow, :Analysis)
        module_exports = Set(names(getproperty(EDDPotentials, module_name); all=false))
        @test module_exports ⊆ top_exports
    end

    @test EDDPotentials.Search.RssPruneOption === EDDPotentials.Workflow.RssPruneOption
    @test EDDPotentials.Search.hyper_gen_relax === EDDPotentials.Workflow.hyper_gen_relax
    @test EDDPotentials.Search.run_rss === EDDPotentials.Workflow.run_rss
    @test EDDPotentials.Simulation.step! === EDDPotentials.Workflow.step!
    @test EDDPotentials.Potential.save_model_artifact ===
          EDDPotentials.Training.save_model_artifact

    core = EDDPotentials.Potential
    @test core.EDDPCalculator === EDDPotentials.EDDPCalculator
    @test core.CellFeature === EDDPotentials.CellFeature
    @test core.eddp_energy === EDDPotentials.eddp_energy
    @test core.eddp_forces === EDDPotentials.eddp_forces
end
