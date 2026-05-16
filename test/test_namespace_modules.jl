using Test
using EDDPotentials

@testset "Logical namespace modules" begin
    @test EDDPotentials.Potential.EDDPCalculator === EDDPotentials.EDDPCalculator
    @test EDDPotentials.Potential.CellFeature === EDDPotentials.CellFeature
    @test EDDPotentials.Model.EDDPCalculator === EDDPotentials.EDDPCalculator

    @test EDDPotentials.Data.StructureContainer === EDDPotentials.StructureContainer
    @test EDDPotentials.Data.FeatureStore === EDDPotentials.FeatureStore
    @test EDDPotentials.Data.FeatureBatch === EDDPotentials.FeatureBatch

    @test EDDPotentials.Training.train_model === EDDPotentials.train_model
    @test EDDPotentials.Training.TrainingResults === EDDPotentials.TrainingResults
    @test EDDPotentials.Training.TrainedModel === EDDPotentials.TrainedModel

    @test EDDPotentials.Simulation.RelaxOption === EDDPotentials.RelaxOption
    @test EDDPotentials.Simulation.relax! === EDDPotentials.relax!
    @test EDDPotentials.Simulation.MDRunner === EDDPotentials.MDRunner

    @test EDDPotentials.Search.HotAIRSSOption === EDDPotentials.HotAIRSSOption
    @test EDDPotentials.Search.hot_airss! === EDDPotentials.hot_airss!
    @test EDDPotentials.Search.model_structures === EDDPotentials.model_structures

    @test EDDPotentials.Workflow.Builder === EDDPotentials.Builder
    @test EDDPotentials.Workflow.link! === EDDPotentials.link!
    @test EDDPotentials.Workflow.run_rss === EDDPotentials.run_rss

    @test EDDPotentials.Analysis.evaluate === EDDPotentials.evaluate
    @test EDDPotentials.Analysis.PhaseDiagram === EDDPotentials.PhaseDiagram
    @test EDDPotentials.Analysis.read_shelx_record === EDDPotentials.read_shelx_record

    @test EDDPotentials.Tools.update_metadata! === EDDPotentials.update_metadata!
    @test EDDPotentials.Tools.ensure_dir === EDDPotentials.ensure_dir
    @test EDDPotentials.Tools.progress_callback! === EDDPotentials.progress_callback!

    @test parentmodule(EDDPotentials.TrainingResults) === EDDPotentials.Training
    @test parentmodule(EDDPotentials.HotAIRSSOption) === EDDPotentials.Search
    @test parentmodule(EDDPotentials.RssPruneOption) === EDDPotentials.Search
    @test parentmodule(EDDPotentials.Builder) === EDDPotentials.Workflow
    @test parentmodule(EDDPotentials.evaluate) === EDDPotentials.Analysis
    @test parentmodule(EDDPotentials.update_metadata!) === EDDPotentials.Tools
    @test parentmodule(EDDPotentials.hyper_gen_relax) === EDDPotentials.Search
    @test parentmodule(EDDPotentials.progress_callback!) === EDDPotentials.Tools
    @test isdefined(EDDPotentials, :_run_rss_link)
    @test parentmodule(EDDPotentials._run_rss_link) === EDDPotentials.Workflow

    mktempdir() do dir
        seed = joinpath(dir, "seed.cell")
        write(seed, "# seed placeholder")
        selected, expanded = EDDPotentials.Search._select_seed([seed])
        @test basename(selected) == "seed.cell"
        @test basename(only(expanded)) == "seed.cell"
    end

    for mod in (
        EDDPotentials.Potential,
        EDDPotentials.Model,
        EDDPotentials.Data,
        EDDPotentials.Training,
        EDDPotentials.Simulation,
        EDDPotentials.Search,
        EDDPotentials.Workflow,
        EDDPotentials.Analysis,
        EDDPotentials.Tools,
    )
        @test all(name -> isdefined(mod, name), names(mod))
    end
end
