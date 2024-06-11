using EDDPotentialsTools
using Test
using EDDPotentials: PhaseDiagram, ComputedRecord
using Plots

@testset "binary hull" begin
    records = [
        ComputedRecord(:H, 0.0),
        ComputedRecord(:O, 0.0),
        ComputedRecord(:H2O, -1.0),
        ComputedRecord(:H2O2, -0.1),
    ]


    phased = PhaseDiagram(records)
    p = EDDPotentialsTools.make_binary_hull_plot(phased)
    @test isa(p, Plots.Plot)
end
