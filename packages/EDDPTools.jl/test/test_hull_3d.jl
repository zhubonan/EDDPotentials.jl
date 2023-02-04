using EDDP: ComputedRecord, PhaseDiagram
using EDDPTools
using PlotlyJS
using Test

@testset "ternary" begin
    records = [
        ComputedRecord(:H, 0.0),
        ComputedRecord(:O, 0.0),
        ComputedRecord(:C, 0.0),
        ComputedRecord(:HCO, -0.5),
        ComputedRecord(:H2CO, -0.5000001),
        ComputedRecord(:H2C2O, 1.0),
    ]

    phased = PhaseDiagram(records)
    p = EDDPTools.make_ternary_plot(phased)

    @test isa(p, PlotlyJS.SyncPlot)
end
