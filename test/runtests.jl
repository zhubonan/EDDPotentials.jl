using EDDPotentials
using CellBase
using Test

@testset "Tools" begin
    cell = Cell(Lattice(10, 10, 10), [:H], [[0.0, 0.0, 4]])
    cpar = cellpar(cell)
    spos = CellBase.scaled_positions(cell)
    # Test for rattling the cell
    EDDPotentials.rattle_cell!(cell, 0.01)
    @test all(cpar .!= cellpar(cell))
    @test all(isapprox.(spos, CellBase.scaled_positions(cell), atol=1e-10))
end

include("utils.jl")
include("test_layer_boundaries.jl")
include("test_source_topology.jl")
include("test_core_module_namespace.jl")

include("test_records.jl")
include("test_cellfeature.jl")
include("test_random_embedded_feature.jl")
include("test_embedding.jl")
include("test_gradients.jl")
include("test_backprop.jl")
include("test_nninterface.jl")
include("test_json_serialization.jl")
include("test_calc.jl")
include("test_preprocess.jl")
include("test_dotcastep.jl")
include("test_train.jl")
include("test_builder.jl")
#include("test_link.jl")

# New MetaCalculator and Modifier tests
include("test_modifiers.jl")
include("test_metacalculator.jl")

# MD tests
include("test_md.jl")

# Native optimizer tests are opt-in while the API stabilizes.
# Enable with: EDDP_RUN_NATIVE_OPT_TESTS=1
if get(ENV, "EDDP_RUN_NATIVE_OPT_TESTS", "0") == "1"
    native_opt_tests = joinpath(@__DIR__, "test_native_optimizers.jl")
    if isfile(native_opt_tests)
        include("test_native_optimizers.jl")
    else
        @warn "EDDP_RUN_NATIVE_OPT_TESTS=1 but test_native_optimizers.jl is missing"
    end
end

@testset "Model" begin
    calc = _get_lco_calc()
    for (n, order) in zip([6, 10, 15], [2, 3, 4])
        combs, structures =
            EDDPotentials.model_structures(unique(species(calc.cell)); order)
        @test length(combs) == n
    end
end
