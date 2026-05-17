using EDDPotentials
using CellBase
using Random
using Test

@testset "Utilities" begin
    cell = Cell(Lattice(10, 10, 10), [:H], [[0.0, 0.0, 4]])
    cpar = cellpar(cell)
    spos = CellBase.scaled_positions(cell)
    # Test for rattling the cell
    EDDPotentials.rattle_cell!(cell, 0.01)
    @test all(cpar .!= cellpar(cell))
    @test all(isapprox.(spos, CellBase.scaled_positions(cell), atol=1e-10))

    Random.seed!(1234)
    length_changes = Float64[]
    for _ = 1:50
        test_cell = Cell(Lattice(10, 10, 10), [:H], [[0.0, 0.0, 4]])
        EDDPotentials.rattle_cell!(test_cell, 0.01)
        append!(length_changes, cellpar(test_cell)[1:3] .- cpar[1:3])
    end
    @test minimum(length_changes) < 0
    @test maximum(length_changes) > 0

    Random.seed!(5678)
    pos_cell = Cell(Lattice(10, 10, 10), [:H], [[5.0, 5.0, 5.0]])
    pos0 = copy(get_positions(pos_cell))
    pos_changes = Float64[]
    for _ = 1:50
        set_positions!(pos_cell, pos0)
        rattle!(pos_cell, 0.01)
        append!(pos_changes, get_positions(pos_cell) .- pos0)
    end
    @test minimum(pos_changes) < 0
    @test maximum(pos_changes) > 0
end

include("utils.jl")
include("test_layer_boundaries.jl")
include("test_source_topology.jl")
include("test_core_module_namespace.jl")
include("test_namespace_modules.jl")
include("test_nnls.jl")

include("test_records.jl")
include("test_cellfeature.jl")
include("test_spline.jl")
include("test_random_embedded_feature.jl")
include("test_multi_embedding.jl")
include("test_embedding.jl")
include("test_gradients.jl")
include("test_backprop.jl")
include("test_nninterface.jl")
include("test_pairpot.jl")
include("test_json_serialization.jl")
include("test_calc.jl")
include("test_eddpcalculator.jl")
include("test_preprocess.jl")
include("test_dotcastep.jl")
include("test_train.jl")
include("test_evaluation.jl")
include("test_threading.jl")
include("test_sgd_training.jl")
include("test_builder.jl")
include("test_formula_sampling.jl")
include("test_rss_pruning.jl")
#include("test_link.jl")

# New MetaCalculator and Modifier tests
include("test_modifiers.jl")
include("test_metacalculator.jl")
include("test_hyper.jl")
include("test_minsep.jl")
include("test_model_structures.jl")

# MD tests
include("test_md.jl")

# Hot-AIRSS tests
include("test_hot.jl")

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

@testset "Model structures" begin
    calc = _get_lco_calc()
    for (n, order) in zip([6, 10, 15], [2, 3, 4])
        combs, structures =
            EDDPotentials.model_structures(unique(species(calc.cell)); order)
        @test length(combs) == n
    end
end
