module EDDPotentials

using CellBase: read_castep
import CellBase
using StatsBase: mean
export mean
using TimerOutputs

const to = TimerOutput()
# Default to have the timer disabled
disable_timer!(to)

enable_eddp_timer!() = enable_timer!(to)
disable_eddp_timer!() = disable_timer!(to)
reset_eddp_timer!() = reset_timer!(to)

export enable_eddp_timer!, disable_eddp_timer!, reset_eddp_timer!



# Re-export `glob`
using Glob
export glob


# abstract types
abstract type EDDPotentialsOption end

# end of abstract types

# Internal layering:
# - `core_layer.jl`: feature generation and evaluator kernel
# - `functional_layer.jl`: training/workflow/analysis code built on top of core
include("core_layer.jl")
include("functional_layer.jl")

export Lattice, reciprocal, cellmat, cellvecs, cellpar, wrap!, volume, frac_pos, lattice
export displace!, clip, Cell, supercell
export distance_matrix, laplacian_matrix, nmodules, find_modules
export get_cell,
    eddp_energy,
    eddp_forces,
    eddp_energy_per_atom,
    eddp_stress,
    eddp_pressure,
    eddp_pressure_gpa,
    eddp_enthalpy,
    CellFeature,
    NNCalc,
    VariableCellCalc,
    eddp_energy_std,
    eddp_energy_std_per_atom
# New MetaCalculator and Modifier exports
export MetaCalculator,
    AbstractModifier,
    ExtraDimensionPenaltyModifier,
    CellRelaxationModifier,
    AdditiveCalculatorModifier,
    get_modifier,
    get_modifier_parameter,
    set_modifier_parameter!,
    get_augmented_positions,
    set_augmented_positions!
export opt_tpsd, opt_fire  # Explicit optimization functions
export AbstractOptimizer, NativeTPSD, NativeFIRE
export initialize!, step!, converged, reset!, get_iteration, optimize!
export get_positions, set_positions!, set_cell!, set_cellmat!
export StructureContainer, FeatureContainer, load_from_jld2
export BuilderState, TrainingOption, link!
export Builder,
    walk_forward_tests, load_ensemble, load_features, load_training_dataset, run_rss
export rmse_per_atom, mae_per_atom, max_ae_per_atom
export nfeatures
export TrainingResults
export PhaseDiagram, read_shelx_record, ShelxRecord, ComputedRecord

const USE_CUDA = Ref{Bool}(false)

function __init__()
    reset_timer!(to)
end

end # module
