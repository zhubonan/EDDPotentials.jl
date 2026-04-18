module EDDPotentials

using CellBase: read_castep
import CellBase
using StatsBase: mean
export mean
using TimerOutputs



# Re-export `glob`
using Glob
export glob


# abstract types
abstract type EDDPotentialsOption end

# end of abstract types

include("core/EDDPotentialsCore.jl")
using .EDDPotentialsCore
const to = EDDPotentialsCore.to

# Bring selected core symbols into the top-level namespace where
# functional files extend/use them without qualification.
import .EDDPotentialsCore:
    AbstractNNInterface,
    CellFeature,
    EnsembleNNInterface,
    FluxInterface,
    GPaToeVAng,
    LinearInterface,
    ManualFluxBackPropInterface,
    _force_update!,
    _stress_update!,
    compute_objectives,
    compute_objectives_diff,
    compute_fv_gv,
    eddp_energy,
    eddp_forces,
    eddp_pressure,
    eddp_energy_std,
    eddp_enthalpy,
    eddp_energy_per_atom,
    eVAngToGPa,
    get_flux_model,
    ginit,
    get_elemental_energy,
    feature_vector,
    glob_allow_abs,
    load_from_jld2,
    setup_fj,
    setup_fj_batched,
    update_feature_vector!,
    reinit,
    reinit!

# Functional layer (training/workflow/analysis)
include("lmsolve.jl")
include("lmsolve_batched.jl")
include("preprocessing.jl")
include("eddpf90.jl")
include("training.jl")

include("tools.jl")
include("functional/functional.jl")

include("optimizer/abstract.jl")
include("optimizer/utils.jl")
include("optimizer/tpsd.jl")
include("optimizer/fire.jl")
include("opt.jl")

include("records.jl")
include("quickbuild.jl")
include("link/link.jl")
include("hyper.jl")

include("minsep.jl")
include("plotting/recipes.jl")
include("lazy_res.jl")
include("model_structures.jl")
include("plotting.jl")

export EDDPotentialsCore
export enable_eddp_timer!, disable_eddp_timer!, reset_eddp_timer!
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
    reset_eddp_timer!()
end

end # module
