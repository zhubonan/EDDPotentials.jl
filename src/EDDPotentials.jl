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
    AbstractCellFeatureSpec,
    AbstractNNInterface,
    CellFeature,
    RandomEmbeddedCellFeature,
    MultiRandomEmbeddedCellFeature,
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
    load_feature_artifact,
    load_feature_json,
    load_model,
    load_model_artifact,
    load_from_jld2,
    save_feature_json,
    save_model_artifact,
    nparams,
    one_body_offset,
    _make_core_repulsion,
    _pair_rcut,
    suggest_rcut,
    setup_fj,
    setup_fj_batched,
    setup_fj_batched_threaded,
    ThreadBufferPool,
    compute_objectives_threaded,
    rmse_per_atom_threaded,
    setup_fj_with_embedding,
    setup_fj_batched_with_embedding,
    grad_embedding!,
    recompute_features!,
    compute_feature_stats!,
    update_feature_vector!,
    reinit,
    EDDPCalculator,
    load_calculator,
    extract_nncalc,
    predict,
    reinit!

# Functional layer (training/workflow/analysis)
include("lmsolve.jl")
include("lmsolve_batched.jl")
include("preprocessing.jl")
include("tools.jl")
include("eddpf90.jl")
include("sgd_training.jl")
include("training.jl")
include("evaluation.jl")
include("optimizer/abstract.jl")
include("functional/functional.jl")

include("optimizer/utils.jl")
include("optimizer/tpsd.jl")
include("optimizer/fire.jl")
include("opt.jl")
include("md.jl")
include("hot.jl")
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
    RandomEmbeddedCellFeature,
    MultiRandomEmbeddedCellFeature,
    NNCalc,
    VariableCellCalc,  # deprecated — returns MetaCalculator
    EDDPCalculator,
    load_calculator,
    extract_nncalc,
    create_calculator,
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
export StructureContainer,
    FeatureStore,
    FeatureBatch,
    each_batch,
    get_data,
    save_feature_store,
    save_feature_store_streaming,
    load_feature_store,
    load_batch,
    get_fit_data,
    load_from_jld2,
    load_model,
    load_model_artifact,
    save_model_artifact,
    load_feature_json,
    load_feature_artifact,
    save_feature_json
export split_and_featurize
export BuilderState, TrainingOption, link!
export Builder,
    walk_forward_tests, load_ensemble, load_features, load_training_dataset, run_rss,
    split_then_featurize
export rmse_per_atom, mae_per_atom, max_ae_per_atom
export TrainedModel, train_model, predict
export nfeatures
# MD exports
export MDRunner, MDState, VelocityVerlet, Langevin, BerendsenNVT
export run!, step!, kinetic_energy, temperature, maxwell_boltzmann_velocities
export total_energy, instantaneous_pressure, save_trajectory, save_frame
export time_fs, time_ps
# Hot-AIRSS exports
export HotAIRSSOption, hot_airss!, HotAIRSSResult
export TrainingResults
export evaluate, EvaluationConfig, AggregatedMetrics
export EnergyMetrics, ForceMetrics, StressMetrics, LayerSaturationReport
export detect_saturation, load_evaluation_data, benchmark_prediction
export compare_energy, compare_forces, compare_stress
export PhaseDiagram, read_shelx_record, ShelxRecord, ComputedRecord
export synthetic_force_data

const USE_CUDA = Ref{Bool}(false)

# Stub overridden by EDDPotentialsGPU extension
function train_sgd_gpu! end

function __init__()
    reset_eddp_timer!()
end

end # module
