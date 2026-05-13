module EDDPotentials

using CellBase: read_castep, read_res, write_res, read_cell, write_xyz, read_stru, write_stru
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
    predict,
    reinit!,
    clear_transient_gradients!

# Functional layer (data/training/simulation/search/workflow/analysis)
include("training/lmsolve.jl")
include("training/lmsolve_batched.jl")
include("data/preprocessing.jl")
using .Data
using .Data: _load_structures_from_path, _parse_path_slice, _count_extxyz_frames, _python_slice_to_range, _split_group_labels
include("tools.jl")
# include("eddpf90.jl")  -- removed (Fortran interface, unused in production)
include("training/sgd_training.jl")
include("training/training.jl")

include("simulation/Simulation.jl")
import .Simulation: step!
using .Simulation:
    AbstractOptimizer,
    NativeTPSD,
    NativeFIRE,
    AbstractStepConstraint,
    ZeroTranslationConstraint,
    FixAtoms,
    FixCartesian,
    FixScaled,
    FixRelative,
    FixRigidGroup,
    FixLinearIonic,
    IonicConstraint,
    FixCell,
    FixCellConstraint,
    FixVolumeConstraint,
    TrajectoryFrame,
    RelaxOption,
    Relax,
    RelaxResult,
    MDIntegrator,
    MDRunner,
    MDState,
    VelocityVerlet,
    Langevin,
    BerendsenNVT,
    apply_step_constraints!,
    initialize!,
    converged,
    reset!,
    get_iteration,
    optimize!,
    opt_tpsd,
    opt_fire,
    relax!,
    multirelax!,
    external_stress_matrix,
    external_pressure_matrix,
    run!,
    kinetic_energy,
    temperature,
    maxwell_boltzmann_velocities,
    compute_kinetic_energy,
    compute_temperature,
    rescale_velocities!,
    zero_com_velocity!,
    KB,
    total_energy,
    instantaneous_pressure,
    save_trajectory,
    save_frame,
    apply_force_constraints!,
    apply_stress_constraints!,
    time_fs,
    time_ps

include("search/buildcell.jl")
include("search/transforms.jl")
include("search/hot.jl")
include("search/rss.jl")
include("search/minsep.jl")
include("search/model_structures.jl")

include("analysis/evaluation.jl")
include("analysis/records.jl")
include("link/link.jl")
include("workflow/hyper.jl")

include("analysis/plotting/recipes.jl")
include("analysis/lazy_res.jl")
include("analysis/plotting.jl")
include("namespaces.jl")

export EDDPotentialsCore
export cellmat, cellpar
export Cell
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
    EDDPCalculator,
    load_calculator,
    create_calculator,
    eddp_energy_std,
    eddp_energy_std_per_atom
# New MetaCalculator and Modifier exports
export MetaCalculator,
    AbstractModifier,
    ExtraDimensionPenaltyModifier,
    VolumeConstraintModifier,
    AdditiveCalculatorModifier,
    get_modifier,
    get_modifier_parameter,
    set_modifier_parameter!
export opt_tpsd, opt_fire  # Explicit optimization functions
export relax!
export AbstractOptimizer, NativeTPSD, NativeFIRE
export AbstractStepConstraint,
    ZeroTranslationConstraint,
    FixAtoms,
    FixCartesian,
    FixScaled,
    FixRelative,
    FixRigidGroup,
    FixLinearIonic,
    IonicConstraint,
    FixCell,
    FixCellConstraint,
    FixVolumeConstraint,
    apply_step_constraints!,
    apply_force_constraints!,
    apply_stress_constraints!
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
export BuilderState, TrainingOption, RssPruneOption, link!
export Builder,
    walk_forward_tests, load_ensemble, load_features, load_training_dataset, run_rss,
    split_then_featurize, diagnose_formula_sampling
export rmse_per_atom, mae_per_atom, max_ae_per_atom
export TrainedModel, train_model, predict
export nfeatures
# MD exports
export MDRunner, MDState, VelocityVerlet, Langevin, BerendsenNVT
export run!, step!, kinetic_energy, temperature, maxwell_boltzmann_velocities
export total_energy, save_frame
export compute_kinetic_energy, compute_temperature
export rescale_velocities!, zero_com_velocity!, KB
# Hot-AIRSS exports
export HotAIRSSOption, hot_airss!, HotAIRSSResult
# Hyperspace relaxation exports
export hyper_get_calc, scheduled_penalty_relax!, hyper_gen_relax, HyperRoundDiagnostics
export TrainingResults
export evaluate, EvaluationConfig, AggregatedMetrics
export EnergyMetrics, ForceMetrics, StressMetrics, LayerSaturationReport
export detect_saturation, load_evaluation_data, benchmark_prediction
export compare_energy, compare_forces, compare_stress
export PhaseDiagram, read_shelx_record, ShelxRecord, ComputedRecord
export read_castep, read_res, write_res, read_cell, write_xyz, read_stru, write_stru
export synthetic_force_data

const USE_CUDA = Ref{Bool}(false)

# Stub overridden by EDDPotentialsGPU extension
function train_sgd_gpu! end

function __init__()
    reset_eddp_timer!()
end

end # module
