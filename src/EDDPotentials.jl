module EDDPotentials

using CellBase: read_castep, read_res, write_res, read_cell, write_xyz, read_stru, write_stru
import CellBase
using StatsBase: mean
export mean
using TimerOutputs

const to = TimerOutput()
# Default to have the timer disabled
disable_timer!(to)

"""
    enable_eddp_timer!()

Enable internal timing collection for EDDP operations.
"""
enable_eddp_timer!() = enable_timer!(to)
"""
    disable_eddp_timer!()

Disable internal timing collection for EDDP operations.
"""
disable_eddp_timer!() = disable_timer!(to)
"""
    reset_eddp_timer!()

Reset internal timing counters.
"""
reset_eddp_timer!() = reset_timer!(to)

# Re-export `glob`
using Glob
export glob


# abstract types
abstract type EDDPotentialsOption end

# end of abstract types

include("potential/Potential.jl")
using .Potential

# Bring selected core symbols into the top-level namespace where
# functional files extend/use them without qualification.
import .Potential:
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

include("training/Training.jl")
using .Training
using .Training:
    _load_structures_from_path,
    _parse_path_slice,
    _count_extxyz_frames,
    _python_slice_to_range,
    _split_group_labels

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

const USE_CUDA = Ref{Bool}(false)

# Stub overridden by EDDPotentialsGPU extension
function train_sgd_gpu! end

# Public package namespaces and their source ownership.

include("search/Search.jl")
include("workflow/Workflow.jl")
include("analysis/Analysis.jl")

using .Search
using .Workflow
using .Analysis
import .Potential:
    _fd_energy,
    _fd_energy_vc,
    _fd_strain
import .Training:
    _prepare_presplit,
    levenberg_marquardt,
    levenberg_marquardt_batched,
    levenberg_marquardt_gpu,
    constant_lr,
    step_decay_lr,
    cosine_annealing_lr,
    _scatter_matrix,
    _hcat_batch
import .Search:
    bP,
    _rss_candidate,
    _rss_pool_mean,
    _rss_pool_median,
    _rss_should_flush,
    _rss_select_pruned,
    _validate_rss_prune_option
import .Workflow:
    TRAINING_DIR,
    LocalLinkBackend,
    add_threads_env,
    generated_input_count,
    required_success_count,
    validate_dft_outputs!,
    validate_generated_structures!,
    is_generation_data_ready,
    is_training_dataset_ready,
    run_dft_phase!,
    run_training_phase!,
    load_training_data,
    _run_rss_link,
    _render_training_job_script,
    _training_submit_cmd,
    _required_models_for_ensemble,
    _generate_random_structures,
    _models_ready_for_ensemble,
    _inject_formula_directive,
    _build_formula_pool,
    _has_neutral_oxidation_state,
    _import_known_structures!,
    _formula_sampling_context,
    _make_seed_text_transform,
    client_server_run_dft_phase!,
    client_server_run_training_phase!,
    _client_server_wait_for_dft!,
    _render_client_server_client_job_script,
    _client_server_submit_clients
import .Analysis:
    bary_coords,
    contains_point,
    coords_from_bary,
    get_coord,
    _extract_reference_energy,
    _extract_reference_forces,
    _extract_reference_stress,
    _r2_score

export Potential
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
export BuilderState, TrainingOption, RssPruneOption, ClientServerOption, link!
export Builder,
    walk_forward_tests, load_ensemble, load_features, load_training_dataset, run_rss,
    split_then_featurize, diagnose_formula_sampling
export client_server_master,
    link_client_server,
    client_server_client,
    link_worker,
    client_server_train_slot,
    client_server_write_heartbeat,
    client_server_worker_states,
    client_server_prepare_crud_workdir!,
    client_server_collect_crud_results!,
    client_server_prepare_training_phase!,
    client_server_claim_training_slot!,
    client_server_mark_training_slot_done!,
    client_server_mark_training_slot_failed!,
    client_server_training_model_path
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

function __init__()
    reset_eddp_timer!()
end

end # module
