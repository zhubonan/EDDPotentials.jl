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

include("data/preprocessing.jl")
using .Data
using .Data: _load_structures_from_path, _parse_path_slice, _count_extxyz_frames, _python_slice_to_range, _split_group_labels

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

module Potential

using ..EDDPotentialsCore
using CellBase:
    reciprocal,
    clip,
    frac_pos
import ..EDDPotentialsCore:
    AbstractCellFeatureSpec,
    AbstractNNInterface,
    LinearInterface,
    FluxInterface,
    ManualFluxBackPropInterface,
    EnsembleNNInterface,
    load_from_jld2

export Lattice, reciprocal, cellmat, cellvecs, cellpar, wrap!, volume, frac_pos, lattice
export clip, Cell
export distance_matrix
export get_positions, set_positions!, set_cell!, set_cellmat!, remove_dimensions
export AbstractCalc, get_cell
export eddp_energy,
    eddp_forces,
    eddp_energy_per_atom,
    eddp_stress,
    eddp_pressure,
    eddp_pressure_gpa,
    eddp_pressure_GPa,
    eddp_enthalpy,
    eddp_energy_std,
    eddp_energy_std_per_atom
export CellFeature,
    nfeatures,
    nfeatures_per_channel,
    nchannels,
    nbodyfeatures,
    feature_size,
    one_body_offset,
    suggest_rcut
export feature_names, two_body_channel, two_body_range, three_body_channel, three_body_range
export spec_index, spec_symbol, lookup
export RandomEmbeddedCellFeature, MultiRandomEmbeddedCellFeature
export FeatureProto, ModelArtifact, FeatureArtifact
export load_model,
    load_model_artifact,
    save_model_artifact,
    load_feature_json,
    load_feature_artifact,
    save_feature_json,
    load_from_jld2
export AbstractNNInterface,
    LinearInterface,
    FluxInterface,
    ManualFluxBackPropInterface,
    EnsembleNNInterface
export EDDPCalculator, load_calculator, predict, create_calculator, calculate!
export MetaCalculator,
    AbstractModifier,
    ExtraDimensionPenaltyModifier,
    VolumeConstraintModifier,
    AdditiveCalculatorModifier,
    get_modifier,
    get_modifier_parameter,
    set_modifier_parameter!
export CoreRepulsion, leaky_tanh
export clear_transient_gradients!, reinit, reinit!
export enable_eddp_timer!, disable_eddp_timer!, reset_eddp_timer!

end # module Potential

module Model

using ..Potential

export Lattice, reciprocal, cellmat, cellvecs, cellpar, wrap!, volume, frac_pos, lattice
export clip, Cell
export distance_matrix
export get_positions, set_positions!, set_cell!, set_cellmat!, remove_dimensions
export AbstractCalc, get_cell
export eddp_energy,
    eddp_forces,
    eddp_energy_per_atom,
    eddp_stress,
    eddp_pressure,
    eddp_pressure_gpa,
    eddp_pressure_GPa,
    eddp_enthalpy,
    eddp_energy_std,
    eddp_energy_std_per_atom
export CellFeature,
    nfeatures,
    nfeatures_per_channel,
    nchannels,
    nbodyfeatures,
    feature_size,
    one_body_offset,
    suggest_rcut
export feature_names, two_body_channel, two_body_range, three_body_channel, three_body_range
export spec_index, spec_symbol, lookup
export RandomEmbeddedCellFeature, MultiRandomEmbeddedCellFeature
export FeatureProto, ModelArtifact, FeatureArtifact
export load_model,
    load_model_artifact,
    save_model_artifact,
    load_feature_json,
    load_feature_artifact,
    save_feature_json,
    load_from_jld2
export AbstractNNInterface,
    LinearInterface,
    FluxInterface,
    ManualFluxBackPropInterface,
    EnsembleNNInterface
export EDDPCalculator, load_calculator, predict, create_calculator, calculate!
export MetaCalculator,
    AbstractModifier,
    ExtraDimensionPenaltyModifier,
    VolumeConstraintModifier,
    AdditiveCalculatorModifier,
    get_modifier,
    get_modifier_parameter,
    set_modifier_parameter!
export CoreRepulsion, leaky_tanh
export clear_transient_gradients!, reinit, reinit!
export enable_eddp_timer!, disable_eddp_timer!, reset_eddp_timer!

end # module Model

module Tools

using ..EDDPotentialsCore
import ..EDDPotentialsCore:
    LinearInterface,
    _force_update!,
    _stress_update!,
    compute_fv_gv,
    update_feature_vector!
using CellBase
using CellBase: rattle!, reduce, Composition
import CellBase: write_res
using LinearAlgebra
using NLSolversBase
using Printf
using ProgressMeter: @showprogress
using Base.Threads
using JLD2
using Dates
using UUIDs

function progress_callback! end

include("tools.jl")

export _ensure_dir,
    allclose,
    ensure_dir,
    fd_desc,
    force_gradient,
    get_label,
    gvec_index_transfer,
    is_unique_fvec,
    lj_baseline_calc,
    lj_baseline_feature,
    lj_baseline_model,
    lj_baseline_weights,
    lj_like_calc,
    progress_callback!,
    stem,
    stress_gradient,
    test_finite_difference,
    update_metadata!,
    write_res

end # module Tools

module Training

using ..EDDPotentialsCore
using ..EDDPotentialsCore.NNLS: nnls
import ..EDDPotentialsCore:
    AbstractCellFeatureSpec,
    AbstractNNInterface,
    EDDPCalculator,
    EnsembleNNInterface,
    ManualFluxBackPropInterface,
    predict,
    save_model_artifact,
    setup_fj,
    setup_fj_batched,
    setup_fj_batched_threaded
using ..Data
using ..Data: _split_group_labels
using ..Tools: ensure_dir
import ..Tools: progress_callback!
using Base.Threads
using Dates
using JLD2
using Glob
using Printf
using NLSolversBase
using ProgressMeter: Progress
using Parameters
using StatsBase: corspearman, mean, std
using Flux: Dense, Chain
import Flux
import Base
import CellBase
using CellBase: natoms
import Zygote
using Logging
using NonNegLeastSquares
using TimerOutputs

include("training/context.jl")
include("training/lmsolve.jl")
include("training/lmsolve_batched.jl")
include("training/sgd_training.jl")
include("training/training.jl")

export LevenbergMarquardt,
    BatchedLevenbergMarquardt,
    SGDState,
    AdamState,
    TrainingContext,
    TrainingResults,
    TrainedModel,
    create_ensemble,
    generate_chain,
    generate_f_g_optim,
    train!,
    train_lm!,
    train_lm_batched!,
    train_model,
    train_sgd!,
    progress_callback!,
    nnls_weights,
    predict_energy,
    rmse_per_atom,
    mae_per_atom,
    max_ae_per_atom,
    save_model_artifact

end # module Training

module Search

using ..EDDPotentialsCore
import ..EDDPotentialsCore:
    AbstractNNInterface,
    GPaToeVAng,
    eVAngToGPa,
    glob_allow_abs,
    _make_core_repulsion
using ..Simulation
using ..Tools: get_label, stem, update_metadata!, is_unique_fvec
import ..EDDPotentials: EDDPotentialsOption
using CellBase
using LinearAlgebra
import CellBase:
    add_dimensions,
    read_res,
    write_res,
    set_cellmat!,
    set_positions!,
    remove_dimensions

include("search/buildcell.jl")
include("search/transforms.jl")
include("search/hot.jl")
include("workflow/hyper.jl")
include("search/rss.jl")
include("search/minsep.jl")
include("search/model_structures.jl")

run_rss(args...; kwargs...) = _run_rss(args...; kwargs...)

export HotAIRSSOption,
    HotAIRSSResult,
    RssPruneOption,
    SepecieSeparation,
    bP,
    build_one,
    build_random_structures,
    build_and_relax,
    build_and_relax_one,
    check_energy_threshold!,
    compute_specie_separations,
    gather_minsep_stats,
    get_specie_pairs,
    hot_airss!,
    model_structures,
    rattle_cell!,
    run_rss,
    shake_res,
    scheduled_penalty_relax!,
    hyper_gen_relax,
    HyperRoundDiagnostics

end # module Search

module Workflow

using ..EDDPotentialsCore
import ..EDDPotentialsCore:
    CellFeature,
    CellEmbedding,
    EnsembleNNInterface,
    glob_allow_abs,
    load_model_artifact,
    save_model_artifact,
    _make_core_repulsion
using ..Data
using ..Training
using ..Search
import ..Search: run_rss, hyper_gen_relax
using ..Simulation
import ..Simulation: step!
using ..Tools
import ..EDDPotentials: EDDPotentialsOption
using Parameters
using JSON
using YAML
using TOML
using ArgParse
using Random
using UUIDs
using JLD2
using CellBase
import CellBase: write_res

include("link/link.jl")
include("workflow/hyper_builder.jl")

export AbstractTrainer,
    Builder,
    BuilderOption,
    BuilderState,
    CellFeatureConfig,
    EmbeddingOption,
    FormulaSamplingContext,
    ClientServerOption,
    MockCalculator,
    RssSetting,
    RssPruneOption,
    SchedulerConfig,
    TrainingOption,
    acrud,
    builder_short_uuid,
    builder_uuid,
    dataset_name,
    diagnose_formula_sampling,
    get_external_calculator,
    client_server_client,
    client_server_claim_training_slot!,
    client_server_collect_crud_results!,
    client_server_mark_training_slot_done!,
    client_server_mark_training_slot_failed!,
    client_server_master,
    client_server_prepare_crud_workdir!,
    client_server_prepare_training_phase!,
    client_server_train_slot,
    client_server_training_model_path,
    client_server_worker_states,
    client_server_write_heartbeat,
    has_ensemble,
    input_structure_dir,
    is_training_data_ready,
    link,
    link!,
    link_client_server,
    link_worker,
    load_ensemble,
    load_features,
    load_structures,
    load_training_dataset,
    make_symbol_keys,
    nstructures,
    nstructures_calculated,
    num_atoms,
    num_existing_models,
    output_structure_dir,
    run_rss,
    run_trainer,
    save_builder,
    split_then_featurize,
    step!,
    walk_forward_tests,
    write_dataset,
    write_model_structures,
    hyper_get_calc,
    hyper_gen_relax

end # module Workflow

module Analysis

using ..EDDPotentialsCore
import ..EDDPotentialsCore:
    ManualFluxBackPropInterface,
    glob_allow_abs,
    get_elemental_energy,
    eVAngToGPa
using ..Data
using ..Data:
    _load_structures_from_path,
    _parse_path_slice
using ..Training
using ..Workflow
using CellBase
import CellBase
using Printf
using LinearAlgebra
using StatsBase: mean, countmap
import Base

include("analysis/evaluation.jl")
include("analysis/records.jl")
include("analysis/plotting/recipes.jl")
include("analysis/lazy_res.jl")
include("analysis/plotting.jl")

export AbstractRecord,
    ShelxTITL,
    ShelxRecord,
    ComputedRecord,
    Simplex,
    PhaseDiagram,
    EvaluationConfig,
    EnergyMetrics,
    ForceMetrics,
    StressMetrics,
    LayerSaturationReport,
    AggregatedMetrics,
    benchmark_prediction,
    compare_energy,
    compare_forces,
    compare_stress,
    detect_saturation,
    energy_absolute_error,
    energy_ae_per_atom,
    evaluate,
    find_simplex,
    get_2d_plot_data,
    get_decomposition,
    get_e_above_hull,
    get_ternary_hulldata,
    load_evaluation_data,
    read_shelx_record,
    record_comp,
    record_energy,
    record_id,
    record_reduced_comp,
    plot_ternary,
    plot_binary,
    plot_binary_plotly,
    plot_minsep,
    plot_hist2d,
    plot_scatter,
    plot_std_vs_error,
    as_dataframe,
    select_refine

end # module Analysis

using .Tools
using .Training
using .Search
using .Workflow
using .Analysis
import .Tools:
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
