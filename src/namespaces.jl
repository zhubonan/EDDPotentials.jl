# Logical namespace facades.
#
# These modules give callers and future source files stable dependency-oriented
# homes without changing the current top-level API. They intentionally re-export
# existing bindings; implementation files can be moved under these namespaces in
# later, smaller patches.

module Potential

using ..EDDPotentialsCore
using CellBase:
    reciprocal,
    clip,
    frac_pos
import ..EDDPotentialsCore:
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

module Training

import ..EDDPotentials:
    LevenbergMarquardt,
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
    nnls_weights,
    predict_energy,
    rmse_per_atom,
    mae_per_atom,
    max_ae_per_atom,
    save_model_artifact

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
    nnls_weights,
    predict_energy,
    rmse_per_atom,
    mae_per_atom,
    max_ae_per_atom,
    save_model_artifact

end # module Training

module Search

import ..EDDPotentials:
    HotAIRSSOption,
    HotAIRSSResult,
    RssPruneOption,
    SepecieSeparation,
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
    shake_res

export HotAIRSSOption,
    HotAIRSSResult,
    RssPruneOption,
    SepecieSeparation,
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
    shake_res

end # module Search

module Analysis

import ..EDDPotentials:
    AbstractRecord,
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

module Workflow

import ..EDDPotentials:
    AbstractTrainer,
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
    write_model_structures

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
    write_model_structures

end # module Workflow

module Tools

import ..EDDPotentials:
    _ensure_dir,
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
