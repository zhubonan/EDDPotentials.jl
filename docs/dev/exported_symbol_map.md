# Exported Symbol Map

This map captures the public exports before the `Reexport`-based top-level
module wiring. It is intended as a migration reference for preserving
`using EDDPotentials` behavior while allowing submodules to own their exports.

## Submodule Exports

### Potential

`AbstractCalc`, `AbstractModifier`, `AbstractNNInterface`, `AdditiveCalculatorModifier`,
`Cell`, `CellFeature`, `CoreRepulsion`, `EDDPCalculator`, `EnsembleNNInterface`,
`ExtraDimensionPenaltyModifier`, `FeatureArtifact`, `FeatureProto`, `FluxInterface`,
`Lattice`, `LinearInterface`, `ManualFluxBackPropInterface`, `MetaCalculator`,
`ModelArtifact`, `MultiRandomEmbeddedCellFeature`, `Potential`,
`RandomEmbeddedCellFeature`, `VolumeConstraintModifier`, `_ensure_dir`, `allclose`,
`calculate!`, `cellmat`, `cellpar`, `cellvecs`, `clear_transient_gradients!`, `clip`,
`create_calculator`, `disable_eddp_timer!`, `distance_matrix`, `eddp_energy`,
`eddp_energy_per_atom`, `eddp_energy_std`, `eddp_energy_std_per_atom`,
`eddp_enthalpy`, `eddp_forces`, `eddp_pressure`, `eddp_pressure_GPa`,
`eddp_pressure_gpa`, `eddp_stress`, `enable_eddp_timer!`, `ensure_dir`, `fd_desc`,
`feature_names`, `feature_size`, `force_gradient`, `frac_pos`, `get_cell`,
`get_label`, `get_modifier`, `get_modifier_parameter`, `get_parameter`,
`get_positions`, `gvec_index_transfer`, `is_unique_fvec`, `lattice`, `leaky_tanh`,
`lj_baseline_calc`, `lj_baseline_feature`, `lj_baseline_model`,
`lj_baseline_weights`, `lj_like_calc`, `load_calculator`, `load_feature_artifact`,
`load_feature_json`, `load_from_jld2`, `load_model`, `load_model_artifact`,
`lookup`, `nbodyfeatures`, `nchannels`, `nfeatures`, `nfeatures_per_channel`,
`one_body_offset`, `post_calculate!`, `pre_calculate!`, `predict`, `reciprocal`,
`reinit`, `reinit!`, `remove_dimensions`, `requires_forces`, `reset_eddp_timer!`,
`save_feature_json`, `save_model_artifact`, `set_cell!`, `set_cellmat!`,
`set_modifier_parameter!`, `set_parameter!`, `set_positions!`, `spec_index`,
`spec_symbol`, `stem`, `stress_gradient`, `suggest_rcut`, `test_finite_difference`,
`three_body_channel`, `three_body_range`, `two_body_channel`, `two_body_range`,
`update_metadata!`, `volume`, `wrap!`, `write_res`.

### Training

`AdamState`, `BatchedLevenbergMarquardt`, `FeatureBatch`, `FeatureBatchIterator`,
`FeatureStore`, `LevenbergMarquardt`, `SGDState`, `StructureContainer`,
`TrainedModel`, `Training`, `TrainingContext`, `TrainingResults`,
`constant_lr`, `cosine_annealing_lr`, `create_ensemble`,
`each_batch`, `enthalpy_per_atom`, `generate_chain`, `generate_f_g_optim`,
`get_data`, `get_fit_data`, `load_batch`, `load_batch_H`, `load_feature_store`,
`mae_per_atom`, `max_ae_per_atom`, `nnls_weights`, `predict_energy`,
`progress_callback!`, `rmse_per_atom`, `save_feature_store`,
`save_feature_store_streaming`, `save_model_artifact`, `split_and_featurize`,
`standardize!`, `step_decay_lr`, `synthetic_force_data`, `train!`,
`train_lm!`, `train_lm_batched!`, `train_model`, `train_sgd!`,
`train_val_split`.

### Simulation

`AbstractOptimizer`, `AbstractStepConstraint`, `BerendsenNVT`, `FixAtoms`,
`FixCartesian`, `FixCell`, `FixCellConstraint`, `FixLinearIonic`, `FixRelative`,
`FixRigidGroup`, `FixScaled`, `FixVolumeConstraint`, `IonicConstraint`, `KB`,
`Langevin`, `MDIntegrator`, `MDRunner`, `MDState`, `NativeFIRE`, `NativeTPSD`,
`Relax`, `RelaxOption`, `RelaxResult`, `Simulation`, `TrajectoryFrame`,
`VelocityVerlet`, `ZeroTranslationConstraint`, `apply_force_constraints!`,
`apply_step_constraints!`, `apply_stress_constraints!`, `compute_kinetic_energy`,
`compute_temperature`, `converged`, `external_pressure_matrix`,
`external_stress_matrix`, `get_iteration`, `initialize!`, `instantaneous_pressure`,
`kinetic_energy`, `maxwell_boltzmann_velocities`, `multirelax!`, `opt_fire`,
`opt_tpsd`, `optimize!`, `relax!`, `rescale_velocities!`, `reset!`, `run!`,
`save_frame`, `save_trajectory`, `step!`, `temperature`, `time_fs`, `time_ps`,
`total_energy`, `zero_com_velocity!`.

### Search

`HotAIRSSOption`, `HotAIRSSResult`, `HyperRoundDiagnostics`, `RssPruneOption`,
`Search`, `SepecieSeparation`, `bP`, `build_and_relax`, `build_and_relax_one`,
`build_one`, `build_random_structures`, `check_energy_threshold!`,
`compute_specie_separations`, `gather_minsep_stats`, `get_specie_pairs`,
`hot_airss!`, `hyper_gen_relax`, `model_structures`, `rattle_cell!`, `run_rss`,
`scheduled_penalty_relax!`, `shake_res`.

### Workflow

`AbstractTrainer`, `Builder`, `BuilderOption`, `BuilderState`,
`CellFeatureConfig`, `ClientServerOption`, `EmbeddingOption`,
`FormulaSamplingContext`, `MockCalculator`, `RssPruneOption`, `RssSetting`,
`SchedulerConfig`, `TrainingOption`, `Workflow`, `acrud`, `builder_short_uuid`,
`builder_uuid`, `client_server_claim_training_slot!`, `client_server_client`,
`client_server_collect_crud_results!`, `client_server_mark_training_slot_done!`,
`client_server_mark_training_slot_failed!`, `client_server_master`,
`client_server_prepare_crud_workdir!`, `client_server_prepare_training_phase!`,
`client_server_train_slot`, `client_server_training_model_path`,
`client_server_worker_states`, `client_server_write_heartbeat`, `dataset_name`,
`diagnose_formula_sampling`, `get_external_calculator`, `has_ensemble`,
`hyper_gen_relax`, `hyper_get_calc`, `input_structure_dir`,
`is_training_data_ready`, `link`, `link!`, `link_client_server`, `link_worker`,
`load_ensemble`, `load_features`, `load_structures`, `load_training_dataset`,
`make_symbol_keys`, `nstructures`, `nstructures_calculated`, `num_atoms`,
`num_existing_models`, `output_structure_dir`, `run_rss`, `run_trainer`,
`save_builder`, `split_then_featurize`, `step!`, `walk_forward_tests`,
`write_dataset`, `write_model_structures`.

### Analysis

`AbstractRecord`, `AggregatedMetrics`, `Analysis`, `ComputedRecord`,
`EnergyMetrics`, `EvaluationConfig`, `ForceMetrics`, `LayerSaturationReport`,
`PhaseDiagram`, `ShelxRecord`, `ShelxTITL`, `Simplex`, `StressMetrics`,
`as_dataframe`, `benchmark_prediction`, `compare_energy`, `compare_forces`,
`compare_stress`, `detect_saturation`, `energy_absolute_error`,
`energy_ae_per_atom`, `enthalpyandvolume`, `enthalpyandvolume!`, `evaluate`,
`find_simplex`, `fitquality`, `fitquality!`, `get_2d_plot_data`,
`get_decomposition`, `get_e_above_hull`, `get_ternary_hulldata`, `inoutsample`,
`inoutsample!`, `load_evaluation_data`, `plot_binary`, `plot_binary_plotly`,
`plot_hist2d`, `plot_minsep`, `plot_scatter`, `plot_std_vs_error`,
`plot_ternary`, `read_shelx_record`, `record_comp`, `record_energy`, `record_id`,
`record_reduced_comp`, `relativeabsoluteerror`, `relativeabsoluteerror!`,
`select_refine`.

## Top-Level Reexport Policy

The top-level package reexports each working submodule wholesale:

```julia
@reexport using .Potential
@reexport using .Training
@reexport using .Simulation
@reexport using .Search
@reexport using .Workflow
@reexport using .Analysis
```

The practical ownership of the exported API is:

- `Potential`: calculator, feature, modifier, model-loading, and cell helper symbols.
- `Training`: training data containers, fit data helpers, training metrics, and model training entry points.
- `Simulation`: optimizer, relaxation, constraint, and molecular dynamics symbols.
- `Search`: Hot-AIRSS and hyperspace search symbols.
- `Workflow`: builder, link, client/server workflow, and workflow-owned RSS aliases.
- `Analysis`: evaluation, records, saturation, and comparison symbols.
- External packages: `CellBase` file I/O helpers, `Glob.glob`, and `StatsBase.mean`.

Ambiguous public symbols are currently identical bindings across their owners:
`RssPruneOption`, `hyper_gen_relax`, `run_rss`, `save_model_artifact`, and `step!`.
The namespace tests assert these aliases remain identical, so broad reexports do
not introduce binding conflicts.

## Internal Import Policy

The top-level module should not import module-specific internals. White-box tests
and implementation files import internals from the owning module directly, such as
`EDDPotentials.Potential.setup_fj`, `EDDPotentials.Training._parse_path_slice`,
`EDDPotentials.Search._rss_candidate`, `EDDPotentials.Workflow._run_rss_link`, or
`EDDPotentials.Analysis._extract_reference_energy`.
