module Workflow

using ..Potential
import ..Potential:
    CellFeature,
    CellEmbedding,
    EnsembleNNInterface,
    glob_allow_abs,
    load_model_artifact,
    save_model_artifact,
    _make_core_repulsion
using ..Training
import ..Training: boltzmann
using ..Search
import ..Search: run_rss, hyper_gen_relax, _run_rss
using ..Simulation
import ..Simulation: step!
import ..Potential:
    ensure_dir,
    get_label,
    stem,
    swapext,
    update_metadata!
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
include("link/hyper_builder.jl")

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
