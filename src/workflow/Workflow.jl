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
    get_elemental_energy,
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
using LinearAlgebra: norm
using Unitful: ustrip
using Glob: glob
import CellBase: write_res

include("link/link.jl")
include("link/hyper_builder.jl")

export AbstractTrainer,
    Builder,
    BuilderOption,
    BuilderState,
    CellFeatureConfig,
    DISTRIBUTED_LOG_GROUP,
    EmbeddingOption,
    FormulaSamplingContext,
    DistributedLinkBackend,
    DistributedLinkOption,
    WorkerState,
    WORKER_BOOT,
    WORKER_IDLE,
    WORKER_WAIT,
    WORKER_DFT_RUNNING,
    WORKER_DFT_DONE,
    WORKER_DFT_FAILED,
    WORKER_TRAINING_RUNNING,
    WORKER_TRAINING_DONE,
    WORKER_STOPPING_CHILD,
    WORKER_STOPPED,
    WORKER_EXIT,
    MockCalculator,
    normalize_airsspy_res_file!,
    RssSetting,
    RssPruneOption,
    SchedulerConfig,
    TRAINING_DIR,
    TrainingOption,
    acrud,
    airsspy_crud_cmd,
    builder_short_uuid,
    builder_uuid,
    dataset_name,
    diagnose_formula_sampling,
    get_external_calculator,
    advance_worker!,
    distributed_worker,
    has_ensemble,
    input_structure_dir,
    is_training_data_ready,
    link,
    link!,
    link_distributed,
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
    read_worker_states,
    run_airsspy_crud,
    server_run_dft_phase!,
    server_run_training_phase!,
    run_rss,
    run_trainer,
    save_builder,
    server_collect_dft_results!,
    server_prepare_dft_phase!,
    server_prepare_training_phase!,
    _distributed_dft_cmd,
    _distributed_log_path,
    _distributed_training_cmd,
    _render_distributed_worker_job_script,
    _required_commands,
    _server_submit_workers,
    _server_request_phase_stop!,
    _server_wait_for_dft!,
    _server_wait_for_training!,
    _with_distributed_logger,
    _worker_handle_control!,
    _worker_run_training!,
    _worker_supervise_child!,
    split_then_featurize,
    step!,
    walk_forward_tests,
    validate_builder_for_link!,
    write_worker_heartbeat!,
    write_dataset,
    write_model_structures,
    hyper_get_calc,
    hyper_gen_relax

end # module Workflow
