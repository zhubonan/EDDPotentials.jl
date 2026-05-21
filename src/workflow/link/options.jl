using Configurations
using GarishPrint
import ..Potential: CellFeature, fr, gfr

"""
    BuilderState

    Holder of the state of the builder object used for the `link!` function
to perform various actions during the construction of the potentials.
    May be specified using a configuration file (`link.toml`) under the field
`state`.

## Attributes

- `seedfile`: Name of the seed file used for random structure generation. calculation. 
  may be specified by passing a sequence. Its content is passed to the `buildcell` program.
- `seedfile_weights`: Weights for randomly selecting each `seedfile` when multiple 
  seed files are used. Can be useful for setting the ratio of binary/ternary structures for
  a multi-composition search.
- `seedfile_calc`: The seed file needed for performing DFT calculations through the `crud.pl`
  (or equivalent) runners. A `cell` file should be a passed and a corresponding `param` should
  also exist when using CASTEP.
- `interation`: The main iteration number of the builder. Will be updated based on the exists of
  the trained models.
- `workdir`: Path to the working directory.
- `max_iterations`: The maximum number of iteration to be performed for building the potential.
- `per_generation`: Number of structures to write for each generation. When RSS pruning is enabled,
  more raw structures may be generated/relaxed internally to reach this kept-output count.
- `shake_per_generation`: The number of shaken structure for **each** relaxed structure to be generated.
- `buildcell_nprocs`: Number of concurrent `buildcell` processes to use during generation 0 bulk structure generation.
- `build_timeout`: Timeout in seconds for each `buildcell` process during generation 0 bulk structure generation.
- `shake_amp`: The amplitude of atomic displacements for shaking relaxed structures.
- `shake_cell_amp`: The amplitude of cell deformations for shaking relaxed structures.
- `n_parallel`: Deprecated; retained for compatibility with older configs.
- `mpinp`: The number of MPI tasks to run (e.g. the number of cores) for `crud.pl`.
- `n_initial`: The number of initial structures to be generated (for generation 0). 
- `dft_mode`: The model for local DFT calculations. Supported local options are `mock`, `pp3`, `crud`, and `acrud`.
  Scheduler-backed local DFT modes such as `crud-queue` and `disp-castep` are superseded by the distributed backend.
- `dft_kwargs`: The key word arguments to be used for the function that runs the DFT calculations.
- `rss_pressure_gpa`: The external pressure used when running random structure searching.
- `rss_pressure_gpa_range`: The range of the pressure used when running random structure searching.
- `rss_niggli_reduce`: Perform niggli reduce for the generated-relaxed structures.
- `rss_nprocs`: The number of processes to use for generating/relaxing random structures.
- `rss_external`: Whether to run the random structure searching in separate Julia processes.
- `rss_num_threads`: The number of threads to used for launched Julia processes.
- `rss_prune`: Optional pruning for generated relaxed structures before writing training inputs.
- `core_size`: Size of the soft sphere core. Note that the repulsion kick-in at about 3/4 of the core size.
- `ensemble_std_min`: The minimum ensemble standard deviation for valid structures during random structure searching.
  Can be used to exclude structures that are already well-described by the potential.
- `ensemble_std_max`: The maximum ensemble standard deviation for valid structures during random structure searching.
  Can be used to exclude pathological structures.
- `run_walk_forward`: Whether to run the walk-forward test or not.
- `project_prefix_override`: Deprecated project prefix override retained for compatibility with older configs.
- `builder_file_path`: Path to this builder file. This field will re automatically filled.
- `relax`: Options for performing relaxations of the generated structures.
- `elemental_energies`: The energies of elements species. It is **very important** to specify them for multi-composition searches.
   The reference energies may be obtained via *atom-in-a-large-cell* calculations or total energies of the pseudo atom calculation.
   The latter is printed in the `.castep` file for the on-the-fly generated potentials of CASTEP.
- `known_structure_dir`: Directory (relative to `workdir` if not absolute) containing known structures for generation 0.
  Files with `.res` are copied directly; `.cell` files are converted to `.res`.
- `formula_sampling_enabled`: Enable composition-space sampling by injecting `#FORMULA=...` into buildcell seed text.
- `formula_max_reduced_formula_coeff`: Maximum per-element coefficient in reduced-formula enumeration (e.g. `Li2TiO3` has max coeff `3`).
- `formula_list`: Optional explicit formula list (reduced and deduplicated) to sample from.
- `formula_target_atom_volumes`: Optional per-element target volumes used to derive `#VARVOL=...`
  automatically from the reduced formula by summing one entry per species present.
- `formula_oxidation_states`: Optional allowed oxidation states map for formula filtering (e.g. `Li=[1], O=[-2]`).
- `formula_require_charge_neutral`: If true, keep only formulas that can be charge-neutral under `formula_oxidation_states`.
"""
@option mutable struct BuilderState <: EDDPotentialsOption
    seedfile::Union{String,Vector{String}}
    seedfile_weights::Vector{Float64} = [1.0]
    seedfile_calc::String
    iteration::Int = 0
    workdir::String = "."
    max_iterations::Int = 5
    per_generation::Int = 100
    per_generation_threshold::Float64 = 0.7
    shake_per_minima::Int = 10
    buildcell_nprocs::Int = 1
    build_timeout::Float64 = 60.0
    shake_amp::Float64 = 0.02
    shake_cell_amp::Float64 = 0.02
    n_parallel::Int = 1
    mpinp::Int = 2
    n_initial::Int = 1000
    dft_mode::String = "crud"
    dft_kwargs::Dict{String,Any} = Dict{String,Any}()
    rss_pressure_gpa::Float64 = 0.1
    rss_pressure_gpa_range::Union{Nothing,Vector{Float64}} = nothing
    rss_niggli_reduce::Bool = true
    rss_nprocs::Int = 2
    rss_external::Bool = true
    rss_num_threads::Int = 1
    rss_prune::RssPruneOption = RssPruneOption()
    core_size::Union{Float64,Dict{String,Float64}} = 0.5
    ensemble_std_min::Float64 = 0.0
    ensemble_std_max::Float64 = -1.0
    run_walk_forward::Bool = true
    project_prefix_override::String = ""
    builder_file_path::String = ""
    relax::RelaxOption = RelaxOption()
    elemental_energies = Dict{String,Any}()
    known_structure_dir::String = "known_structures"
    formula_sampling_enabled::Bool = false
    formula_max_reduced_formula_coeff::Int = 6
    formula_list::Vector{String} = String[]
    formula_target_atom_volumes::Dict{String,Float64} = Dict{String,Float64}()
    formula_oxidation_states::Dict{String,Vector{Int}} = Dict{String,Vector{Int}}()
    formula_require_charge_neutral::Bool = true
    "Relaxation mode: `standard` (3D), `hyperspace`, or `hot`"
    relax_mode::String = "standard"
    "Number of MD+relax rounds for hot-AIRSS mode"
    hot_n_rounds::Int = 10
    "MD temperature (K) for hot-AIRSS mode"
    hot_md_temperature::Float64 = 300.0
    "Number of MD steps per round for hot-AIRSS mode"
    hot_md_steps::Int = 1000
    "MD timestep (fs) for hot-AIRSS mode"
    hot_md_timestep::Float64 = 0.5
    "Allow cell relaxation during hot-AIRSS"
    hot_relax_cell::Bool = true
    use_synthetic_force_data::Bool = false
    synthetic_force_n::Int = 1
    synthetic_force_max_disp::Float64 = 0.05
end

abstract type AbstractTrainer end

"""
    TrainerOption

    Options for training the EDDP potentials.

## Attributes

- `energy_threshold`: Threshold (per atom) energy to filter away high energy structures.
- `nmax`: Maximum number of neighbours when building the neighbour list.
- `nmodels`: The number of models to be trained for forming the ensemble through NNLS.
- `max_iter`: Maximum number of epochs (batched LM) or iterations (full LM) for training.
- `n_nodes`: The number of hidden nodes in each layer.
- `earlystop`: The number of none-improving iterations (on the validation set) to wait before early stopping the training.
- `show_progress`: Whether to show the progress of the training.
- `training_mode`: The mode of training. Currently, only `manual_backprop` and `linear` are supported.
- `training_kwargs`: The key word arguments to be used for the training function.
- `train_split`: The split ratios for the training set, in the order [train, validation, test].
- `respect_shakes`: Keep rattled/unrattled `-shake-N` families in the same split when preparing datasets.
- `use_validation_for_ensemble`: Whether to use the validation set for ensemble building.
- `save_each_model`: Deprecated no-op retained for compatibility with older configs.
- `p`: The power for computing the loss function for the Levernberg-Marquardt algorithm.
- `keep_best`: Whether to keep the best model during the training using the Levernberg-Marquardt algorithm.
- `log_file`: The file to store the training log.
- `prefix`: The prefix of the output files.
- `num_workers`: The number of workers to use for training.
- `num_threads_per_worker`: The number of threads to use for each worker.
- `type`: The type of potential to be trained. Currently, only `nn` and `nn_multi` are supported.
- `external`: Whether to launch external Julia processes for training. Link workflows require this to be true;
  `false` is retained only for compatibility with older configs and direct trainer calls.
- `external_mode`: External training backend. `local` launches local Julia worker processes.
  Scheduler-submitted local training modes are superseded by the distributed backend.
- `external_wait`: Whether to poll for trained model artifacts after command submission.
- `external_poll_interval`: Seconds between model-artifact polling checks.
- `external_timeout`: Maximum seconds to wait for external model artifacts. `0` means no timeout.
- `external_completion_ratio`: Fraction of `nmodels` required before external polling stops.
- `boltermann_kt`: The parameter for Boltzmann weighting for the loss function, set to positive to activate.
- `use_cuda`: Use GPU (CUDA) to accelerate the training.
- `cuda_visible_devices`: The CUDA visible devices to use.
- `ensemble_error_threshold`: The threshold for the error per atom for selecting observations to create the ensemble via NNLS.
  Observations with errors larger than this threshold will be excluded.
- `clean_models`: Whether to clean the models after training.
"""
@option mutable struct TrainingOption <: EDDPotentialsOption
    energy_threshold::Float64 = 10.0
    nmax::Int = 3000
    nmodels::Int = 256
    max_iter::Int = 300
    "number of hidden nodes in each layer"
    n_nodes::Vector{Int} = [8]
    earlystop::Int = 30
    show_progress::Bool = true
    training_mode::String = "manual_backprop"
    training_kwargs::Dict{String,Any} = Dict{String,Any}()
    train_split::Vector{Float64} = [0.8, 0.1, 0.1]
    respect_shakes::Bool = false
    use_validation_for_ensemble::Bool = true
    "Deprecated no-op retained for compatibility with older configs."
    save_each_model::Bool = true
    p::Float64 = 1.25
    keep_best::Bool = true
    log_file::String = ""
    prefix::String = ""
    num_workers::Int = 1
    num_threads_per_worker::Int = 1
    external = true
    "External training backend. Link workflows support `local`; scheduler-managed workers use the distributed backend."
    external_mode::String = "local"
    "Poll for trained model artifacts after submitting external commands."
    external_wait::Bool = true
    "Seconds between model-artifact polling checks."
    external_poll_interval::Float64 = 30.0
    "Maximum seconds to wait for external model artifacts. 0 means no timeout."
    external_timeout::Float64 = 0.0
    "Fraction of requested models required before external polling stops."
    external_completion_ratio::Float64 = 0.9
    boltzmann_kt::Float64 = -1.0
    clean_models::Bool = true
    use_cuda::Bool = false
    cuda_visible_devices::String = ""
    ensemble_error_threshold::Float64 = 0.5
    "Batch size for batched LM training. 0 means use full (non-batched) LM."
    batch_size::Int = 0
    "Training method: `lm` (full-batch LM), `lm_batched` (batched LM), `sgd` (SGD/Adam)"
    train_method::String = "lm"
    "SGD optimizer: `adam`, `sgd`, or `sgd_momentum`"
    optimizer::String = "adam"
    "Learning rate for SGD training"
    lr::Float64 = 1e-3
    "Momentum for SGD training"
    momentum::Float64 = 0.9
    "L2 weight decay regularization"
    weight_decay::Float64 = 0.0
    "Gradient clipping threshold (0.0 = no clipping)"
    grad_clip::Float64 = 1.0
end


"""
    RssSetting

    Options for running random structure searching.

## Attributes

- `packed`: Whether to pack the structures into a single file or not.
- `seedfile`: Name of the seed file used for random structure generation. May be specified by passing a sequence. Its content is passed to the `buildcell` program.
- `seedfile_weights`: Weights for randomly selecting each `seedfile` when multiple seed files are used. Can be useful for setting the ratio of binary/ternary structures for a multi-composition search.
- `ensemble_id`: The ID of the ensemble to be used for random structure searching, default is to use the last ensemble.
- `max`: The maximum number of structures to write for random structure searching. When pruning is
  enabled, more raw structures may be generated/relaxed internally to reach this kept-output count.
- `subfolder_name`: The name of the subfolder to store the generated structures.
- `show_progress`: Whether to show the progress of the random structure searching.
- `ensemble_std_max`: The maximum ensemble standard deviation for valid structures during random structure searching. Can be used to exclude pathological structures.
- `ensemble_std_min`: The minimum ensemble standard deviation for valid structures during random structure searching. Can be used to exclude structures that are already well-described by the potential.
- `eng_threshold`: The threshold for filtering structures based on their energy (per atom) referenced against the lowest energy structure.
- `niggli_reduce_output`: Whether to perform niggli reduce for the generated-relaxed structures.
- `max_err`: The maximum number of errors to be tolerated during random structure searching.
- `pressure_gpa`: The external pressure used (GPa) when running random structure searching.
- `relax`: Options for performing relaxations of the generated structures.
- `prune`: Optional pruning for generated relaxed structures before writing search output.
"""
@option mutable struct RssSetting <: EDDPotentialsOption
    packed::Bool = true
    seedfile::Union{String,Vector{String}} = "null"
    seedfile_weights::Vector{Float64} = Float64[1.0]
    ensemble_id::Int = -1
    max::Int = 1000
    subfolder_name::String = "search"
    show_progress::Bool = true
    ensemble_std_max::Float64 = 0.2
    ensemble_std_min::Float64 = -1.0
    eng_threshold::Float64 = -1.0
    niggli_reduce_output::Bool = true
    max_err::Int = 10
    pressure_gpa::Float64 = 0.01
    relax::RelaxOption = RelaxOption()
    prune::RssPruneOption = RssPruneOption()
end


"""
    CellFeatureConfig

    Configuration for generating cell features.



## Attributes

- `elements`: The elements to be considered for generating cell features.`
- `p2`: The power range for the second-order terms (start, stop, numbers).
- `p3`: The power range for the third-order terms (start, stop, numbers).
- `q3`: The power range for the third-order terms (start, stop, numbers).
- `geometry_sequence`: Whether to use the geometry sequence for generating cell features.
- `rcut2`: The cut-off radius for the second-order terms.
- `rcut3`: The cut-off radius for the third-order terms.
- `f2`: The function name for the second-order terms.
- `f3`: The function name for the third-order terms.
- `g2`: The function name for the second-order terms.
- `g3`: The function name for the third-order terms.
"""
@option mutable struct CellFeatureConfig <: EDDPotentialsOption
    elements::Vector{String}
    p2::Vector{Float64} = [2, 10, 5]
    p3::Vector{Float64} = [2, 10, 5]
    q3::Vector{Float64} = [2, 10, 5]
    geometry_sequence::Bool = true
    rcut2::Float64 = 6.0
    rcut3::Float64 = 6.0
    f2::String = "fr"
    f3::String = "fr"
    g2::String = "gfr"
    g3::String = "gfr"
end

"""
    CellFeature(config::CellFeatureConfig)

Construct a `CellFeature` from `CellFeatureConfig`.
"""
function CellFeature(config::CellFeatureConfig)
    args = (config.elements,)
    (; p2, p3, q3, geometry_sequence, rcut2, rcut3) = config
    config.f2 == "fr" ? f2 = fr : throw(ErrorException("Unknown option for f2!"))
    config.f3 == "fr" ? f3 = fr : throw(ErrorException("Unknown option for f3!"))
    config.g2 == "gfr" ? g2 = gfr : throw(ErrorException("Unknown option for f2!"))
    config.g3 == "gfr" ? g3 = gfr : throw(ErrorException("Unknown option for f2!"))

    kwargs = (; p2, p3, q3, geometry_sequence, rcut2, rcut3, f2, f3, g2, g3)
    CellFeature(args...; kwargs...)
end

"""
    EmbeddingOption

Options for embedding the cell features.
"""
@option struct EmbeddingOption <: EDDPotentialsOption
    m::Int = -1
    n::Int = -1
end

"""
    SchedulerConfig


Configuration for running jobs through a scheduler.
"""
@option struct SchedulerConfig <: EDDPotentialsOption
    script::String = "job_script.sh"
    njobs::Int = 1
    type::String = "SGE"
    clean_workdir::Bool = false
    submit_jobs::Bool = true
end

get_job_script_content(sch::SchedulerConfig) = read(sch.script, String)

"""
    DistributedLinkOption

Options for the file-backed distributed link server/worker workflow.
"""
@option mutable struct DistributedLinkOption <: EDDPotentialsOption
    enabled::Bool = false
    coordination_dir::String = ".eddp-distributed"
    control_poll_interval_seconds::Float64 = 10.0
    worker_poll_interval_seconds::Float64 = 10.0
    worker_heartbeat_interval_seconds::Float64 = 30.0
    worker_child_grace_seconds::Float64 = 60.0
    worker_child_kill_seconds::Float64 = 30.0
    dft_crud_workdir::String = "crud-work"
    "Seconds after which top-level DFT `.castep` output is considered stale."
    dft_stale_seconds::Float64 = 300.0
    dft_worker_jobs::Int = 1
    training_worker_jobs::Int = 1
    "Environment variables applied to distributed DFT child commands."
    dft_env::Dict{String,Any} = Dict{String,Any}()
    "Environment variables applied to distributed training child commands."
    training_env::Dict{String,Any} = Dict{String,Any}()
    "Deprecated no-op retained for compatibility with slot-based distributed training configs."
    training_slot_stale_seconds::Float64 = 86400.0
    manage_worker_jobs::Bool = true
    log_enabled::Bool = false
    log_level::String = "info"
    log_dir::String = "logs"
    log_interval_seconds::Float64 = 30.0
end

"""
    BuilderOption

The container for all options used for building the EDDP potentials used by the `link!` function.
"""
@option mutable struct BuilderOption <: EDDPotentialsOption
    state::BuilderState
    cf::CellFeatureConfig
    cf_embedding::Maybe{EmbeddingOption}
    rss::RssSetting
    trainer::TrainingOption
    scheduler::Maybe{SchedulerConfig}
    distributed::DistributedLinkOption = DistributedLinkOption()
end

BuilderOption(state, cf, cf_embedding, rss, trainer, scheduler) =
    BuilderOption(state, cf, cf_embedding, rss, trainer, scheduler, DistributedLinkOption())


Base.show(io::IO, mime::MIME"text/plain", x::EDDPotentialsOption) = pprint_struct(io, x)
