using Configurations
using GarishPrint

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
- `per_generation`: Number of structure to be generated (and relaxed) for each generation.
- `shake_per_generation`: The number of shaken structure for **each** relaxed structure to be generated.
- `build_timeout`: Timeout for assuming the `buildcell` file has failed to generate the structure.
- `shake_amp`: The amplitude of atomic displacements for shaking relaxed structures.
- `shake_cell_amp`: The amplitude of cell deformations for shaking relaxed structures.
- `n_parallel`: Deprecated.
- `mpinp`: The number of MPI tasks to run (e.g. the number of cores) for `crud.pl`.
- `n_initial`: The number of initial structures to be generated (for generation 0). 
- `dft_mode`: The model of running DFT calculations. The current available options are:crud, acrud, crud-queue, disp-castep, pp3.
- `dft_kwargs`: The key word arguments to be used for the function that runs the DFT calculations.
- `rss_pressure_gpa`: The external pressure used when running random structure searching.
- `rss_pressure_gpa_range`: The range of the pressure used when running random structure searching.
- `rss_niggli_reduce`: Perform niggli reduce for the generated-relaxed structures.
- `rss_nprocs`: The number of processes to use for generating/relaxing random structures.
- `rss_external`: Whether to run the random structure searching in separate Julia processes.
- `rss_num_threads`: The number of threads to used for launched Julia processes.
- `core_size`: Size of the soft sphere core. Note that the repulsion kick-in at about 3/4 of the core size.
- `ensemble_std_min`: The minimum ensemble standard deviation for valid structures during random structure searching.
  Can be used to exclude structures that are already well-described by the potential.
- `ensemble_std_max`: The maximum ensemble standard deviation for valid structures during random structure searching.
  Can be used to exclude pathological structures.
- `run_walk_forward`: Whether to run the walk-forward test or not.
- `project_prefix_override`: Override the prefix of the project. Used for `disp-castep` runner only.
- `project_prefix`: The prefix of the project. Used for `disp-castep` runner only.
- `builder_file_path`: Path to this builder file. This field will re automatically filled.
- `relax`: Options for performing relaxations of the generated structures.
- `elemental_energies`: The energies of elements species. It is **very important** to specify them for multi-composition searches.
  The reference energies may be obtained via *atom-in-a-large-cell* calculations or total energies of the pseudo atom calculation.
  The latter is printed in the `.castep` file for the on-the-fly generated potentials of CASTEP.
"""
@option mutable struct BuilderState <: EDDPotentialsOption
    seedfile::Union{String,Vector{String}}
    seedfile_weights::Vector{Float64} = [1.0]
    seedfile_calc::String
    iteration::Int = 0
    workdir::String = "."
    max_iterations::Int = 5
    per_generation::Int = 100
    per_generation_threshold::Float64 = 0.98
    shake_per_minima::Int = 10
    build_timeout::Float64 = 1.0
    shake_amp::Float64 = 0.02
    shake_cell_amp::Float64 = 0.02
    n_parallel::Int = 1
    mpinp::Int = 2
    n_initial::Int = 1000
    dft_mode::String = "crud-queue"
    dft_kwargs::Dict{String,Any} = Dict{String,Any}()
    rss_pressure_gpa::Float64 = 0.1
    rss_pressure_gpa_range::Union{Nothing,Vector{Float64}} = nothing
    rss_niggli_reduce::Bool = true
    rss_nprocs::Int = 2
    rss_external::Bool = true
    rss_num_threads::Int = 1
    core_size::Float64 = 1.0
    ensemble_std_min::Float64 = 0.0
    ensemble_std_max::Float64 = -1.0
    run_walk_forward::Bool = true
    project_prefix_override::String = ""
    builder_file_path::String = ""
    relax::RelaxOption = RelaxOption()
    elemental_energies = Dict{String,Any}()
end

abstract type AbstractTrainer end

"""
    TrainerOption

    Options for training the EDDP potentials.

## Attributes

- `energy_threshold`: Threshold (per atom) energy to filter away high energy structures.
- `nmax`: Maximum number of neighbours when building the neighbour list.
- `nmodels`: The number of models to be trained for forming the ensemble through NNLS.
- `max_iter`: Maximum number of iterations for training the potential.
- `n_nodes`: The number of hidden nodes in each layer.
- `earlystop`: The number of none-improving iterations (on the test set) to wait before early stopping the training.
- `show_progress`: Whether to show the progress of the training.
- `training_mode`: The mode of training. Currently, only `manual_backprop` and `linear` are supported.
- `training_kwargs`: The key word arguments to be used for the training function.
- `train_split`: The split of the training set for training, validation, and test set.
- `use_test_for_ensemble`: Whether to use the test set for ensemble building.
- `save_each_model`: Whether to save each trained model during the training (deprecated, only used for multi-thread training which is not efficient).
- `p`: The power for computing the loss function for the Levernberg-Marquardt algorithm.
- `keep_best`: Whether to keep the best model during the training using the Levernberg-Marquardt algorithm.
- `log_file`: The file to store the training log.
- `prefix`: The prefix of the output files.
- `num_workers`: The number of workers to use for training.
- `num_threads_per_worker`: The number of threads to use for each worker.
- `type`: The type of potential to be trained. Currently, only `nn` and `nn_multi` are supported.
- `external`: Whether to launch external Julia processes for training.
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
    use_test_for_ensemble::Bool = true
    save_each_model::Bool = true
    p::Float64 = 1.25
    keep_best::Bool = true
    log_file::String = ""
    prefix::String = ""
    num_workers::Int = 1
    num_threads_per_worker::Int = 1
    external = true
    boltzmann_kt::Float64 = -1.0
    clean_models::Bool = true
    use_cuda::Bool = false
    cuda_visible_devices::String = ""
    ensemble_error_threshold::Float64 = 0.5
end


"""
    RssSetting

    Options for running random structure searching.

## Attributes

- `packed`: Whether to pack the structures into a single file or not.
- `seedfile`: Name of the seed file used for random structure generation. May be specified by passing a sequence. Its content is passed to the `buildcell` program.
- `seedfile_weights`: Weights for randomly selecting each `seedfile` when multiple seed files are used. Can be useful for setting the ratio of binary/ternary structures for a multi-composition search.
- `ensemble_id`: The ID of the ensemble to be used for random structure searching, default is to use the last ensemble.
- `max`: The maximum number of structures to be generated for random structure searching.
- `subfolder_name`: The name of the subfolder to store the generated structures.
- `show_progress`: Whether to show the progress of the random structure searching.
- `ensemble_std_max`: The maximum ensemble standard deviation for valid structures during random structure searching. Can be used to exclude pathological structures.
- `ensemble_std_min`: The minimum ensemble standard deviation for valid structures during random structure searching. Can be used to exclude structures that are already well-described by the potential.
- `eng_threshold`: The threshold for filtering structures based on their energy (per atom) referenced against the lowest energy structure.
- `niggli_reduce_output`: Whether to perform niggli reduce for the generated-relaxed structures.
- `max_err`: The maximum number of errors to be tolerated during random structure searching.
- `pressure_gpa`: The external pressure used (GPa) when running random structure searching.
- `relax`: Options for performing relaxations of the generated structures.
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

Construct a `CellFeature` from `CellFeatureCOnfig`.
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
    BuilderOption

The container for all options used for building the EDDP potentials used by the `link!` function.
"""
@option mutable struct BuilderOption <: EDDPotentialsOption
    state::BuilderState
    cf::CellFeatureConfig
    cf_embedding::Maybe{Embedding}
    rss::RssSetting
    trainer::TrainingOption
    scheduler::Maybe{SchedulerConfig}
end


Base.show(io::IO, mime::MIME"text/plain", x::EDDPotentialsOption) = pprint_struct(io, x)
