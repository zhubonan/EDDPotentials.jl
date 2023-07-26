using Configurations
using GarishPrint

@option mutable struct BuilderState <: EDDPOption
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
    dft_mode::String = "castep"
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
    "Run walk-forward test before re-training"
    run_walk_forward::Bool = true
    "Override the project_prefix"
    project_prefix_override::String = ""
    builder_file_path::String = ""
    relax::RelaxOption = RelaxOption()
end

abstract type AbstractTrainer end

@option mutable struct TrainingOption <: EDDPOption
    energy_threshold::Float64 = 10.0
    nmax::Int = 3000
    nmodels::Int = 256
    user_test_for_ensemble::Bool = true
    max_iter::Int = 300
    "number of hidden nodes in each layer"
    n_nodes::Vector{Int} = [8]
    earlystop::Int = 30
    show_progress::Bool = true
    "Store the data used for training in the archive"
    store_training_data::Bool = true
    rmse_threshold::Float64 = 0.5
    training_mode::String = "manual_backprop"
    training_kwargs::Dict{String,Any} = Dict{String,Any}()
    train_split::Vector{Float64} = [0.8, 0.1, 0.1]
    use_test_for_ensemble::Bool = true
    save_each_model::Bool = true
    p::Float64 = 1.25
    keep_best::Bool = true
    log_file::String = ""
    prefix::String = ""
    max_train::Int = 999
    "Number of workers to be launcher in parallel"
    num_workers::Int = 1
    "The number of threads per worker"
    num_threads_per_worker::Int = 1
    type::String
    external = true
    "Parameter for Boltzmann weighting, positive to activate"
    boltzmann_kt::Float64 = -1.0
end

@option mutable struct RssSetting <: EDDPOption
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


@option mutable struct CellFeatureConfig <: EDDPOption
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

@option struct EmbeddingOption <: EDDPOption
    m::Int = -1
    n::Int = -1
end

@option struct SchedulerConfig <: EDDPOption
    script::String = "job_script.sh"
    njobs::Int = 1
    type::String = "SGE"
    clean_workdir::Bool = false
    submit_jobs::Bool = true
end

get_job_script_content(sch::SchedulerConfig) = read(sch.script, String)
@option mutable struct BuilderOption <: EDDPOption
    state::BuilderState
    cf::CellFeatureConfig
    cf_embedding::Maybe{Embedding}
    rss::RssSetting
    trainer::TrainingOption
    scheduler::Maybe{SchedulerConfig}
end


Base.show(io::IO, mime::MIME"text/plain", x::EDDPOption) = pprint_struct(io, x)
