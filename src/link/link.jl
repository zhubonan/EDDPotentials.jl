#=
Code for iteratively building the model
=#

module Link
import ..FeatureOptions
using Parameters
using JSON

const XT_NAME="xt"
const YT_NAME="yt"
const FEATURESPEC_NAME="cf"

"""
    BuildOPtions

Options controlling the iterative build process.

Mandatory Keyword arguments:
  - workdir: Working directory
  - seedfile: Path to the seed file for random structure generation
"""
@with_kw mutable struct BuildOptions
    iteration::Int=0
    workdir::String
    seedfile::String
    max_iterations::Int=5
    per_generation::Int=100
    shake_per_minima::Int=10
    build_timeout::Float64=1.
    shake_amp::Float64=0.02
    shake_cell_amp::Float64=0.02
    n_parallel::Int=1
    mpinp::Int=2
    n_initial::Int=1000
    datapaths::Vector{String}=[]
    dft_mode::String="castep"
    build_only::Bool = false
    relax_extra_opts::Dict{Symbol,Any} = Dict()
end

@with_kw struct TrainingOptions
    nmodels::Int=256
    max_iter::Int=300
    "number of hidden nodes in each layer"
    n_nodes::Vector{Int}=[8]
    yt_name::String=YT_NAME
    xt_name::String=XT_NAME
    featurespec_name::String=FEATURESPEC_NAME
    earlystop::Int=30
    show_progress::Bool=false
    "Store the data used for training in the archive"
    store_training_data::Bool=true
    rmse_threshold::Float64=0.5
end


function get_builder_uuid(workdir='.')
    fname = joinpath(workdir, ".eddp_builder")
    if isfile(fname)
        uuid = open(fname) do fh
            chomp(read(fh))
        end
    else 
        uuid = string(uuid4())
        open(fname, "w") do fh
            write(fh, uuid)
            write(fh, "\n")
        end
    end
    uuid
end


"""
    iterative_build(state::BuildOptions, feature_opts::FeatureOptions, training_opts::TrainingOptions) 

Iteratively build the model by repetitively train the model based on existing data,
perform searches, do DFT calculations to expand the training set.

Note that the speed of slows down significantly with increasing data point (qudratic at least).
Hence the training set needs to be selected carefully. 
"""
function link(state::BuildOptions, feature_opts::FeatureOptions, 
        training_opts::TrainingOptions
    )
    while state.iteration <= state.max_iterations
        step!(state;
        training_opts, 
        feature_opts
        )
    end
end

"""
Perform single point calculations using external tools (for training data)
"""
function run_relaxation(state::BuildOptions, indir, outdir)
    @info "Relaxation from $(indir)"
    if state.dft_mode == "castep"
        run_crud(state.workdir, indir, outdir;state.mpinp, state.n_parallel)
    elseif  state.dft_mode == "pp3"
        run_pp3_many(joinpath(state.workdir, ".pp3_work"), indir, outdir, state.seedfile;n_parallel=state.n_parallel)
    elseif  state.dft_mode == "disp-castep"
        run_disp_castep(indir, outdir, state.seedfile;state.relax_extra_opts...)
    else
        throw(ErrorException("Unknown dft_mode: $(state.dft_mode)"))
    end
end

"""
Run relaxation through DISP
"""
function run_disp_castep(indir, outdir, seedfile;categories, priority=90, project_prefix="eddp.jl",
                         monitor_only=false,
                         watch_every=60,
                         threshold=0.98,
                         kwargs...)
    file_pattern = joinpath(indir, "*.res")
    seed = splitext(seedfile)[1]
    # Setup the inputs
    project_name = joinpath(gethostname(), project_prefix, abspath(indir)[2:end])
    seed_stem = splitext(basename(seedfile))[1]
    cmd=`disp deploy singlepoint --seed $seed_stem --base-cell $seed.cell --param $seed.param --cell $file_pattern --project $project_name --priority $priority` 

    # Define the categories
    for category in categories
        push!(cmd.exec, "--category")
        push!(cmd.exec, category)
    end

    if !monitor_only
        @info "Command to be run $(cmd)"
        run(cmd)
    else
        @info "Not launching jobs - only watching for completion"
    end

    # Start to monitor the progress
    @info "Start watching for progress"
    sleep(1)
    while true
        cmd = `disp db summary --singlepoint --project $project_name --json`
        json_string = readchomp(pipeline(cmd))
        data = parse_disp_output(json_string)
        ncomp = get(data, "COMPLETED", 0) 
        nall = get(data, "ALL", -1)
        if ncomp / nall > threshold
            @info " $(ncomp)/$(nall) calculation completed - moving on ..."
            break
        else
            @info "Completed calculations: $(ncomp)/$(nall) - waiting ..."
        end
        sleep(watch_every)
    end
    # Pulling calculations down
    isdir(outdir) || mkdir(outdir)
    cmd = Cmd(`disp db retrieve-project --project $project_name`, dir=outdir)
    run(cmd)
    @info "Calculation results pulled into $outdir."
end


"""
    run_pp3_many(workdir, indir, outdir, seedfile; n_parallel=1, keep=false)

Use PP3 for singlepoint calculation - launch many calculations in parallel.
"""
function run_pp3_many(workdir, indir, outdir, seedfile; n_parallel=1, keep=false)
    files = glob(joinpath(indir, "*.res"))
    ensure_dir(workdir)
    for file in files
        working_path = joinpath(workdir, splitpath(file)[end])
        cp(file, working_path, force=true)
        try
            run_pp3(working_path, seedfile, joinpath(outdir, splitpath(file)[end]))
        catch error
            if typeof(error) <: ArgumentError
                @warn "Failed to calculate energy for $(file)!"
                continue
            end
            throw(error)
        end
        if !keep
            for suffix in [".cell", ".conv", "-out.cell", ".pp", ".res"]
                rm(swapext(working_path, suffix))
            end
        end
    end
end


"""
    run_pp3(file, seedfile, outpath)

Use `pp3` for single point calculations.
"""
function run_pp3(file, seedfile, outpath)
    if endswith(file, ".res")
        cell = CellBase.read_res(file)
        # Write as cell file
        CellBase.write_cell(swapext(file, ".cell"), cell)
    else
        cell = CellBase.read_cell(file)
    end
    # Copy the seed file
    cp(swapext(seedfile, ".pp"), swapext(file, ".pp"), force=true)
    # Run pp3 relax
    # Read enthalpy
    enthalpy = 0.0
    pressure = 0.0
    for line in eachline(pipeline(`pp3 -n $(splitext(file)[1])`))
        if contains(line, "Enthalpy")
            enthalpy = parse(Float64, split(line)[end])
        end
        if contains(line, "Pressure")
            pressure = parse(Float64, split(line)[end])
        end
    end
    # Write res
    cell.metadata[:enthalpy] = enthalpy
    cell.metadata[:pressure] = pressure
    cell.metadata[:label] = stem(file)
    CellBase.write_res(outpath, cell)
    cell
end


"""
    parse_disp_output(json_string)

Parse the output of `disp db summary`
"""
function parse_disp_output(json_string)
    tmp = JSON.parse(json_string)
    data = Dict{String, Int}()
    for (x, y) in tmp
        # How to unpack this way due to nested multi-index
        if contains(x, "RES")
            data["RES"] = first(values(first(values(y))))
        elseif contains(x, "ALL")
            data["ALL"] = first(values(first(values(y))))
        elseif contains(x, "COMPLETED")
            data["COMPLETED"] = first(values(first(values(y))))
        end
    end
    data
end


end # Link module