#=
Routines for building structures
=#

using StatsBase: Weights, sample

"""
    build_one(seedfile;timeout=10, init_structure_transform=nothing, seed_text_transform=nothing)

Build a single structure via `buildcell`.
"""
function build_one(
    seedfile;
    timeout=10,
    init_structure_transform=nothing,
    max_attemp=100,
    seed_text_transform=nothing,
)
    local cell::Cell{Float64,3}
    i = 1
    seed_text = read(seedfile, String)
    while true
        if i >= max_attemp
            throw(ErrorException("Maximum attempt for building structure exceeded!"))
        end
        this_seed = isnothing(seed_text_transform) ? seed_text : seed_text_transform(seed_text)
        lines = try
            cellout = read(
                pipeline(
                    `timeout $(timeout) buildcell`,
                    stdin=IOBuffer(this_seed),
                    stderr=devnull,
                ),
                String,
            )
            split(cellout, "\n")
        catch err
            if typeof(err) <: ProcessFailedException
                @warn " `buildcell` failed to make the structure"
            else
                throw(err)
            end
        end
        isnothing(lines) && continue

        # Generate a unique label
        cell = CellBase.read_cell(lines)

        if !isnothing(init_structure_transform)
            cell = init_structure_transform(cell)
            if isnothing(cell)
                # This generated structure is no good....
                i += 1
                continue
            end
        end
        break
    end
    return cell
end


"""
    _select_seed(names::AbstractVector, weights::AbstractVector)

Select a random seed from a vector of names and expand glob pattern if needed, and the resolved paths.

# Examples 

Selecting seeds with pattern `Si-*.cell` with equal weightings.
```julia
_select_seed(["Si-*.cell"], [1.0])
```

"""
function _select_seed(
    names::AbstractVector,
    weights::AbstractVector=repeat([1.0], length(names)),
)

    actual_names = String[]
    actual_weights = Float64[]
    for (i, name) in enumerate(names)
        for j in glob_allow_abs(name)
            push!(actual_names, j)
            if i <= length(weights)
                i_ = i
            else
                i_ = length(weights)
            end
            push!(actual_weights, weights[i_])
        end
    end
    actual_weights ./= sum(actual_weights)
    @assert !isempty(actual_names) "No valid file found with $names"
    sample(actual_names, Weights(actual_weights)), actual_names
end

_select_seed(names::AbstractString, weights=[1.0]) = _select_seed([names], weights)



"""
    build_random_structures(seedfile, outdir;n=1, show_progress=false, timeout=60, seedfile_weights)

Build multiple random structures in the target folder. A glob pattern may be used for the
`seedfile` argument. When `nprocs > 1`, multiple `buildcell` subprocesses are launched
concurrently.

"""
function build_random_structures(
    seedfile::Union{AbstractString,Vector},
    outdir;
    n=1,
    show_progress=false,
    timeout=60,
    outfmt="res",
    seedfile_weights=[1.0],
    nprocs=1,
    seed_text_transform=nothing,
)
    if n <= 0
        return
    end

    if show_progress
        prog = Progress(n)
        prog_lock = ReentrantLock()
    end

    function build_and_write_one!()
        this_seed = _select_seed(seedfile, seedfile_weights)[1]
        cell = build_one(this_seed; timeout, seed_text_transform)
        label = get_label(stem(this_seed))
        cell.metadata[:label] = label
        if outfmt == "res"
            write_res(joinpath(outdir, "$(label).res"), cell)
        elseif outfmt == "cell"
            CellBase.write_cell(joinpath(outdir, "$(label).cell"), cell)
        else
            throw(ArgumentError("Unsupported output format: $outfmt"))
        end

        if show_progress
            lock(prog_lock) do
                ProgressMeter.next!(prog)
            end
        end
        return
    end

    nworkers = min(max(nprocs, 1), n)
    if nworkers == 1
        for _ = 1:n
            build_and_write_one!()
        end
        return
    end

    jobs = Channel{Int}(n)
    for i = 1:n
        put!(jobs, i)
    end
    close(jobs)

    tasks = [
        @async begin
            for _ in jobs
                build_and_write_one!()
            end
        end for _ = 1:nworkers
    ]

    foreach(fetch, tasks)
end
