#=
Routines for building structures
=#

"""
    build_one(seedfile;timeout=10, init_structure_transform=nothing)

Build a single structure via `buildcell`.
"""
function build_one(seedfile; timeout=10, init_structure_transform=nothing, max_attemp=100)
    local cell::Cell{Float64}
    i = 1
    while true
        if i >= max_attemp
            throw(ErrorException("Maximum attempt for building structure exceeded!"))
        end
        lines = open(seedfile, "r") do seed
            try
                cellout = read(
                    pipeline(`timeout $(timeout) buildcell`, stdin=seed, stderr=devnull),
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
`seedfile` argument.

"""
function build_random_structures(
    seedfile::Union{AbstractString,Vector},
    outdir;
    n=1,
    show_progress=false,
    timeout=60,
    outfmt="res",
    seedfile_weights=[1.0],
)
    i = 0
    if show_progress
        prog = Progress(n)
    end
    while i < n
        this_seed = _select_seed(seedfile, seedfile_weights)[1]
        cell = build_one(this_seed; timeout)
        label = EDDPotential.get_label(EDDPotential.stem(this_seed))
        cell.metadata[:label] = label
        if outfmt == "res"
            write_res(joinpath(outdir, "$(label).res"), cell)
        else
            CellBase.write_cell(joinpath(outdir, "$(label).cell"), cell)
        end
        i += 1
        if show_progress
            ProgressMeter.update!(prog)
        end
    end
end
