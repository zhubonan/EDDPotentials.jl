using JLD2
using StatsBase: fit, transform!, reconstruct!, ZScoreTransform
using Random
using Glob
using CellBase
using ExtXYZ
using LinearAlgebra: norm
import ProgressMeter
import CellBase: natoms
using CellBase:
    Cell,
    Composition,
    formula_and_factor,
    reduce,
    reduce_composition,
    species,
    volume
using ..Potential:
    AbstractCellFeatureSpec,
    CellFeature,
    GPaToeVAng,
    feature_size,
    feature_vector,
    get_elemental_energy,
    glob_allow_abs
using Base.Threads

export StructureContainer,
    FeatureStore,
    FeatureBatch,
    FeatureBatchIterator,
    each_batch,
    get_data,
    get_fit_data,
    load_batch,
    load_batch_H,
    save_feature_store,
    load_feature_store,
    save_feature_store_streaming,
    split_and_featurize,
    train_val_split,
    enthalpy_per_atom,
    standardize!,
    synthetic_force_data

_local_stem(path::AbstractString) = splitext(basename(path))[1]

"""
    StructureContainer{T,N,D}

Container for storing crystal structures and their energies for EDDP training.

A StructureContainer holds a collection of atomic structures (typically from DFT calculations)
along with their total energies. This is the first step in preparing data for training an
EDDP model. Structures can be loaded from files or created directly from Cell objects.

# Fields
- `paths::Vector{String}`: Labels/paths identifying each structure
- `H::Vector{N}`: Total enthalpies (energies) in eV for each structure
- `structures::Vector{Cell{T,D}}`: Crystal structure objects

# Constructors

## From file paths
```julia
StructureContainer(paths; threshold=10.0, pressure_gpa=1.0, select_func=minimum)
```
Load structures from .res or other CellBase-compatible files.

## From Cell objects
```julia
StructureContainer(structures, engs, labels; threshold=10.0, select_func=minimum)
```
Create directly from vectors of Cell objects and energies.

# Examples
```julia
# Load structures from AIRSS .res files
sc = StructureContainer("carbon_data/*.res")
println(length(sc), " structures loaded")

# Load with custom energy threshold
sc = StructureContainer("diamond.res"; threshold=5.0)

# Create from Cell objects directly
structures = [cell1, cell2, cell3]
energies = [-10.5, -10.3, -10.7]  # eV
labels = ["struct1", "struct2", "struct3"]
sc = StructureContainer(structures, energies, labels)
```

# Notes
- Structures with per-atom energy more than `threshold` eV above the minimum (per composition) are automatically excluded
- This helps remove high-energy unphysical structures from training data
- The `select_func` determines the reference (default: `minimum`)
- Supports glob patterns in file paths (e.g., "*.res")

# See also
[`FeatureStore`](@ref), [`CellBase.read_res_many`](@ref)
"""
struct StructureContainer{T,N,D}
    paths::Vector{String}
    "Enthalpy"
    H::Vector{N}
    structures::Vector{Cell{T,D}}
end

Base.length(v::StructureContainer) = length(v.H)

function _uniform_direction(rng, ndim::Integer)
    direction = randn(rng, ndim)
    normdir = norm(direction)
    while iszero(normdir)
        direction .= randn(rng, ndim)
        normdir = norm(direction)
    end
    direction ./= normdir
end

"""
    _parse_path_slice(path)

Parse `path@slice` syntax for partial extxyz reading.
Returns `(filepath, slice_string)` if a valid slice is found, otherwise `(path, nothing)`.

A valid slice must contain at least one `:` (e.g., `:2000`, `-1000:`, `::2`).
"""
function _parse_path_slice(path::AbstractString)
    # Find the last '@' that is followed by a slice-like pattern
    at_pos = findlast('@', path)
    if at_pos === nothing || at_pos == length(path)
        return (path, nothing)
    end
    slice_part = path[at_pos + 1:end]
    # A valid slice must contain ':'
    if !contains(slice_part, ':')
        return (path, nothing)
    end
    # Validate: should only contain digits, '-', and ':'
    if !all(c -> isdigit(c) || c == '-' || c == ':', slice_part)
        return (path, nothing)
    end
    filepath = path[1:at_pos - 1]
    # Filepath must look like a real path (not empty)
    if isempty(filepath)
        return (path, nothing)
    end
    return (filepath, slice_part)
end

"""
    _count_extxyz_frames(path)

Fast frame counter for extxyz files. Only reads the atom-count header of each frame,
skipping the actual data lines.
"""
function _count_extxyz_frames(path::AbstractString)
    count = 0
    open(path) do io
        while !eof(io)
            line = readline(io)
            strip(line) == "" && continue
            natoms = parse(Int, strip(line))
            for _ in 1:(natoms + 1)
                readline(io)
            end
            count += 1
        end
    end
    count
end

"""
    _python_slice_to_range(slice_str, total)

Convert a Python-style slice string (`"start:stop:step"`) to a Julia range,
given the total number of frames. Handles omitted parts and negative indices.

# Examples (with total=20000)
- `":2000"` → `1:2000` (first 2000)
- `"-1000:"` → `19001:20000` (last 1000)
- `"::2"` → `1:2:20000` (every 2nd)
- `"100:500:3"` → `101:3:500`
"""
function _python_slice_to_range(slice_str::AbstractString, total::Int)
    parts = split(slice_str, ':')
    # Parse start
    start_py = if length(parts) >= 1 && !isempty(parts[1])
        parse(Int, parts[1])
    else
        nothing
    end
    # Parse stop
    stop_py = if length(parts) >= 2 && !isempty(parts[2])
        parse(Int, parts[2])
    else
        nothing
    end
    # Parse step
    step_val = if length(parts) >= 3 && !isempty(parts[3])
        parse(Int, parts[3])
    else
        nothing
    end

    step = step_val === nothing ? 1 : step_val

    # Convert Python start (0-based) to Julia start (1-based inclusive)
    start_jl = if start_py === nothing
        step > 0 ? 1 : total
    elseif start_py >= 0
        min(start_py + 1, total)
    else
        max(total + start_py + 1, 1)
    end

    # Convert Python stop (0-based exclusive) to Julia stop (1-based inclusive)
    stop_jl = if stop_py === nothing
        step > 0 ? total : 1
    elseif stop_py >= 0
        min(stop_py, total)
    else
        max(total + stop_py + 1, 1)
    end

    start_jl:step:stop_jl
end

"""
    _extxyz_dict_to_cell(data::Dict)

Convert a raw ExtXYZ frame dict (as returned by `ExtXYZ.read_frames`) to a `Cell`.
"""
function _extxyz_dict_to_cell(data::Dict)
    metadata = Dict{Symbol,Any}()
    for (key, value) in data["info"]
        metadata[Symbol(key)] = value
    end
    arrays = Dict{Symbol,AbstractArray}()
    for (key, value) in data["arrays"]
        if !(key in ("species", "pos"))
            arrays[Symbol(key)] = value
        end
    end
    cell = Cell(
        Lattice(collect(transpose(data["cell"]))),
        Symbol.(data["arrays"]["species"]),
        data["arrays"]["pos"],
    )
    cell.arrays = arrays
    cell.metadata = metadata
    cell
end

function _normalize_extxyz_cell!(cell, fallback_label)
    metadata = cell.metadata
    if !haskey(metadata, :enthalpy) && haskey(metadata, :energy)
        metadata[:enthalpy] = metadata[:energy]
    end
    if !haskey(metadata, :label)
        metadata[:label] = haskey(metadata, :name) ? string(metadata[:name]) : fallback_label
    end
    if !haskey(cell.arrays, :REF_forces)
        if haskey(cell.arrays, :forces)
            cell.arrays[:REF_forces] = cell.arrays[:forces]
        elseif haskey(cell.arrays, :force)
            cell.arrays[:REF_forces] = cell.arrays[:force]
        end
    end
    metadata[:volume] = get(metadata, :volume, volume(cell))
    cell
end

function _load_structures_from_path(path::AbstractString)
    filepath, slice_str = _parse_path_slice(path)
    ext = lowercase(splitext(filepath)[2])
    if ext in (".xyz", ".extxyz")
        cells = Cell[]
        if slice_str !== nothing
            total = _count_extxyz_frames(filepath)
            frame_range = _python_slice_to_range(slice_str, total)
            # ExtXYZ.read_frames only supports contiguous ranges.
            # Read the full contiguous span, then subsample for step != 1.
            contiguous = first(frame_range):last(frame_range)
            raw_dicts = ExtXYZ.read_frames(filepath, contiguous)
            indices = frame_range .- first(frame_range) .+ 1
            for (i, idx) in enumerate(indices)
                push!(cells, _normalize_extxyz_cell!(_extxyz_dict_to_cell(raw_dicts[idx]), "$(_local_stem(filepath))-$i"))
            end
            return cells
        end
        loaded = ExtXYZ.load(filepath)
        systems = loaded isa AbstractVector ? loaded : [loaded]
        for (i, system) in enumerate(systems)
            push!(cells, _normalize_extxyz_cell!(Cell(system), "$(_local_stem(filepath))-$i"))
        end
        return cells
    end
    CellBase.read_res_many(filepath)
end

function _structure_enthalpy(cell)
    if haskey(cell.metadata, :enthalpy)
        return cell.metadata[:enthalpy]
    elseif haskey(cell.metadata, :energy)
        return cell.metadata[:energy]
    end
    throw(ArgumentError("Structure `$(cell.metadata[:label])` is missing `:enthalpy`/`:energy` metadata."))
end

function StructureContainer(
    structures::Vector{T},
    engs,
    labels=["structure_$(i)" for i = 1:length(structures)];
    threshold=10.0,
    select_func=minimum,
) where {T<:Cell}
    H = engs
    Ha = H ./ natoms.(structures)
    mask = _select_per_atom_threshold(structures, Ha; select_func, threshold)
    for s in structures
        if !haskey(s.arrays, :REF_forces)
            if haskey(s.arrays, :forces)
                s.arrays[:REF_forces] = s.arrays[:forces]
            elseif haskey(s.arrays, :force)
                s.arrays[:REF_forces] = s.arrays[:force]
            end
        end
    end
    StructureContainer(labels[mask], H[mask], structures[mask])
end

"""
    StructureContainer(paths::Vector)

Args:

    - `threshold`: structures with per-atom enthalpy higher than this are excluded. 
      Relative to the median energy.
    - `pressure_gpa`: Pressure under which the enthalpy is calculated. Defaults to 1.0 GPa.
"""
function StructureContainer(
    paths::Vector;
    threshold=10.0,
    pressure_gpa=1.0,
    select_func=minimum,
)
    resolved_paths = String[]
    for path in paths
        filepath, _ = _parse_path_slice(path)
        if contains(filepath, "*") || contains(filepath, "?")
            # Glob on the file path, keep the full path (with @slice) for loading
            for resolved in glob_allow_abs(filepath)
                # Re-attach the slice spec if present
                _, slice_str = _parse_path_slice(path)
                full = slice_str !== nothing ? "$(resolved)@$(slice_str)" : resolved
                push!(resolved_paths, full)
            end
        else
            push!(resolved_paths, path)
        end
    end
    tmp = []
    actual_labels = String[]
    for path in resolved_paths
        vres = _load_structures_from_path(path)
        append!(tmp, vres)
        append!(actual_labels, map(x -> x.metadata[:label], vres))
    end

    structures = typeof(tmp[1])[x for x in tmp]
    H = [
        _structure_enthalpy(cell) + get(cell.metadata, :volume, volume(cell)) * GPaToeVAng(pressure_gpa) for
        cell in structures
    ]
    Ha = H ./ natoms.(structures)
    mask = _select_per_atom_threshold(structures, Ha; select_func, threshold)
    H_0Pa = [structures[i].metadata[:enthalpy] for i in mask]
    StructureContainer(actual_labels[mask], H_0Pa, structures[mask])
end

StructureContainer(path::AbstractString; kwargs...) =
    StructureContainer([path]; kwargs...)

"""
    synthetic_force_data(sc::StructureContainer; kwargs...)

Generate synthetic energy-only data by uniformly rattling each structure and
projecting the parent forces onto the applied displacements using a first-order
Taylor expansion.

# Keyword Arguments
- `nsynthetic::Int=1`: Number of synthetic structures to generate per parent
- `max_disp::Real=0.05`: Maximum displacement magnitude per atom in Cartesian coordinates
- `force_key::Symbol=:REF_forces`: `cell.arrays` key used to read the parent force matrix
- `rng=Random.default_rng()`: Random number generator
- `label_suffix="-force"`: Suffix appended to the parent label before the shake index
- `keep_original=true`: Whether to include the original structures in the returned container
- `copy_metadata=true`: Whether to copy parent metadata onto synthetic structures
- `skip_missing=false`: If true, silently skip structures that lack the force key
"""
function synthetic_force_data(
    sc::StructureContainer;
    nsynthetic::Int=1,
    max_disp::Real=0.05,
    force_key::Symbol=:REF_forces,
    rng=Random.default_rng(),
    label_suffix="-force",
    keep_original=true,
    copy_metadata=true,
    skip_missing=false,
)
    nsynthetic > 0 || throw(ArgumentError("`nsynthetic` must be positive, got $nsynthetic."))
    max_disp >= 0 || throw(ArgumentError("`max_disp` must be non-negative, got $max_disp."))

    paths = String[]
    H = eltype(sc.H)[]
    structures = eltype(sc.structures)[]

    if keep_original
        append!(paths, sc.paths)
        append!(H, sc.H)
        append!(structures, sc.structures)
    end

    for i in eachindex(sc.structures)
        parent = sc.structures[i]
        parent_label = sc.paths[i]
        parent_energy = sc.H[i]

        if !haskey(parent.arrays, force_key)
            skip_missing && continue
            throw(ArgumentError("Structure `$parent_label` is missing force array key `$(force_key)`."))
        end

        forces = parent.arrays[force_key]
        size(forces) == size(get_positions(parent)) || throw(
            ArgumentError(
                "Structure `$parent_label` has force array of size $(size(forces)), expected $(size(get_positions(parent))).",
            ),
        )

        for isynth = 1:nsynthetic
            synthetic = deepcopy(parent)
            disp = zeros(eltype(get_positions(parent)), size(get_positions(parent)))
            for iat in axes(disp, 2)
                direction = _uniform_direction(rng, size(disp, 1))
                magnitude = rand(rng) * max_disp
                disp[:, iat] .= direction .* magnitude
            end

            set_positions!(synthetic, get_positions(parent) .+ disp)
            energy_delta = -sum(forces .* disp)
            synthetic_energy = parent_energy + energy_delta
            label = "$(parent_label)$(label_suffix)-shake-$(isynth)"

            metadata = copy_metadata ? copy(parent.metadata) : Dict{Symbol,Any}()
            metadata[:label] = label
            metadata[:synthetic] = true
            metadata[:synthetic_parent_label] = parent_label
            metadata[:synthetic_parent_energy] = parent_energy
            metadata[:synthetic_energy_delta] = energy_delta
            metadata[:synthetic_displacement_norm] = norm(disp)
            metadata[:synthetic_max_displacement] = isempty(disp) ? zero(max_disp) :
                                                    maximum(norm.(eachcol(disp)))
            synthetic.metadata = metadata

            push!(paths, label)
            push!(H, synthetic_energy)
            push!(structures, synthetic)
        end
    end

    StructureContainer(paths, H, structures)
end


"""
    _idx_group_by_composition(structures)

Return the index of structures of each unique composition.
"""
function _idx_group_by_composition(structures)
    reduced_comps = reduce_composition.(Composition.(structures))
    unique_comp = unique(reduced_comps)
    out = Dict{Composition,Vector{Int}}()
    for comp in unique_comp
        idx = findall(x -> x == comp, reduced_comps)
        out[comp] = idx
    end
    out
end

"""
    _select_per_atom_threshold(structures, Ha; select_func=minimum, threshold=10.0)

Return index selected based on per-formula atomic energy
"""
function _select_per_atom_threshold(structures, Ha; select_func=minimum, threshold=10.0)
    reduced_comps = reduce_composition.(Composition.(structures))
    unique_comp = unique(reduced_comps)
    selected_idx = Int[]
    for comp in unique_comp
        idx = findall(x -> x == comp, reduced_comps)
        refval = select_func(Ha[idx])
        selected = filter(x -> Ha[x] < refval + threshold, idx)
        append!(selected_idx, selected)
    end
    selected_idx
end

_select_per_atom_threshold(sc; select_func=minimum, threshold=10.0) =
    sc[_select_per_atom_threshold(
        sc.structures,
        sc.H ./ natoms(sc);
        select_func,
        threshold,
    )]


"""
Split a vector by integer numbers
"""
function _split_vector(c, nsplit::Vararg{Int}; shuffle=true, seed=42)
    out = []
    rng = MersenneTwister(seed)
    if shuffle
        perm = randperm(rng, length(c))
    else
        perm = collect(1:length(c))
    end
    i = 1
    for n in nsplit
        push!(out, c[perm[i:n-1+i]])
        i += n
    end
    Tuple(out)
end

"""
Split a vector by either integer counts or ratio values.
"""
function _split_vector(c, nsplit::Vararg{Real}; shuffle=true, seed=42)
    counts = Int[]
    total = length(c)
    ratio_indices = Int[]
    ratio_remainders = Float64[]
    for (i, n) in enumerate(nsplit)
        if isinteger(n)
            push!(counts, Int(round(n)))
        elseif 0.0 <= n <= 1.0
            raw = total * n
            push!(counts, Int(floor(raw)))
            push!(ratio_indices, i)
            push!(ratio_remainders, raw - floor(raw))
        else
            error(
                "Invalid split value $n. Use integer counts or ratios in [0, 1].",
            )
        end
    end
    remainder = total - sum(counts)
    full_ratio_partition = length(ratio_indices) == length(nsplit) &&
                           isapprox(sum(Float64, nsplit), 1.0)
    if remainder > 0 && full_ratio_partition
        order = sortperm(ratio_remainders; rev=true)
        for j in 1:remainder
            counts[ratio_indices[order[mod1(j, length(order))]]] += 1
        end
    elseif remainder < 0
        error("Split counts $(counts) exceed container length $total.")
    end
    _split_vector(c, counts...; shuffle, seed)
end

function _shake_parent_label(label::AbstractString, metadata=nothing)
    if metadata !== nothing && haskey(metadata, :synthetic_parent_label)
        return string(metadata[:synthetic_parent_label])
    end
    result = match(r"^(.+?)(?:-force)?-shake-\d+$", label)
    return isnothing(result) ? String(label) : result.captures[1]
end

function _split_group_labels(c::StructureContainer)
    [
        _shake_parent_label(c.paths[i], c.structures[i].metadata)
        for i in eachindex(c.paths)
    ]
end

"""
    FeatureStore

Unified container for feature data supporting both in-memory and disk-backed storage
with a DataLoader-style interface for batched iteration.

A `FeatureStore` holds feature vectors computed from atomic structures along with
target energies and metadata. It operates in two modes:

- **In-memory** (`data !== nothing`): All feature matrices are held in a `Vector{Matrix}`.
  Supports splitting, slicing, concatenation, and full random access.
- **Disk-backed** (`data === nothing`): Features are stored in a JLD2 file organised in
  fixed consecutive batches, loaded on demand with per-batch caching. Supports indexed
  access, iteration, and `each_batch` for I/O-efficient training.

# Access patterns

| Method | Returns | Use case |
|--------|---------|----------|
| `store[i]` | `Matrix{Float64}` | Per-structure feature matrix (LM) |
| `for mat in store` | yields `Matrix{Float64}` | Iterate all structures |
| `get_data(store)` | `Vector{Matrix{Float64}}` | Full materialization |
| `each_batch(store; batch_size)` | iterator of `FeatureBatch` | Batched training (SGD, batched LM) |
| `get_fit_data(store)` | `(x, H)` | Training dispatch |

# Constructors

```julia
FeatureStore(sc::StructureContainer, feature::CellFeature; kwargs...)  # in-memory
FeatureStore(data::Vector{Matrix}, H, labels, metadata, feature)       # from raw data
load_feature_store(path)                                                # disk-backed
```

# See also
`FeatureBatch`, `each_batch`, `get_data`, `save_feature_store`, `load_feature_store`
"""
mutable struct FeatureStore
    H::Vector{Float64}
    labels::Vector{String}
    metadata::Vector{Dict{Symbol,Any}}
    feature::Any
    xt::Any
    yt::Any
    elemental_energies::Dict{Symbol,Float64}
    sizes::Vector{Int}
    data::Union{Vector{Matrix{Float64}},Nothing}
    path::String
    n_disk_batches::Int
    disk_batch_size::Int
    disk_batch_ranges::Vector{UnitRange{Int}}
    cached_batch_id::Int
    cached_batch::Vector{Matrix{Float64}}
    _cache_lock::ReentrantLock
end

function _split_group_labels(c::FeatureStore)
    [
        _shake_parent_label(c.labels[i], c.metadata[i])
        for i in eachindex(c.labels)
    ]
end


function Base.show(io::IO, o::MIME"text/plain", v::StructureContainer)
    ls = length.(v.structures)
    size_max = length(v.structures[findfirst(x -> x == maximum(ls), ls)])
    size_min = length(v.structures[findfirst(x -> x == minimum(ls), ls)])

    println(io, "StructureContainer:")
    println(io, "  $(length(v)) structures ")
    println(io, "  Max size: $(size_max)")
    println(io, "  Min size: $(size_min)")
    println(io, "  Max enthalpy_per_atom: $(maximum(enthalpy_per_atom(v)))")
    print(io, "  Min enthalpy_per_atom: $(minimum(enthalpy_per_atom(v)))")
end


function Base.show(io::IO, o::MIME"text/plain", v::FeatureStore)
    mode = v.data !== nothing ? "in-memory" : "disk-backed ($(v.path))"
    println(io, "FeatureStore ($mode):")
    println(io, "  $(length(v)) data points")
    println(io, "  Max atoms: $(maximum(v.sizes)), Min atoms: $(minimum(v.sizes))")
    if v.data === nothing && v.n_disk_batches > 0
        println(io, "  $(v.n_disk_batches) disk batches (batch_size=$(v.disk_batch_size))")
    end
    println(io, "With feature spec:")
    show(io, o, v.feature)
end

Base.show(io::IO, v::FeatureStore) = Base.show(io, MIME("text/plain"), v)


Base.length(v::FeatureStore) = length(v.H)

function FeatureStore(
    sc::StructureContainer,
    feature::AbstractCellFeatureSpec;
    nmax=500,
    show_progress=true,
    elemental_energies=Dict{Symbol,Float64}(),
    kwargs...,
)
    fvecs = Vector{Matrix{Float64}}(undef, length(sc))
    H = copy(sc.H)
    labels = collect(String, sc.paths)
    metadata = [cell.metadata for cell in sc.structures]
    if show_progress
        p = ProgressMeter.Progress(length(sc))
    end
    jj = Atomic{Int}(0)
    l = ReentrantLock()
    Threads.@threads for i = 1:length(sc)
        fvecs[i] = feature_vector(feature, sc.structures[i]; nmax, kwargs...)
        m = metadata[i]
        form, nf = CellBase.formula_and_factor(sc.structures[i])
        m[:formula] = form
        m[:nformula] = nf
        Threads.atomic_add!(jj, 1)
        if show_progress
            lock(l) do
                ProgressMeter.update!(p, jj[])
            end
        end
    end
    if length(elemental_energies) > 0
        for i = 1:length(sc)
            H[i] -= get_elemental_energy(sc.structures[i], elemental_energies)
        end
    end
    sizes = [size(fv, 2) for fv in fvecs]
    FeatureStore(
        H, labels, metadata, feature, nothing, nothing,
        Dict{Symbol,Float64}(elemental_energies), sizes,
        fvecs, "", 0, 0, UnitRange{Int}[], -1, Matrix{Float64}[], ReentrantLock(),
    )
end

function FeatureStore(sc::StructureContainer; cf_kwargs=NamedTuple(), kwargs...)
    symbols = reduce(vcat, unique.(species.(sc.structures)))
    FeatureStore(
        sc::StructureContainer,
        CellFeature(unique(symbols); cf_kwargs...);
        kwargs...,
    )
end

function FeatureStore(
    data::Vector{<:AbstractMatrix{<:AbstractFloat}},
    H::AbstractVector{<:Real},
    labels::AbstractVector{<:AbstractString},
    metadata::AbstractVector{<:AbstractDict},
    feature;
    elemental_energies::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
    xt=nothing,
    yt=nothing,
)
    sizes = [size(d, 2) for d in data]
    FeatureStore(
        Vector{Float64}(H), Vector{String}(labels),
        Vector{Dict{Symbol,Any}}(metadata), feature, xt, yt,
        elemental_energies, sizes,
        Vector{Matrix{Float64}}(data),
        "", 0, 0, UnitRange{Int}[], -1, Matrix{Float64}[], ReentrantLock(),
    )
end

Base.IndexStyle(::Type{<:StructureContainer}) = IndexLinear()

Base.iterate(v::StructureContainer, state=1) =
    state > length(v) ? nothing : (v.structures[state], state + 1)
Base.eltype(::Type{StructureContainer{T,N,D}}) where {T,N,D} = Cell{T,D}

Base.getindex(v::StructureContainer, i::Int) = v.structures[i]

function Base.getindex(
    v::StructureContainer,
    idx::Union{UnitRange,Vector{T}},
) where {T<:Int}
    structures = v.structures[idx]
    paths = v.paths[idx]
    H = v.H[idx]
    StructureContainer(paths, H, structures)
end

Base.IndexStyle(::Type{FeatureStore}) = IndexLinear()

function Base.iterate(v::FeatureStore, state=1)
    state > length(v) ? nothing : (v[state], state + 1)
end
Base.eltype(::Type{FeatureStore}) = Matrix{Float64}
Base.broadcastable(fs::FeatureStore) = fs

function Base.getindex(s::FeatureStore, i::Int)
    if s.data !== nothing
        return s.data[i]
    end
    bid = cld(i, s.disk_batch_size)
    batch = load_batch(s, bid)
    local_idx = i - first(s.disk_batch_ranges[bid]) + 1
    batch[local_idx]
end

function Base.getindex(s::FeatureStore, idx::Union{UnitRange{Int},Vector{Int}})
    if s.data !== nothing
        return _in_memory_slice(s, idx)
    end
    error(
        "Indexing a disk-backed FeatureStore with a range is not supported — " *
        "it would require materializing data from disk. Use an in-memory store " *
        "or pre-split your data before saving to disk."
    )
end

function _in_memory_slice(s::FeatureStore, idx)
    data = s.data[idx]
    sizes = s.sizes[idx]
    FeatureStore(
        s.H[idx], s.labels[idx], s.metadata[idx],
        s.feature, s.xt, s.yt, s.elemental_energies,
        sizes, data, "", 0, 0, UnitRange{Int}[], -1, Matrix{Float64}[], ReentrantLock(),
    )
end
function _select_by_label(all_labels, labels)
    id_selected = Int[]
    for val in labels
        id_temp = findfirst(x -> x == val, all_labels)
        if isnothing(id_temp)
            @warn "$(val) is not found and hence skipped!"
            continue
        end
        if id_temp in id_selected
            @warn "$(val) is duplicated - indexing by label can be seriously broken!"
        end
        push!(id_selected, id_temp)
    end
    id_selected
end


function Base.getindex(v::StructureContainer, idx::Vector{T}) where {T<:AbstractString}
    labels = [x.metadata[:label] for x in v.structures]
    v[_select_by_label(labels, idx)]
end

function Base.getindex(v::FeatureStore, idx::Vector{T}) where {T<:AbstractString}
    v[_select_by_label(v.labels, idx)]
end


"""
    train_val_split(v::Union{FeatureStore,StructureContainer}; ratio_val=0.1, shuffle=true)

Split a container into training and validation sets.

# Arguments
- `v::Union{FeatureStore,StructureContainer}`: Container to split

# Keyword Arguments
- `ratio_val=0.1`: Fraction of data to use for validation (0.0 to 1.0)
- `shuffle=true`: Randomly shuffle data before splitting

# Returns
- `Tuple{typeof(v), typeof(v)}`: (training_set, validation_set)

# See also
[`split`](@ref)
"""
function train_val_split(
    v::Union{FeatureStore,StructureContainer};
    ratio_val=0.1,
    shuffle=true,
)
    if isa(v, FeatureStore) && v.data === nothing
        error(
            "Cannot split a disk-backed FeatureStore. Provide pre-split stores " *
            "or use an in-memory store."
        )
    end
    nval = Int(floor(length(v) * ratio_val))
    ntrain = length(v) - nval
    if shuffle
        perm = randperm(length(v))
        return v[perm[1:ntrain]], v[perm[ntrain+1:end]]
    end
    return v[1:ntrain], v[ntrain+1:end]
end

"""
    split(c::Container, n1, n2, ...;shuffle=true)

Split the container into multiple parts, each with N number of structures.
By default, the output container will have their standardization transforms fitted.
"""
function Base.split(
    c::Union{StructureContainer,FeatureStore},
    nsplit::Vararg;
    shuffle=true,
    standardize=true,
    seed=42,
    respect_shakes=false,
)
    if isa(c, FeatureStore) && c.data === nothing
        error(
            "Cannot split a disk-backed FeatureStore. Disk-backed stores are " *
            "intended to be created pre-split (e.g. via `save_feature_store_streaming`). " *
            "To split, either use an in-memory store or provide separate train/val/test stores."
        )
    end
    if respect_shakes
        group_labels = _split_group_labels(c)
        unique_labels = unique(group_labels)
        out_labels = _split_vector(unique_labels, nsplit...; shuffle, seed)
        is_fc = isa(c, FeatureStore)
        out = map(out_labels) do selected
            c[findall(x -> x in selected, group_labels)]
        end
    else
        is_fc = isa(c, FeatureStore)
        out = _split_vector(c, nsplit...; shuffle, seed)
    end
    is_fc && standardize && standardize!(out...)
    out
end

"""
    split_and_featurize(sc::StructureContainer, feature, nsplit...;
        shuffle=true, seed=42, respect_shakes=false, standardize=true,
        elemental_energies=Dict{Symbol,Float64}(), nmax=500, show_progress=true,
        use_force_data=false, nsynthetic=1, max_disp=0.05, force_key=:REF_forces,
        kwargs...)

Memory-efficient split-then-featurize pipeline. Splits the StructureContainer first,
then generates features for each split independently, avoiding the peak memory of
holding all features plus split copies simultaneously.

When `use_force_data=true`, synthetic force-augmented entries are generated from
parent structures that carry force arrays via Taylor expansion before splitting.
Synthetic entries from the same parent are kept in the same split (`respect_shakes`
is forced to `true`).

Returns a tuple of FeatureStores, already standardized if `standardize=true`.

# Examples
```julia
sc = StructureContainer("structures/*.res")
cf = CellFeature([:C])
train, val, test = split_and_featurize(sc, cf, 0.8, 0.1, 0.1)
```
"""
function split_and_featurize(
    sc::StructureContainer,
    feature::AbstractCellFeatureSpec,
    nsplit::Vararg;
    shuffle=true,
    seed=42,
    respect_shakes=false,
    standardize=true,
    elemental_energies=Dict{Symbol,Float64}(),
    nmax=500,
    show_progress=true,
    use_force_data=false,
    nsynthetic=1,
    max_disp=0.05,
    force_key=:REF_forces,
    kwargs...,
)
    if use_force_data
        n_orig = length(sc)
        sc = synthetic_force_data(
            sc;
            nsynthetic,
            max_disp,
            force_key,
            rng=MersenneTwister(seed),
            keep_original=true,
            skip_missing=true,
        )
        if length(sc) == n_orig
            throw(ArgumentError(
                "`use_force_data=true` but none of the $n_orig structures carry a " *
                "`$(force_key)` array. Ensure the input data includes force arrays " *
                "(e.g. from DFT-calculated extxyz files with a `REF_forces` column).",
            ))
        end
        respect_shakes = true
        # When splitting with respect_shakes, the split operates on unique parent
        # labels, not individual structures. Convert integer counts to ratios
        # based on the pre-expansion count so the split proportions are preserved.
        if !isempty(nsplit) && all(x -> x isa Integer, nsplit)
            nsplit = ntuple(i -> Float64(nsplit[i] / n_orig), length(nsplit))
        end
    end

    split_scs =
        Base.split(sc, nsplit...; shuffle, seed, respect_shakes, standardize=false)

    fcs = map(split_scs) do split_sc
        FeatureStore(
            split_sc,
            feature;
            nmax,
            show_progress,
            elemental_energies,
            kwargs...,
        )
    end

    standardize && standardize!(fcs...)

    return Tuple(fcs)
end

enthalpy_per_atom(sc::StructureContainer) = sc.H ./ natoms.(sc.structures)
enthalpy_per_atom(fs::FeatureStore) = fs.H ./ fs.sizes

natoms(fs::FeatureStore) = fs.sizes
natoms(sc::StructureContainer) = natoms.(sc.structures)
natoms(x::AbstractVector{<:AbstractMatrix}) = [size(v, 2) for v in x]



"""
    standardize!(fs::FeatureStore)

Fit standardization transforms for the `FeatureStore`. The feature vectors
are **not** modified — transforms are stored in `fs.xt` and `fs.yt` and applied
on-the-fly by the NN interface during `forward!`.
"""
function standardize!(fs::FeatureStore; xt=nothing, yt=nothing)
    if xt === nothing
        if isnothing(fs.xt)
            if fs.data === nothing
                error(
                    "Cannot fit standardization transforms on a disk-backed FeatureStore " *
                    "without providing `xt` explicitly. Either pass `xt=...` or call " *
                    "`standardize!` on an in-memory store."
                )
            end
            total_x_train = reduce(hcat, fs.data)
            xt = fit(ZScoreTransform, total_x_train, dims=2)
            for i in eachindex(xt.scale)
                if abs(xt.scale[i]) < 1e-6
                    xt.scale[i] = 1.0
                end
            end
            fs.xt = xt
        end
    else
        fs.xt = xt
    end
    if yt === nothing
        if isnothing(fs.yt)
            y = fs.H
            yt = fit(ZScoreTransform, reshape(y ./ fs.sizes, 1, length(y)), dims=2)
            fs.yt = yt
        end
    else
        fs.yt = yt
    end
    fs
end



"""
    standardize!(fc_train, fcs...)

Standardise multiple feature containers in-place. Only the first argument is used for fitting.
Returns a tuple of the same containers with transforms fitted.
"""
function standardize!(fc_train, fc1, fcs...; kwargs...)
    standardize!(fc_train)
    standardize!(fc1; xt=fc_train.xt, yt=fc_train.yt, kwargs...)
    for fc in fcs
        standardize!(fc; xt=fc_train.xt, yt=fc_train.yt, kwargs...)
    end
    (fc_train, fc1, fcs...)
end


function get_fit_data(fs::FeatureStore)
    if fs.data !== nothing
        return fs.data, fs.H
    end
    return fs, fs.H
end

function get_data(fs::FeatureStore)
    if fs.data !== nothing
        return fs.data
    end
    Matrix{Float64}[fs[i] for i in 1:length(fs)]
end

function transform_y(fs::FeatureStore; yt=fs.yt)
    Hps = fs.H ./ fs.sizes
    Hps .-= yt.mean[1]
    Hps ./= yt.scale[1]
    Hps .* fs.sizes
end

function Base.:+(a::StructureContainer, b::StructureContainer)
    StructureContainer(
        vcat(a.paths, b.paths),
        vcat(a.H, b.H),
        vcat(a.structures, b.structures),
    )
end


function Base.:+(a::FeatureStore, b::FeatureStore)
    @assert a.feature == b.feature
    @assert a.elemental_energies == b.elemental_energies
    if a.data !== nothing && b.data !== nothing
        FeatureStore(
            vcat(a.H, b.H),
            vcat(a.labels, b.labels),
            vcat(a.metadata, b.metadata),
            a.feature,
            a.xt,
            a.yt,
            a.elemental_energies,
            vcat(a.sizes, b.sizes),
            vcat(a.data, b.data),
            "", 0, 0, UnitRange{Int}[], -1, Matrix{Float64}[], ReentrantLock(),
        )
    else
        error("Concatenation of disk-backed FeatureStores is not supported. Use `get_data` to materialize first.")
    end
end


# ═══════════════════════════════════════════════════════════════════════════
# FeatureStore — disk-backed constructors and I/O
# ═══════════════════════════════════════════════════════════════════════════

function FeatureStore(
    path::AbstractString,
    n_data::Int,
    n_batches::Int,
    batch_size::Int,
    batch_ranges::Vector{UnitRange{Int}},
    sizes::Vector{Int},
    H::Vector{Float64},
    labels::Vector{String},
    metadata::Vector{Dict{Symbol,Any}},
    feature,
    xt,
    yt,
    elemental_energies::Dict{Symbol,Float64},
)
    FeatureStore(
        H, labels, metadata, feature, xt, yt, elemental_energies, sizes,
        nothing, path, n_batches, batch_size, batch_ranges,
        0, Matrix{Float64}[], ReentrantLock(),
    )
end

batch_id_for_index(s::FeatureStore, i::Int) = cld(i, s.disk_batch_size)

function load_batch(s::FeatureStore, batch_id::Int)
    lock(s._cache_lock) do
        if batch_id == s.cached_batch_id
            return s.cached_batch
        end
        batch_data = Vector{Matrix{Float64}}(jldopen(s.path, "r") do f
            f["batch_$(batch_id)"]
        end)
        s.cached_batch_id = batch_id
        s.cached_batch = batch_data
        return batch_data
    end
end

function load_batch_H(s::FeatureStore, batch_id::Int)
    rng = s.disk_batch_ranges[batch_id]
    s.H[rng]
end

function save_feature_store(fs::FeatureStore, path; batch_size=2048)
    data = get_data(fs)
    n_data = length(data)
    n_batches = cld(n_data, batch_size)
    sizes = Int[size(v, 2) for v in data]

    jldopen(path, "w") do f
        f["version"] = "feature_store_v1"
        f["n_data"] = n_data
        f["batch_size"] = batch_size
        f["n_batches"] = n_batches
        f["H"] = fs.H
        f["labels"] = fs.labels
        f["metadata"] = fs.metadata
        f["feature"] = fs.feature
        f["xt"] = fs.xt
        f["yt"] = fs.yt
        f["elemental_energies"] = fs.elemental_energies
        f["sizes"] = sizes

        for b in 1:n_batches
            rng = ((b - 1) * batch_size + 1):min(b * batch_size, n_data)
            f["batch_$(b)"] = data[rng]
        end
    end
    path
end

function load_feature_store(path)
    jldopen(path, "r") do f
        n_data = f["n_data"]
        batch_size = f["batch_size"]
        n_batches = f["n_batches"]
        sizes = Vector{Int}(f["sizes"])
        batch_ranges = UnitRange{Int}[
            ((b - 1) * batch_size + 1):min(b * batch_size, n_data)
            for b in 1:n_batches
        ]
        elemental_energies = if haskey(f, "elemental_energies")
            Dict{Symbol,Float64}(f["elemental_energies"])
        else
            Dict{Symbol,Float64}()
        end

        FeatureStore(
            path,
            n_data,
            n_batches,
            batch_size,
            batch_ranges,
            sizes,
            f["H"],
            f["labels"],
            f["metadata"],
            f["feature"],
            f["xt"],
            f["yt"],
            elemental_energies,
        )
    end
end


# ═══════════════════════════════════════════════════════════════════════════
# FeatureBatch and each_batch — DataLoader-style batched iteration
# ═══════════════════════════════════════════════════════════════════════════

"""
    FeatureBatch

A batch of feature data yielded by `each_batch`. Contains feature matrices,
target energies, indices, and atom counts for a subset of structures.
"""
struct FeatureBatch
    matrices::Vector{Matrix{Float64}}
    H::Vector{Float64}
    indices::Vector{Int}
    natoms::Vector{Int}
end

function _hcat_matrices(mats::Vector{Matrix{Float64}})
    total_atoms = sum(size(m, 2) for m in mats)
    nfeat = size(first(mats), 1)
    X = similar(first(mats), nfeat, total_atoms)
    fill!(X, 0)
    offset = 1
    for m in mats
        nat = size(m, 2)
        X[:, offset:offset+nat-1] = m
        offset += nat
    end
    X
end

Base.hcat(b::FeatureBatch) = _hcat_matrices(b.matrices)

struct FeatureBatchIterator
    store::FeatureStore
    batch_size::Int
    shuffle::Bool
    seed::Int
end

"""
    each_batch(store::FeatureStore; batch_size=32, shuffle=true, seed=42) -> FeatureBatchIterator

Return an iterator that yields `FeatureBatch` objects for DataLoader-style
batched iteration over the FeatureStore.

For disk-backed stores, the iterator is I/O-aware: when the requested
`batch_size` is >= the disk batch size, it iterates disk batches directly
to minimize random I/O.

# Examples
```julia
for batch in each_batch(store; batch_size=64)
    X = hcat(batch)             # concatenate for SGD-style forward pass
    # or iterate batch.matrices for LM-style per-structure processing
end
```
"""
function each_batch(store::FeatureStore; batch_size::Int=32, shuffle::Bool=true, seed::Int=42)
    FeatureBatchIterator(store, batch_size, shuffle, seed)
end

function _io_efficient_batches(store::FeatureStore, batch_size::Int, rng)
    n = length(store)
    if store.data !== nothing
        perm = shuffle ? randperm(rng, n) : collect(1:n)
        batches = Vector{Vector{Int}}()
        for start in 1:batch_size:n
            push!(batches, collect(perm[start:min(start + batch_size - 1, n)]))
        end
        return batches
    end
    if batch_size >= store.disk_batch_size && store.n_disk_batches > 0
        perm = shuffle ? randperm(rng, store.n_disk_batches) : collect(1:store.n_disk_batches)
        return [collect(store.disk_batch_ranges[b]) for b in perm]
    end
    perm = shuffle ? randperm(rng, n) : collect(1:n)
    batches = Vector{Vector{Int}}()
    for start in 1:batch_size:n
        push!(batches, collect(perm[start:min(start + batch_size - 1, n)]))
    end
    return batches
end

function Base.iterate(iter::FeatureBatchIterator, state=nothing)
    if state === nothing
        rng = MersenneTwister(iter.seed)
        batches = _io_efficient_batches(iter.store, iter.batch_size, rng)
        state = (batches, 1)
    end
    batches, pos = state
    if pos > length(batches)
        return nothing
    end
    idx = batches[pos]
    matrices = [iter.store[i] for i in idx]
    batch = FeatureBatch(matrices, iter.store.H[idx], idx, iter.store.sizes[idx])
    return (batch, (batches, pos + 1))
end

Base.length(iter::FeatureBatchIterator) = cld(length(iter.store), iter.batch_size)


# ========== Streaming FeatureStore (out-of-core) ==========

"""
    OnlineStats{T}

Welford's online algorithm for computing mean and variance in a single pass.
"""
mutable struct OnlineStats{T}
    n::Int
    mean::Vector{T}
    M2::Vector{T}
end

OnlineStats(n::Int) = OnlineStats{Float64}(0, zeros(n), zeros(n))

function _welford_update!(s::OnlineStats, values::AbstractVector)
    s.n += 1
    for d in eachindex(values)
        delta = values[d] - s.mean[d]
        s.mean[d] += delta / s.n
        delta2 = values[d] - s.mean[d]
        s.M2[d] += delta * delta2
    end
    s
end

_mean(s::OnlineStats) = copy(s.mean)

function _std(s::OnlineStats)
    s.n < 2 ? ones(length(s.mean)) : sqrt.(s.M2 ./ (s.n - 1))
end


"""
    _prescan_extxyz(path; threshold, max_structures) -> (valid_indices, valid_energies, all_species)

Scan an extxyz file to find valid structures without constructing Cell objects.
Extracts the energy from the info dict of each frame and filters by
`abs(energy) / natoms < threshold`.  Also collects all unique species symbols.

Returns `(valid_frame_indices, valid_energies, sorted_unique_symbols)`.
"""
function _prescan_extxyz(path; threshold=10.0, max_structures=0)
    filepath, slice_str = _parse_path_slice(path)
    ext = lowercase(splitext(filepath)[2])

    if ext in (".xyz", ".extxyz")
        # Determine the frame range to scan
        if slice_str !== nothing
            total = _count_extxyz_frames(filepath)
            frame_range = _python_slice_to_range(slice_str, total)
        else
            frame_range = nothing
        end

        valid = Int[]
        energies = Float64[]
        all_species = Set{Symbol}()

        # Use ExtXYZ.read_frames in chunks for robustness
        if frame_range !== nothing
            scan_range = frame_range
        else
            total = _count_extxyz_frames(filepath)
            scan_range = 1:total
        end

        chunk_size = 10_000
        n_scanned = 0
        n_total = length(scan_range)
        prog = ProgressMeter.Progress(n_total; desc="Prescanning frames: ", showspeed=true)
        for chunk_start in first(scan_range):chunk_size:last(scan_range)
            chunk_end = min(chunk_start + chunk_size - 1, last(scan_range))
            chunk = chunk_start:chunk_end
            raw_dicts = ExtXYZ.read_frames(filepath, chunk)
            for (i, dict) in enumerate(raw_dicts)
                frame_id = chunk[i]
                info = dict["info"]
                energy = if haskey(info, "enthalpy")
                    Float64(info["enthalpy"])
                elseif haskey(info, "energy")
                    Float64(info["energy"])
                else
                    nothing
                end
                natoms = size(dict["arrays"]["pos"], 2)
                # Collect species
                species_arr = dict["arrays"]["species"]
                if species_arr isa AbstractVector
                    union!(all_species, Symbol.(species_arr))
                end
                if energy !== nothing && abs(energy) / natoms < threshold
                    push!(valid, frame_id)
                    push!(energies, energy)
                    if max_structures > 0 && length(valid) >= max_structures
                        ProgressMeter.finish!(prog)
                        return valid, energies, sort!(collect(all_species))
                    end
                end
                n_scanned += 1
            end
            ProgressMeter.update!(prog, n_scanned)
        end
        ProgressMeter.finish!(prog)
        return valid, energies, sort!(collect(all_species))
    else
        error("Streaming prescan only supports .xyz/.extxyz files, got `$path`.")
    end
end


"""
    _contiguous_chunks(indices::Vector{Int}; chunk_size=256)

Split a **sorted** vector of indices into contiguous ranges for efficient I/O.
Each returned range is a maximal contiguous block, split at `chunk_size` boundaries.
"""
function _contiguous_chunks(indices::Vector{Int}; chunk_size=256)
    isempty(indices) && return UnitRange{Int}[]
    ranges = UnitRange{Int}[]
    i = 1
    while i <= length(indices)
        start = indices[i]
        # Extend while consecutive and within chunk_size
        j = i
        n_extensions = 0  # number of times we extended past the start index
        while j < length(indices) && indices[j + 1] == indices[j] + 1 && n_extensions < chunk_size - 1
            j += 1
            n_extensions += 1
        end
        push!(ranges, indices[i]:indices[j])
        i = j + 1
    end
    ranges
end


"""
    save_feature_store_streaming(path, feature, output_path; kwargs...)

Stream features from an extxyz file to a `FeatureStore` on disk without ever
holding all features in memory.  Suitable for datasets exceeding available RAM.

Uses a single streaming pass with Welford's algorithm for online standardization
statistics.  Only one chunk of structures (~256) and one batch buffer
(`batch_size` structures) are in memory at any time.

# Algorithm
1. **Prescan**: read energies from extxyz info lines, filter by `threshold`
2. **Split**: shuffle valid indices, assign to train/val
3. **Stream**: read training frames in contiguous chunks, compute features,
   accumulate online stats, write batches to JLD2
4. **Val**: compute val features in memory (small set), apply same transforms

# Arguments
- `path`: extxyz file path (supports `@:N` slice syntax)
- `feature`: feature specification (e.g. `RandomEmbeddedCellFeature`)
- `output_path`: output JLD2 file path

# Returns
`(store_train::FeatureStore, store_val::FeatureStore)` — the training FeatureStore
loaded from disk and the validation FeatureStore (in memory).
"""
function save_feature_store_streaming(
    path::AbstractString,
    feature::AbstractCellFeatureSpec,
    output_path::AbstractString;
    batch_size::Int=2048,
    val_ratio::Float64=0.1,
    seed::Int=42,
    threshold::Float64=10.0,
    max_structures::Int=0,
    nmax::Int=500,
    skin::Float64=1.0,
    show_progress::Bool=true,
    elemental_energies=Dict{Symbol,Float64}(),
)
    filepath, _ = _parse_path_slice(path)

    # ---- Phase 1: Prescan ----
    println("Prescanning $path for valid structures...")
    valid_indices, valid_energies, _ = _prescan_extxyz(path; threshold, max_structures)
    n_valid = length(valid_indices)
    println("Found $n_valid valid structures (threshold=$threshold eV/atom).")

    n_valid > 0 || error("No valid structures found in `$path`.")

    # ---- Phase 2: Split ----
    rng = MersenneTwister(seed)
    perm = randperm(rng, n_valid)
    n_val = max(1, floor(Int, n_valid * val_ratio))
    n_train = n_valid - n_val
    train_perm = perm[n_val+1:end]
    val_perm = perm[1:n_val]
    train_indices = sort(valid_indices[train_perm])
    val_indices = sort(valid_indices[val_perm])

    println("Split: $n_train train / $n_val val structures.")

    # ---- Phase 3: Single streaming pass ----
    fsizes = feature_size(feature)
    n_feat = sum(fsizes)
    stats_x = OnlineStats(n_feat)
    stats_y = OnlineStats(1)

    # These will be updated inside the jldopen closure and read after
    local n_batches, n_train_actual
    local xt, yt_mean, yt_scale

    # Allocate reusable buffers
    batch_buffer = Matrix{Float64}[]
    all_H = Float64[]
    all_sizes = Int[]
    all_labels = String[]
    all_metadata = Dict{Symbol,Any}[]
    batch_id = 1

    # Val data (in-memory)
    val_fvecs = Matrix{Float64}[]
    val_H = Float64[]
    val_labels = String[]
    val_metadata = Dict{Symbol,Any}[]

    # Build lookup for val indices
    val_set = Set(val_indices)

    progress = show_progress ? ProgressMeter.Progress(
        n_train + n_val;
        desc="Computing features: ",
    ) : nothing

    # Collect all frames sorted for contiguous I/O
    all_indices_sorted = sort(vcat(train_indices, val_indices))
    chunks = _contiguous_chunks(all_indices_sorted; chunk_size=256)

    jldopen(output_path, "w") do f
        for chunk_range in chunks
            raw_dicts = try
                ExtXYZ.read_frames(filepath, chunk_range)
            catch e
                @warn "Failed to read frames $chunk_range" exception = (e, catch_backtrace())
                continue
            end

            for (local_i, dict) in enumerate(raw_dicts)
                frame_id = chunk_range[local_i]
                cell = _normalize_extxyz_cell!(
                    _extxyz_dict_to_cell(dict),
                    "$(first(chunk_range))-$local_i",
                )
                fvec = feature_vector(feature, cell; nmax, skin)
                nat = size(fvec, 2)
                H_i = _structure_enthalpy(cell)
                if length(elemental_energies) > 0
                    H_i -= get_elemental_energy(cell, elemental_energies)
                end

                is_val = frame_id in val_set

                if !is_val
                    # Update Welford stats (all features; 1-body overridden to no-op later)
                    for j in 1:nat
                        _welford_update!(stats_x, @view fvec[:, j])
                    end
                    _welford_update!(stats_y, [H_i / nat])

                    # Buffer for JLD2 batch
                    push!(batch_buffer, fvec)
                    push!(all_H, H_i)
                    push!(all_sizes, nat)
                    push!(all_labels, "structure_$(length(all_H))")
                    push!(all_metadata, Dict{Symbol,Any}())

                    # Flush batch when full
                    if length(batch_buffer) >= batch_size
                        # copy: JLD2 may hold a reference; we reuse batch_buffer after flushing
                        f["batch_$batch_id"] = copy(batch_buffer)
                        empty!(batch_buffer)
                        batch_id += 1
                    end
                else
                    # Val: store in memory
                    push!(val_fvecs, fvec)
                    push!(val_H, H_i)
                    push!(val_labels, "structure_$(length(val_H))")
                    push!(val_metadata, Dict{Symbol,Any}())
                end

                if progress !== nothing
                    ProgressMeter.next!(progress)
                end
            end
        end

        # Flush remaining training buffer
        if !isempty(batch_buffer)
            # copy: JLD2 may hold a reference; we reuse batch_buffer after flushing
            f["batch_$batch_id"] = copy(batch_buffer)
            empty!(batch_buffer)
        else
            batch_id -= 1
        end

        # ---- Compute transforms from online stats ----
        n_batches = max(batch_id, 1)
        n_train_actual = length(all_H)
        batch_ranges = if n_train_actual > 0
            [((b - 1) * batch_size + 1):min(b * batch_size, n_train_actual)
             for b in 1:n_batches]
        else
            UnitRange{Int}[]
        end

        xt_mean = _mean(stats_x)
        xt_scale = _std(stats_x)
        # Reset near-zero std to 1 to avoid division by near-zero
        for i in eachindex(xt_scale)
            if abs(xt_scale[i]) < 1e-6
                xt_scale[i] = 1.0
            end
        end
        xt = ZScoreTransform(n_feat, 2, xt_mean, xt_scale)
        yt_mean = _mean(stats_y)
        yt_scale = _std(stats_y)
        yt = ZScoreTransform(1, 2, yt_mean, yt_scale)

        # ---- Write metadata ----
        f["version"] = "feature_store_v1"
        f["n_data"] = n_train_actual
        f["batch_size"] = batch_size
        f["n_batches"] = n_batches
        f["H"] = all_H
        f["sizes"] = all_sizes
        f["labels"] = all_labels
        f["metadata"] = all_metadata
        f["feature"] = feature
        f["xt"] = xt
        f["yt"] = yt
        f["elemental_energies"] = Dict{Symbol,Float64}(elemental_energies)
    end

    println("Wrote $n_train_actual training structures in $n_batches batches to $output_path")

    val_sizes = [size(fv, 2) for fv in val_fvecs]
    fs_val = FeatureStore(
        val_H, val_labels, val_metadata, feature,
        nothing, nothing,
        Dict{Symbol,Float64}(elemental_energies),
        val_sizes,
        val_fvecs, "", 0, 0, UnitRange{Int}[], -1, Matrix{Float64}[], ReentrantLock(),
    )
    standardize!(fs_val;
        xt=xt,
        yt=ZScoreTransform(1, 2, yt_mean, yt_scale),
    )

    store = load_feature_store(output_path)
    println("Val set: $(length(fs_val)) structures.")

    return store, fs_val
end
