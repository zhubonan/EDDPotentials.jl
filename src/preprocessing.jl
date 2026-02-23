using JLD2
using StatsBase
import StatsBase: transform!, reconstruct!, ZScoreTransform
using Random
using Glob
using CellBase
import ProgressMeter
import CellBase: natoms, reduce, Composition
import Base

using Base.Threads

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
[`FeatureContainer`](@ref), [`CellBase.read_res_many`](@ref)
"""
struct StructureContainer{T,N,D}
    paths::Vector{String}
    "Enthalpy"
    H::Vector{N}
    structures::Vector{Cell{T,D}}
end

Base.length(v::StructureContainer) = length(v.H)

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
        if contains(path, "*") || contains(path, "?")
            append!(resolved_paths, glob_allow_abs(path))
        else
            push!(resolved_paths, path)
        end
    end
    tmp = []
    actual_labels = String[]
    for path in resolved_paths
        vres = CellBase.read_res_many(path)
        append!(tmp, vres)
        append!(actual_labels, map(x -> x.metadata[:label], vres))
    end

    structures = typeof(tmp[1])[x for x in tmp]
    H = [
        cell.metadata[:enthalpy] + cell.metadata[:volume] * GPaToeVAng(pressure_gpa) for
        cell in structures
    ]
    Ha = H ./ natoms.(structures)
    mask = _select_per_atom_threshold(structures, Ha; select_func, threshold)
    H_0Pa = [structures[i].metadata[:enthalpy] for i in mask]
    StructureContainer(actual_labels[mask], H_0Pa, structures[mask])
end

StructureContainer(path::AbstractString; kwargs...) =
    StructureContainer(glob(path); kwargs...)


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
function _split_vector(c, nsplit::Vararg{<:Real}; shuffle=true, seed=42)
    counts = Int[]
    total = length(c)
    for n in nsplit
        if isinteger(n)
            push!(counts, Int(round(n)))
        elseif 0.0 <= n <= 1.0
            push!(counts, Int(floor(total * n)))
        else
            error(
                "Invalid split value $n. Use integer counts or ratios in [0, 1].",
            )
        end
    end
    _split_vector(c, counts...; shuffle, seed)
end

"""
    FeatureContainer{T,N}

Container for holding computed features and energies ready for neural network training.

A FeatureContainer stores the feature vectors computed from atomic structures,
along with target energies and metadata. This is created from a StructureContainer
and CellFeature specification, and is directly used for training EDDP models.

# Fields
- `fvecs::Vector{T}`: Vector of feature matrices (nfeatures × natoms), one per structure
- `feature::CellFeature`: Feature specification used to generate features
- `H::Vector{N}`: Target total energies (in eV), one per structure
- `labels::Vector{String}`: Structure identifiers
- `metadata::Vector{Dict{Symbol,Any}}`: Per-structure metadata (formula, etc.)
- `is_x_transformed::Bool`: Whether features have been standardized
- `xt::Any`: Feature standardization transformation (ZScoreTransform)
- `yt::Any`: Energy standardization transformation
- `elemental_energies::Dict{Symbol,Float64}`: Reference energies per element

# Constructor
```julia
FeatureContainer(sc::StructureContainer, feature::CellFeature; kwargs...)
```

# Examples
```julia
# Basic workflow: structures → features → training
using EDDPotentials

# Step 1: Load structures
sc = StructureContainer("carbon_structures/*.res")
println(length(sc), " structures loaded")

# Step 2: Define features (two-body only for simplicity)
cf = CellFeature([:C])  # Carbon structures

# Step 3: Compute features
fc = FeatureContainer(sc, cf)
println(size(fc.fvecs[1]), " - feature dimensions for first structure")

# Step 4: Split into train/test
fc_train, fc_test = split(fc, 450, 50)  # 450 train, 50 test

# Step 5: Use for training
itf = FluxInterface(Chain(Dense(10, 20, tanh), Dense(20, 1)))
train!(itf, fc_train, fc_test; maxIter=500)
```

# Notes
- Feature computation can be slow for large systems; progress is shown by default
- Features are computed in parallel using threading
- Features can be standardized using `standardize!` or automatically via `split`
- Saved to disk using `save_fc(fc, "features.jld2")` and loaded with `load_fc`

# See also
[`StructureContainer`](@ref), [`CellFeature`](@ref), [`train!`](@ref), [`split`](@ref)
"""
mutable struct FeatureContainer{T,N}
    fvecs::Vector{T}
    feature::CellFeature
    H::Vector{N}
    labels::Vector{String}
    metadata::Vector{Dict{Symbol,Any}}
    is_x_transformed::Bool
    xt::Any
    yt::Any
    elemental_energies::Dict{Symbol,Float64}
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


function Base.show(io::IO, o::MIME"text/plain", v::FeatureContainer)
    ls = length.(v.fvecs)
    size_max = size(v.fvecs[findfirst(x -> x == maximum(ls), ls)])
    size_min = size(v.fvecs[findfirst(x -> x == minimum(ls), ls)])

    println(io, "FeatureContainer:")
    println(io, "  $(length(v)) data points ")
    println(io, "  Max size size: $(size_max)")
    println(io, "  Min size size: $(size_min)")
    println(io, "With CellFeature:")
    show(io, o, v.feature)
end

Base.show(io::IO, v::Union{StructureContainer,FeatureContainer}) =
    Base.show(io, MIME("text/plain"), v)

"""
    save_fc(fc, fname)

Save FeatureContainer into a file.
"""
function save_fc(fc, fname)
    jldopen(fname, "w") do fhandle
        fhandle["version"] = "1.0"
        fhandle["fvecs"] = fc.fvecs
        fhandle["feature"] = fc.feature
        fhandle["H"] = fc.H
        fhandle["labels"] = fc.labels
        fhandle["metadata"] = fc.metadata
        fhandle["is_x_transformed"] = fc.is_x_transformed
        fhandle["yt"] = fc.yt
        fhandle["xy"] = fc.xt
    end
end

"""
    load_fc(fname)

Load FeatureContainer from a file.
"""
function load_fc(fname)
    vreq = "1.0"
    jldopen(fname, "r") do fhandle
        ver = fhandle["version"]
        @assert fhandle["version"] == vreq "Inconsistent FeatureContainer version current $(ver) required $(vreq)."
        fvecs = fhandle["fvecs"]
        feature = fhandle["feature"]
        H = fhandle["H"]
        labels = fhandle["labels"]
        metadata = fhandle["metadata"]
        if "is_x_transformed" in keys(fhandle)
            is_x_transformed = fhandle["is_x_transformed"]
        else
            is_x_transformed = false
        end
        yt = fhandle["yt"]
        xt = fhandle["xy"]
        FeatureContainer(fvecs, feature, H, labels, metadata, is_x_transformed, yt, xt)
    end
end


Base.length(v::FeatureContainer) = length(v.fvecs)

"""
    FeatureContainer(sc::StructureContainer, featurespec; nmax=500, kwargs...)

Get a feature container.
"""
function FeatureContainer(
    sc::StructureContainer,
    feature::CellFeature;
    nmax=500,
    show_progress=true,
    elemental_energies=Dict{Symbol,Any}(),
    kwargs...,
)

    fvecs = Vector{Matrix{Float64}}(undef, length(sc))
    H = copy(sc.H)
    labels = collect(String, sc.paths)
    metadata = [cell.metadata for cell in sc.structures]
    if show_progress
        p = Progress(length(sc))
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
    # Process reference_energies
    if length(elemental_energies) > 0
        for i = 1:length(sc)
            H[i] -= get_elemental_energy(sc.structures[i], elemental_energies)
        end
    end
    FeatureContainer(
        fvecs,
        feature,
        H,
        labels,
        metadata,
        false,
        nothing,
        nothing,
        Dict{Symbol,Float64}(elemental_energies),
    )
end

function FeatureContainer(sc::StructureContainer; cf_kwargs=NamedTuple(), kwargs...)
    symbols = reduce(vcat, unique.(species.(sc.structures)))
    FeatureContainer(
        sc::StructureContainer,
        CellFeature(unique(symbols); cf_kwargs...);
        kwargs...,
    )
end

Base.IndexStyle(T::StructureContainer) = IndexLinear()
Base.IndexStyle(T::FeatureContainer) = IndexLinear()

Base.iterate(v::Union{FeatureContainer,StructureContainer}, state=1) =
    state > length(v) ? nothing : (v[state], state + 1)
Base.eltype(::Type{StructureContainer{T,N,D}}) where {T,N,D} = Cell{T,D}
Base.eltype(::Type{FeatureContainer{T,N}}) where {T,N} = Tuple{T,N}

Base.getindex(v::StructureContainer, i::Int) = v.structures[i]
Base.getindex(v::FeatureContainer, i::Int) = (v.fvecs[i], v.H[i])

function Base.getindex(
    v::StructureContainer,
    idx::Union{UnitRange,Vector{T}},
) where {T<:Int}
    structures = v.structures[idx]
    paths = v.paths[idx]
    H = v.H[idx]
    StructureContainer(paths, H, structures)
end

function Base.getindex(v::FeatureContainer, idx::Union{UnitRange,Vector{T}}) where {T<:Int}
    FeatureContainer(
        v.fvecs[idx],
        v.feature,
        v.H[idx],
        v.labels[idx],
        v.metadata[idx],
        v.is_x_transformed,
        v.xt,
        v.yt,
        v.elemental_energies,
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

function Base.getindex(v::FeatureContainer, idx::Vector{T}) where {T<:AbstractString}
    v[_select_by_label(v.labels, idx)]
end


"""
    train_test_split(v::Union{FeatureContainer,StructureContainer}; ratio_test=0.1, shuffle=true)

Split a container into training and test sets.

This is a convenience function for creating a simple train/test split with a
specified ratio. For more control over split sizes, use `split` instead.

# Arguments
- `v::Union{FeatureContainer,StructureContainer}`: Container to split

# Keyword Arguments
- `ratio_test=0.1`: Fraction of data to use for testing (0.0 to 1.0)
- `shuffle=true`: Randomly shuffle data before splitting

# Returns
- `Tuple{typeof(v), typeof(v)}`: (training_set, test_set)

# Examples
```julia
# Simple 90/10 train/test split
fc_train, fc_test = train_test_split(fc; ratio_test=0.1)

# 80/20 split without shuffling
sc_train, sc_test = train_test_split(sc; ratio_test=0.2, shuffle=false)
```

# See also
[`split`](@ref)
"""
function train_test_split(
    v::Union{FeatureContainer,StructureContainer};
    ratio_test=0.1,
    shuffle=true,
)
    ntest = Int(floor(length(v) * ratio_test))
    ntrain = length(v) - ntest
    if shuffle
        perm = randperm(length(v))
        return v[perm[1:ntrain]], v[perm[ntrain+1:end]]
    end
    return v[1:ntrain], v[train+1:end]
end

"""
    split(c::Container, n1, n2, ...;shuffle=true)

Split the container into multiple parts, each with N number of structures.
By default, the output container will have their feature vectors standardized.
"""
function Base.split(
    c::Union{StructureContainer,FeatureContainer},
    nsplit::Vararg;
    shuffle=true,
    standardize=true,
    apply_transform=true,
    seed=42,
    respect_shakes=false,
)
    if respect_shakes
        # Respect the shakes - ensure that the shake structures are always in one set
        isa(c, StructureContainer) ? (labels = c.paths) : (labels = c.labels)
        valid_labels = []
        for i = 1:length(c)
            result = match(r"(^.*)-shake.*$", labels[i])
            if isnothing(result)
                push!(valid_labels, labels[i])
            else
                push!(valid_labels, result[1])
            end
        end
        unique_labels = unique(valid_labels)
        # Split the set by the labels
        out_labels = _split_vector(unique_labels, nsplit...; shuffle, seed)
        # Construct the containers
        out = map(out_labels) do selected
            c[findall(x -> x in selected, valid_labels)]
        end
    else
        out = _split_vector(c, nsplit...; shuffle, seed)
    end
    isa(c, FeatureContainer) && standardize && standardize!(out...; apply_transform)
    out
end

enthalpy_per_atom(sc::StructureContainer) = sc.H ./ natoms.(sc.structures)
enthalpy_per_atom(fc::FeatureContainer) = fc.H ./ natoms(fc)

natoms(fc::FeatureContainer) = size.(fc.fvecs, 2)
natoms(fc::StructureContainer) = natoms.(fc.structures)



"""
    standardize!(fc::FeatureContainer)

Standardize the data in the `FeatureContainer`. The feature vectors are modified.
Note that only the input features are fitted and scaled, the outputs are only fitted
but the standardisation is not applied.
"""
function standardize!(fc::FeatureContainer; xt=nothing, yt=nothing, apply_transform=true)

    if xt === nothing
        if isnothing(fc.xt)
            total_x_train = reduce(hcat, fc.fvecs)
            n1 = feature_size(fc.feature)[1]
            xt = fit(StatsBase.ZScoreTransform, @view(total_x_train[n1+1:end, :]), dims=2)
            if apply_transform && !fc.is_x_transformed
                # We have to make copy to avoid affect the original data which may present in other objects
                fvecs = copy.(fc.fvecs)
                transform_x!(xt, fvecs)
                fc.fvecs .= fvecs
                fc.is_x_transformed = true
            end
            fc.xt = xt
        end
    else
        if !(fc.xt === nothing)
            @warn "FeatureContainer already has x transformation...."
            reconstruct_x!(fc)
        end
        fc.xt = xt
        # Make copy
        if apply_transform
            fvecs = copy.(fc.fvecs)
            transform_x!(xt, fvecs)
            fc.fvecs .= fvecs
            fc.is_x_transformed = true
        end
    end
    if yt === nothing
        if isnothing(fc.yt)
            y = fc.H
            yt = fit(ZScoreTransform, reshape(y ./ natoms(fc), 1, length(y)), dims=2)
            fc.yt = yt
        end
    else
        # No checks is needed here
        fc.yt = yt
    end
    fc
end



"""
    standardize!(fc_train, fcs...) 

Standardise multiple feature containers. Only the first argument is used for fitting.
"""
function standardize!(fc_train, fc1, fcs...; kwargs...)
    standardize!(fc_train)
    standardize!(fc1; xt=fc_train.xt, yt=fc_train.yt, kwargs...)
    for fc in fcs
        standardize!(fc; xt=fc_train.xt, yt=fc_train.yt, kwargs...)
    end

end

"""
    standardize(fc_train, fcs...) 

Standardise multiple feature containers. Only the first argument is used for fitting.
"""
function standardize(fc_train, fcs...; kwargs...)
    _fc_train = deepcopy(fc_train)
    _fcs = deepcopy.(fcs)
    standardize!(_fc_train; kwargs...)
    for fc in _fcs
        standardize!(fc; xt=fc_train.xt, yt=fc_train.yt, kwargs...)
    end
    (_fc_train, _fcs...)
end


function get_fit_data(fc::FeatureContainer)
    fc.fvecs, fc.H
end

"""
Apply transformation for training X inputs
"""
function transform_x!(xt, x_train)
    for data in x_train
        if size(data, 1) > xt.len
            transform!(xt, @view(data[end-xt.len+1:end, :]))
        else
            transform!(xt, data)
        end
    end
    x_train
end

function transform_x!(fc::FeatureContainer; xt=fc.xt)
    fvecs = copy.(fc.fvecs)
    @assert !fc.is_x_transformed
    transform_x!(xt, fvecs)
    fc.fvecs .= fvecs
    fc.is_x_transformed = true
end

function transform_y(fc::FeatureContainer; yt=fc.yt)
    Hps = fc.H ./ natoms(fc)
    Hps .-= yt.mean[1]
    Hps ./= yt.scale[1]
    Hps .* natoms(fc)
end

"""
Recover training X inputs before standardisation
"""
function reconstruct_x!(xt, x_train)
    for data in x_train
        if size(data, 1) > xt.len
            reconstruct!(xt, @view(data[end-xt.len+1:end, :]))
        end
    end
    x_train
end

"""
Recover training X inputs before standardisation
"""
function reconstruct_x!(fc::FeatureContainer; xt=fc.xt)
    fvecs = copy.(fc.fvecs)
    @assert fc.is_x_transformed
    reconstruct_x!(xt, fvecs)
    fc.fvecs .= fvecs
    fc.is_x_transformed = false
end

function Base.:+(a::StructureContainer, b::StructureContainer)
    StructureContainer(
        vcat(a.paths, b.paths),
        vcat(a.H, b.H),
        vcat(a.structures, b.structures),
    )
end


function Base.:+(a::FeatureContainer, b::FeatureContainer)
    @assert a.feature == b.feature
    #@assert a.xt == b.xt
    #@assert a.yt == b.yt
    @assert a.is_x_transformed == b.is_x_transformed
    @assert a.elemental_energies == b.elemental_energies
    FeatureContainer(
        vcat(a.fvecs, b.fvecs),
        a.feature,
        vcat(a.H, b.H),
        vcat(a.labels, b.labels),
        vcat(a.metadata, b.metadata),
        a.is_x_transformed,
        a.xt,
        b.yt,
        a.elemental_energies,
    )
end
