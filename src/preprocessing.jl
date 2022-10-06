using JLD2
using StatsBase
using Random
using Glob
using CellBase
import CellBase: natoms
import Base


"""
    StructureContainer{T}

Loader for storing structure data
"""
struct StructureContainer{T, N}
    paths::Vector{String}
    "Enthalpy"
    H::Vector{N}
    structures::Vector{Cell{T}}
end

Base.length(v::StructureContainer) = length(v.H)

"""
    StructureContainer(paths::Vector)

Args:

    - `energy_threshold`: structures with per-atom energy higher than this are excluded. 
      Relative to the median energy.
"""
function StructureContainer(paths::Vector;energy_threshold=10.)
    resolved_paths = String[]
    for path in paths
        if contains(path, "*") || contains(path, "?")
            append!(resolved_paths, glob(path))
        else
            push!(resolved_paths, path)
        end
    end
    tmp = []
    actual_paths = String[]
    for path in resolved_paths
        if contains(path, "packed")
            vres = CellBase.read_res_many(path)
            append!(tmp, vres)
            append!(actual_paths, map(x -> x.metadata[:label], vres)) 
        else
            push!(tmp, CellBase.read_res(path))
            push!(actual_paths, path)
        end
    end
    structures = typeof(tmp[1])[x for x in tmp]

    H = [cell.metadata[:enthalpy] for cell in structures]
    Ha = H ./ natoms.(structures)

    mask = Ha .< (median(Ha) + energy_threshold)
    StructureContainer(actual_paths[mask], H[mask], structures[mask])
end


"""
Split a vector by integer numbres
"""
function _split_vector(c, nsplit::Vararg{Int};shuffle=true)
    out = []
    if shuffle
        perm = randperm(length(c))
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
Split a vector by fractions
"""
function _split_vector(c, nsplit::Vararg{Real};shuffle=true)
    ntot = length(c)
    intsplit = nsplit .* ntot .|> floor .|> Int
    _split_vector(c, intsplit...;shuffle)
end

"""
    FeatureContainer{T,N}

Container for holding features from structures.
"""
mutable struct FeatureContainer{T, N}
    fvecs::Vector{T}
    feature::CellFeature
    H::Vector{N}
    labels::Vector{String}
    metadata::Vector{Dict{Symbol, Any}}
    xt
    yt
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
    ls = length.(v.structures)
    size_max = length(v.fvecs[findfirst(x -> x == maximum(ls), ls)])
    size_min = length(v.fvecs[findfirst(x -> x == minimum(ls), ls)])

    println(io, "StructureContainer:")
    println(io, "  $(length(v)) data points ")
    println(io, "  Max structure size: $(size_max)")
    println(io, "  Min structure size: $(size_min)")
    println(io, "With CellFeature:")
    show(io, o, v.feature)
end


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
        yt = fhandle["yt"]
        xt = fhandle["xy"]
        FeatureContainer(fvecs, feature, H, labels, metadata, yt, xt)
    end
end


Base.length(v::FeatureContainer) = length(v.fvecs)

"""
    FeatureContainer(sc::StructureContainer, featurespec; nmax=500, kwargs...)

Get a feature container.
"""
function FeatureContainer(sc::StructureContainer, feature::CellFeature; nmax=500, kwargs...)

    fvecs = Vector{Matrix{Float64}}(undef, length(sc))
    for i=1:length(sc)
        fvecs[i] = EDDP.feature_vector(feature, sc.structures[i];nmax, kwargs...)
    end
    metadata = [cell.metadata for cell in sc.structures]
    H = copy(sc.H)
    labels = collect(String, m[:label] for m in metadata)
    for (i, m) in enumerate(metadata)
        form, nf = CellBase.formula_and_factor(sc.structures[i])
        m[:formula] = form
        m[:nformula] = nf
    end
    FeatureContainer(fvecs, feature, H, labels, metadata, nothing, nothing)
end

function FeatureContainer(sc::StructureContainer, feature::FeatureOptions; nmax=500, kwargs...)
    FeatureContainer(sc::StructureContainer, CellFeature(feature); nmax, kwargs...)
end

function FeatureContainer(sc::StructureContainer; kwargs...)
    symbols = reduce(vcat, unique.(species.(sc.structures)))
    FeatureContainer(sc::StructureContainer, FeatureOptions(elements=unique(symbols));kwargs...)
end


Base.IndexStyle(T::StructureContainer) = IndexLinear()
Base.IndexStyle(T::FeatureContainer) = IndexLinear()

Base.iterate(v::Union{FeatureContainer, StructureContainer}, state=1) = state > length(v) ? nothing : (v[state], state+1)
Base.eltype(::Type{StructureContainer{T, N}}) where{T, N} = Cell{T}
Base.eltype(::Type{FeatureContainer{T, N}})  where{T, N} = Tuple{T, N}

Base.getindex(v::StructureContainer, i::Int) = v.structures[i]
Base.getindex(v::FeatureContainer, i::Int) = (v.fvecs[i], v.H[i])

function Base.getindex(v::StructureContainer, idx::Union{UnitRange, Vector{T}}) where {T<: Int}   
    structures = v.structures[idx]
    paths = v.paths[idx]
    H = v.H[idx]
    StructureContainer(paths, H, structures)
end

function Base.getindex(v::FeatureContainer, idx::Union{UnitRange, Vector{T}}) where {T<:Int}   
    FeatureContainer(v.fvecs[idx], v.feature, v.H[idx], v.labels[idx], v.metadata[idx], v.xt, v.yt)
end
function _select_by_label(all_labels, labels)
    id_selected = Int[]
    for val in labels
        id_temp = findfirst(x -> x==val, all_labels)
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


function Base.getindex(v::StructureContainer, idx::Vector{T}) where {T<: AbstractString}   
    labels = [x.metadata[:label] for x in v.structures]
    v[_select_by_label(labels, idx)]
end

function Base.getindex(v::FeatureContainer, idx::Vector{T}) where {T<:AbstractString}   
    v[_select_by_label(v.labels, idx)]
end


"""
    tain_test_split(v::FeatureContainer; ratio_test=0.1, shuffle=true)

Split the training 
"""
function train_test_split(v::Union{FeatureContainer, StructureContainer}; ratio_test=0.1, shuffle=true)
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
"""
Base.split(c::Union{StructureContainer, FeatureContainer}, nsplit::Vararg;shuffle=true) = _split_vector(c, nsplit...;shuffle)

enthalpy_per_atom(sc::StructureContainer) = sc.H ./ natoms.(sc.structures)
enthalpy_per_atom(fc::FeatureContainer) = fc.H ./ natoms(fc)

natoms(fc::FeatureContainer) = size.(fc.fvecs, 2)
natoms(fc::StructureContainer) = natoms.(fc.structures)

load_structures(files::AbstractString, featurespec;energy_threshold=20., nmax=500) = load_structures(glob(files), featurespec;energy_threshold, nmax)

"""
    training_data(fc::FeatureContainer;ratio_test=0.1, shuffle_data=true)

Obtain training data.
"""
function training_data(fc::FeatureContainer;ratio_test=0.1, shuffle_data=true)

    fc_train, fc_test = train_test_split(fc;ratio_test,shuffle=shuffle_data)

    # enthalpy per atom for normalisation
    y_train = fc_train.H 
    y_test = fc_test.H
    x_train = fc_train.fvecs
    x_test = fc_test.fvecs

    # Normalization - peratom
    total_x_train = reduce(hcat, x_train)

    # Check fitting the transform should exclude the one-body vectors as they are one-hot encoders....
    n1 = feature_size(fc.feature)[1]
    xt = fit(StatsBase.ZScoreTransform, @view(total_x_train[n1+1:end, :]), dims=2) 

    # Standardise the per-atom y data
    yt = fit(ZScoreTransform, reshape(y_train ./ natoms(fc_train), 1, length(y_train)), dims=2)

    (;x_train, y_train, x_test, y_test, xt, yt)
end


"""
    standardize!(fc::FeatureContainer)

Standardise the data in the `FeatureContainer`. The feature vectors are modified.
"""
function standardize!(fc::FeatureContainer; xt=nothing, yt=nothing)
    if xt === nothing
        if isnothing(fc.xt) 
            total_x_train = reduce(hcat, fc.fvecs)
            n1 = feature_size(fc.feature)[1]
            xt = fit(StatsBase.ZScoreTransform, @view(total_x_train[n1+1:end, :]), dims=2) 
            # We have to make copy to avoid affect the original data which may present in other objects
            fvecs = copy.(fc.fvecs)
            transform_x!(xt, fvecs)
            fc.fvecs .= fvecs
            fc.xt = xt
        end
    else
        @assert fc.xt === nothing "FeatureContainer already has x transformation...."
        fc.xt = xt
        # Make copy
        fvecs = copy.(fc.fvecs)
        transform_x!(xt, fvecs)
        fc.fvecs .= fvecs
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

function standardize!(fc_train, fcs...) 
    standardize!(fc_train)
    for fc in fcs
        standardize!(fc;xt=fc_train.xt, yt=fc_train.yt)
    end

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

"""
Recover training X inputs
"""
function reconstruct_x!(xt, x_train)
    for data in x_train
        if size(data, 1) > xt.len
            reconstruct!(xt, @view(data[end-xt.len+1:end, :]))
        end
    end
    x_train
end