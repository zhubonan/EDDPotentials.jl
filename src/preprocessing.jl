using JLD2
using StatsBase
using Random
using Glob
using CellBase
import CellBase: natoms


"""
    StructureContainer{T}

Loader for storing structure data
"""
struct StructureContainer{T}
    paths::Vector{String}
    "Enthalpy"
    H::Vector{Float64}
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
    FeatureContainer{T,N}

Container for holding features from structures.
"""
struct FeatureContainer{T,N}
    structure_container::StructureContainer{T}
    fvecs::Vector{N}
    feature::CellFeature
end

Base.length(v::FeatureContainer) = length(v.structure_container)

"""
    FeatureContainer(sc::StructureContainer, featurespec; nmax=500, kwargs...)

Get a feature container.
"""
function FeatureContainer(sc::StructureContainer, feature::CellFeature; nmax=500, kwargs...)

    fvecs = Vector{Matrix{Float64}}(undef, length(sc))
    for i=1:length(sc)
        fvecs[i] = EDDP.feature_vector(feature, sc.structures[i];nmax, kwargs...)
    end
    FeatureContainer(sc, fvecs, feature)
end

function FeatureContainer(sc::StructureContainer, feature::FeatureOptions; nmax=500, kwargs...)
    FeatureContainer(sc::StructureContainer, CellFeature(feature); nmax, kwargs...)
end

function FeatureContainer(sc::StructureContainer; kwargs...)
    symbols = reduce(vcat, unique.(species.(sc.structures)))
    FeatureContainer(sc::StructureContainer, FeatureOptions(elements=unique(symbols));kwargs...)
end

Base.IndexStyle(T::StructureContainer) = IndexLinear()

Base.getindex(v::StructureContainer, i::Int) = v.structures[i]
Base.getindex(v::FeatureContainer, i::Int) = v.structure_container.structures[i]

function Base.getindex(v::StructureContainer, idx::Union{UnitRange, Vector})   
    structures = v.structures[idx]
    paths = v.paths[idx]
    H = v.H[idx]
    StructureContainer(paths, H, structures)
end

function Base.getindex(v::FeatureContainer, idx::Union{UnitRange, Vector})   
    new_sc = v.structure_container[idx]
    fcs = v.fvecs[idx]
    FeatureContainer(new_sc, fcs, v.feature)
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

enthalpy_per_atom(sc::StructureContainer) = sc.H ./ natoms.(sc.structures)
enthalpy_per_atom(fc::FeatureContainer) = enthalpy_per_atom(fc.structure_container)

natoms(fc::FeatureContainer) = natoms(fc.structure_container)
natoms(fc::StructureContainer) = natoms.(fc.structures)

load_structures(files::AbstractString, featurespec;energy_threshold=20., nmax=500) = load_structures(glob(files), featurespec;energy_threshold, nmax)

"""
    training_data(fc::FeatureContainer;ratio_test=0.1, shuffle_data=true)

Obtain training data.
"""
function training_data(fc::FeatureContainer;ratio_test=0.1, shuffle_data=true)

    fc_train, fc_test = train_test_split(fc;ratio_test,shuffle=shuffle_data)

    # enthalpy per atom for normalisation
    y_train = fc_train.structure_container.H
    y_test = fc_test.structure_container.H
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