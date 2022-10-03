using JLD2
using StatsBase
using Random
using Glob
using CellBase


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
"""
function StructureContainer(paths::Vector)
    resolved_paths = String[]
    for path in paths
        if contains(path, "*") || contains(path, "?")
            append!(resolved_paths, glob(path))
        else
            push!(resolved_paths, path)
        end
    end
    structures = read_res.(resolved_paths)
    H = [cell.metadata[:enthalpy] for cell in structures]
    StructureContainer(resolved_paths, H, structures)
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
    if shuffle
        perm = randperm(length(v))
        ntest = Int(floor(length(v) * ratio_test))
        ntrain = length(v) - ntest
        return v[perm[ntrain+1:end]], v[perm[1:ntrain]] 
    end
    return v[train+1:end], v[1:ntrain] 
end

enthalpy_per_atom(sc::StructureContainer) = sc.H ./ natoms.(sc.structures)
enthalpy_per_atom(fc::FeatureContainer) = enthalpy_per_atom(fc.structure_container)

"""
Load all structures from many paths
"""
function load_structures(fpath::Vector{T}, featurespec;energy_threshold=20., nmax=500) where {T<:AbstractString}
    cells = Cell[]
    for f in fpath
        try
            push!(cells, read_res(f))
        catch err
            @warn "Cannot read $f"
            continue
        end
    end
    load_structures(cells, featurespec;fpath, energy_threshold, nmax)
end

"""
Prepare structure used for training model
"""
function load_structures(cells::Vector{T}, featurespec;fpath=nothing, energy_threshold=20., nmax=500) where {T<:Cell} 
    enthalpy = [ x.metadata[:enthalpy] for x in cells]
    natoms = [EDDP.nions(c) for c in cells];
    enthalpy_per_atom = enthalpy ./ natoms

    if energy_threshold > 0
        # Drop structure that are too high in energy
        mask = enthalpy_per_atom .< (minimum(enthalpy_per_atom) + energy_threshold)
        enthalpy = enthalpy[mask]
        enthalpy_per_atom = enthalpy_per_atom[mask]
        cells = cells[mask]
        natoms = natoms[mask]
        if !isnothing(fpath)
            fpath = fpath[mask]
        end
    end

    # Construct feature vectors
    fvecs = Vector{Matrix{Float64}}(undef, length(cells))
    for i=1:length(cells)
        fvecs[i] = EDDP.feature_vector(featurespec, cells[i];nmax)
    end

    (;cells, enthalpy, enthalpy_per_atom, natoms, fpath, fvecs, featurespec)
end

load_structures(files::AbstractString, featurespec;energy_threshold=20., nmax=500) = load_structures(glob(files), featurespec;energy_threshold, nmax)

function training_data(fc::FeatureContainer;ratio_test=0.1, shuffle_data=true, ignore_one_body=true)

    fc_train, fc_test = train_test_split(fc;ratio_test,shuffle=shuffle_data)

    # trim the size of the feature vectors
    nfe = nfeatures(fc.feature;ignore_one_body)  # expected number of features
    if any(size.(fc.fvecs, 1) .!= nfe)
        x_train = map(x -> x[end-nfe+1:end, :], fc_train.fvecs)
        x_test = map(x -> x[end-nfe+1:end, :], fc_test.fvecs)
    else
        x_train = fc_train.fvecs
        x_test = fc_test.fvecs
    end

    # enthalpy per atom for normalisation
    y_train = enthalpy_per_atom(fc_train)
    y_test = enthalpy_per_atom(fc_test)

    # Normalization - peratom
    total_x_train = reduce(hcat, x_train)

    # Fit normalisation using all of the training data
    xt = fit(StatsBase.ZScoreTransform, total_x_train, dims=2) 
    # Apply transform for individual data set
    x_train_norm = map(x -> StatsBase.transform(xt, x), x_train)
    x_test_norm = map(x -> StatsBase.transform(xt, x), x_test)

    # Normalise y data
    yt = fit(ZScoreTransform, y_train)
    y_train_norm = StatsBase.transform(yt, y_train)
    y_test_norm = StatsBase.transform(yt, y_test)

    (;x_train_norm, y_train_norm, x_test_norm, y_test_norm, xt, yt)
end

"""
    training_data(cell_data;ratio_test=0.1, shuffle_data=true, ignore_one_body=true)

Prepare data for model training from calculated feature vectors and reference energy data.
Split for train-test, and normalise the feature vectors and observables.
Both x and y needs to be normalised before training, and 
"""
function training_data(cell_data::NamedTuple;ratio_test=0.1, shuffle_data=true, ignore_one_body=true)
    # Convert to Float32 format for training
    fvecs = map(x -> convert.(Float32, x), cell_data.fvecs)
    enthalpy_per_atom = convert.(Float32, cell_data.enthalpy_per_atom)


    ntotal = length(fvecs)
    ntest = Int(round(ntotal * ratio_test))

    # Shuffle the order of the data
    if shuffle_data
        tmp = collect(1:length(fvecs))
        shuffle!(tmp)
        fvecs = fvecs[tmp]
        enthalpy_per_atom = enthalpy_per_atom[tmp]
    end

    # trim the size of the feature vectors
    nfe = nfeatures(cell_data.featurespec;ignore_one_body)  # expected number of features
    if size(fvecs[1], 1) != nfe
        fvecs = map(x -> x[end-nfe+1:end, :], fvecs)
    end

    x_train = fvecs[1:ntotal-ntest]
    x_test = fvecs[ntotal-ntest+1:ntotal]
    y_train = enthalpy_per_atom[1:ntotal-ntest]
    y_test = enthalpy_per_atom[ntotal-ntest+1:ntotal]


    # Normalization - peratom
    total_x_train = reduce(hcat, x_train)
    # Fit normalisation using all of the training data
    xt = fit(StatsBase.ZScoreTransform, total_x_train, dims=2) 
    x_train_norm = map(x -> StatsBase.transform(xt, x), x_train)
    x_test_norm = map(x -> StatsBase.transform(xt, x), x_test)

    # Normalise y data
    yt = fit(ZScoreTransform, y_train)
    y_train_norm = StatsBase.transform(yt, y_train)
    y_test_norm = StatsBase.transform(yt, y_test)

    (;x_train_norm, y_train_norm, x_test_norm, y_test_norm, xt, yt)
end