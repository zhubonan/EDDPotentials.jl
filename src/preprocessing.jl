using JLD2
using StatsBase

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



"""
    training_data(cell_data;ratio_test=0.1, shuffle_data=true, ignore_one_body=true)

Prepare data for model training from calculated feature vectors and reference energy data.
Split for train-test, and normalise the feature vectors and observables.
Both x and y needs to be normalised before training, and 
"""
function training_data(cell_data;ratio_test=0.1, shuffle_data=true, ignore_one_body=true)
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
    xt = fit(StatsBase.ZScoreTransform, total_x_train) 
    x_train_norm = map(x -> StatsBase.transform(xt, x), x_train)
    x_test_norm = map(x -> StatsBase.transform(xt, x), x_test)

    # Normalise y data
    yt = fit(ZScoreTransform, y_train)
    y_train_norm = StatsBase.transform(yt, y_train)
    y_test_norm = StatsBase.transform(yt, y_test)

    (;x_train_norm, y_train_norm, x_test_norm, y_test_norm, xt, yt)
end