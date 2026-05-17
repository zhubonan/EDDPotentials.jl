#=
Data loading and management for Builder workflows.

This module contains functions for:
- Loading structures and features from iterations
- Checking data readiness
- Structure counting utilities
=#


"""
    is_training_data_ready(bu::Builder, iteration)

Check if sufficient training data has been calculated for a given iteration.
"""
function is_training_data_ready(bu::Builder, iteration=bu.state.iteration)
    isdir(joinpath(bu.state.workdir, "gen$(iteration)-dft")) || return false
    input_count = generated_input_count(bu, iteration)
    input_count > 0 || return false
    return nstructures_calculated(bu, iteration) >=
           required_success_count(input_count, bu.state.per_generation_threshold)
end


"""
    nstructures(bu::Builder, iteration)

Count the number of generated structures (before DFT) for a given iteration.
"""
nstructures(bu::Builder, iteration) =
    length(glob_allow_abs(joinpath(bu.state.workdir, "gen$(iteration)/*.res")))


"""
    nstructures_calculated(bu::Builder, iteration)

Count the number of DFT-calculated structures for a given iteration.
"""
nstructures_calculated(bu::Builder, iteration) =
    length(glob_allow_abs(joinpath(bu.state.workdir, "gen$(iteration)-dft/*.res")))

"""
    generated_input_count(bu::Builder, iteration)

Count actual generated input structures for an iteration.
"""
generated_input_count(bu::Builder, iteration) = nstructures(bu, iteration)

"""
    required_success_count(input_count, threshold)

Minimum successful DFT outputs required for a generated input count.
"""
function required_success_count(input_count::Integer, threshold::Real)
    input_count > 0 || return 0
    return max(1, ceil(Int, input_count * threshold))
end

"""
    validate_dft_outputs!(bu::Builder, iteration)

Validate that DFT produced enough output structures for an iteration.
"""
function validate_dft_outputs!(bu::Builder, iteration=bu.state.iteration)
    input_count = generated_input_count(bu, iteration)
    input_dir = input_structure_dir(bu.state.workdir, iteration)
    output_dir = output_structure_dir(bu.state.workdir, iteration)
    input_count > 0 || throw(
        ErrorException(
            "No generated input structures found for iteration $iteration in `$input_dir`; cannot run training.",
        ),
    )
    required = required_success_count(input_count, bu.state.per_generation_threshold)
    calculated = nstructures_calculated(bu, iteration)
    calculated >= required && return true
    throw(
        ErrorException(
            "Too few DFT outputs for iteration $iteration: found $calculated successful structures in `$output_dir`, need at least $required from $input_count generated inputs in `$input_dir` (threshold=$(bu.state.per_generation_threshold)). Check DFT settings and generated structures.",
        ),
    )
end


"""
    load_structures(bu::Builder, iteration::Vararg{Int})

Load structures from specified iterations.

# Examples
```julia
sc = load_structures(bu, 0, 1, 2)  # Load from iterations 0, 1, 2
sc = load_structures(bu, 0:5)       # Load from iterations 0 through 5
sc = load_structures(bu)            # Load all iterations up to current
```
"""
function load_structures(bu::Builder, iteration::Vararg{Int})
    dirs = [joinpath(bu.state.workdir, "gen$(iter)-dft/*.res") for iter in iteration]
    sc = StructureContainer(dirs, threshold=bu.trainer.energy_threshold)
    return sc
end

"""
    load_structures(bu::Builder)

Load structures from all iterations (0 to current).
"""
load_structures(bu::Builder) = load_structures(bu, 0:bu.state.iteration)

"""
    load_structures(bu::Builder, iteration)

Load structures from an iteration range or collection.
"""
load_structures(bu::Builder, iteration) = load_structures(bu, iteration...)


"""
    load_features(bu::Builder, iteration...; show_progress=true, kwargs...)

Load features for specific iterations.

# Arguments
- `bu`: Builder object
- `iteration`: One or more iteration numbers
- `show_progress`: Show progress bar during feature generation
- `kwargs`: Additional keyword arguments for FeatureStore

# Examples
```julia
fc = load_features(bu, 0, 1, 2)  # Load from iterations 0, 1, 2
fc = load_features(bu, 0:5)       # Load from iterations 0 through 5
fc = load_features(bu)            # Load all iterations up to current
```
"""
function load_features(bu::Builder, iteration::Vararg{Int}; show_progress=true, kwargs...)
    sc = load_structures(bu, iteration...;)
    elemental_energies = make_symbol_keys(bu.state.elemental_energies)
    return FeatureStore(
        sc,
        bu.cf;
        nmax=bu.trainer.nmax,
        show_progress,
        elemental_energies,
        kwargs...,
    )
end

"""
    load_features(bu::Builder; kwargs...)

Load features from all iterations (0 to current).
"""
load_features(bu::Builder; kwargs...) = load_features(bu, 0:bu.state.iteration; kwargs...)

"""
    load_features(bu::Builder, iteration; kwargs...)

Load features from an iteration range or collection.
"""
load_features(bu::Builder, iteration; kwargs...) =
    load_features(bu, iteration...; kwargs...)

"""
    split_then_featurize(bu::Builder, nsplit...; kwargs...) -> Tuple{FeatureStore, ...}

Memory-efficient pipeline: load structures, split, then generate features for each split
independently. Peak memory is the size of the largest split's features, not the full dataset.
"""
function split_then_featurize(bu::Builder, nsplit::Vararg; kwargs...)
    sc = load_structures(bu)
    elemental_energies = make_symbol_keys(bu.state.elemental_energies)
    split_and_featurize(
        sc,
        bu.cf,
        nsplit...;
        elemental_energies,
        nmax=bu.trainer.nmax,
        use_force_data=bu.state.use_synthetic_force_data,
        nsynthetic=bu.state.synthetic_force_n,
        max_disp=bu.state.synthetic_force_max_disp,
        respect_shakes=bu.trainer.respect_shakes,
        kwargs...,
    )
end


"""
    num_atoms(fs::FeatureStore)

The number of atoms of each structure in the `FeatureStore`.
"""
num_atoms(fs::FeatureStore) = fs.sizes

"""
    num_atoms(sc::StructureContainer)

The number of atoms of each structure in the `StructureContainer`.
"""
num_atoms(sc::StructureContainer) = length(sc.structures)

"""
    eddp_energy_per_atom(fs::FeatureStore)

Energy per atom of each structure in the `FeatureStore`.
"""
function eddp_energy_per_atom(fs::FeatureStore)
    fs.H ./ num_atoms(fs)
end
