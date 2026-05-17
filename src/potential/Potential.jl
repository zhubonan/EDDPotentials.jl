module Potential

using TimerOutputs
import ..EDDPotentials:
    to,
    enable_eddp_timer!,
    disable_eddp_timer!,
    reset_eddp_timer!
using StatsBase: mean, std
using Glob: glob
using CellBase:
    reciprocal,
    clip,
    frac_pos,
    rattle!,
    reduce,
    Composition
using NLSolversBase
using Printf
using ProgressMeter: @showprogress
using Base.Threads
using JLD2
using Dates
using UUIDs
import CellBase: write_res

# Core evaluation stack
# External and shared utilities
include("constants.jl")
include("nnls/nnls.jl")
include("utils.jl")

# Feature generation and embedding
include("feature/feature.jl")
include("embedding.jl")
include("embedding_rules.jl")
include("gradient/gradient.jl")

# NN interface and helpers
include("nn/interface.jl")
include("nntools.jl")

# Calculator kernel
include("calculator.jl")
include("calculator/eddpcalculator.jl")
include("calculator/modifiers/modifiers.jl")
include("calculator/metacalculator.jl")
include("pairpot.jl")
include("utilities.jl")

# CellBase re-exports used by downstream workflows and callers
export Lattice, reciprocal, cellmat, cellvecs, cellpar, wrap!, volume, frac_pos, lattice
export clip, Cell
export distance_matrix
export get_positions, set_positions!, set_cell!, set_cellmat!, remove_dimensions

# Core calculator and feature API
export AbstractCalc, get_cell
export eddp_energy,
    eddp_forces,
    eddp_energy_per_atom,
    eddp_stress,
    eddp_pressure,
    eddp_pressure_gpa,
    eddp_pressure_GPa,
    eddp_enthalpy,
    eddp_energy_std,
    eddp_energy_std_per_atom
export CellFeature, nfeatures, nfeatures_per_channel, nchannels, nbodyfeatures, feature_size, one_body_offset, suggest_rcut
export feature_names, two_body_channel, two_body_range, three_body_channel, three_body_range
export spec_index, spec_symbol, lookup
export RandomEmbeddedCellFeature
export MultiRandomEmbeddedCellFeature
export AbstractNNInterface,
    LinearInterface,
    FluxInterface,
    ManualFluxBackPropInterface,
    EnsembleNNInterface
export FeatureProto, feature_size, one_body_offset
export ModelArtifact,
    FeatureArtifact,
    load_model,
    load_model_artifact,
    load_from_jld2,
    save_model_artifact,
    load_feature_json,
    load_feature_artifact,
    save_feature_json
export EDDPCalculator, load_calculator, predict, create_calculator
export CoreRepulsion, leaky_tanh
export calculate!
export clear_transient_gradients!, reinit, reinit!

# Meta calculator/modifier API used in workflows
export MetaCalculator,
    AbstractModifier,
    ExtraDimensionPenaltyModifier,
    VolumeConstraintModifier,
    AdditiveCalculatorModifier,
    get_modifier,
    get_modifier_parameter,
    set_modifier_parameter!
export _ensure_dir,
    allclose,
    ensure_dir,
    fd_desc,
    force_gradient,
    get_label,
    gvec_index_transfer,
    is_unique_fvec,
    lj_baseline_calc,
    lj_baseline_feature,
    lj_baseline_model,
    lj_baseline_weights,
    lj_like_calc,
    stem,
    stress_gradient,
    test_finite_difference,
    update_metadata!,
    write_res

# Timer controls
export enable_eddp_timer!, disable_eddp_timer!, reset_eddp_timer!

end # module
