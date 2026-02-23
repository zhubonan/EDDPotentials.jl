module EDDPotentialsCore

using TimerOutputs
using StatsBase: mean, std
using Glob: glob

const to = TimerOutput()
# Default to have the timer disabled
disable_timer!(to)

enable_eddp_timer!() = enable_timer!(to)
disable_eddp_timer!() = disable_timer!(to)
reset_eddp_timer!() = reset_timer!(to)

# Core evaluation stack
# External and shared utilities
include("constants.jl")
include("nnls/nnls.jl")
include("utils/utils.jl")

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
include("calculator/modifiers/modifiers.jl")
include("calculator/metacalculator.jl")
include("pairpot.jl")

# CellBase re-exports used by downstream workflows and callers
export Lattice, reciprocal, cellmat, cellvecs, cellpar, wrap!, volume, frac_pos, lattice
export displace!, clip, Cell, supercell
export distance_matrix, laplacian_matrix, nmodules, find_modules
export get_positions, set_positions!, set_cell!, set_cellmat!, remove_dimensions
export glob_allow_abs

# Core calculator and feature API
export AbstractCalc, get_cell
export eddp_energy,
    eddp_forces,
    eddp_energy_per_atom,
    eddp_stress,
    eddp_pressure,
    eddp_pressure_gpa,
    eddp_enthalpy,
    eddp_energy_std,
    eddp_energy_std_per_atom
export CellFeature, nfeatures
export TwoBodyFeature, ThreeBodyFeature, permequal, feature_size
export NNCalc, VariableCellCalc, MultiCalc, DimPenalty
export CoreRepulsion
export get_elemental_energy
export calculate!, _need_calc

# NN interfaces and helpers
export AbstractNNInterface,
    ManualFluxBackPropInterface,
    FluxInterface,
    LinearInterface,
    EnsembleNNInterface,
    paramvector,
    paramvector!,
    setparamvector!,
    nparams,
    get_flux_model,
    setup_fj,
    load_from_jld2,
    flux_mlp_model
export forward!, backward!, gradinp!, gradparam!
export ChainGradients, setup_fg_backprop

# Feature/gradient helpers used by functional tooling
export update_feature_vector!, compute_fv!, compute_fv_gv!, compute_fv_gv, _force_update!, _stress_update!
export eVAngToGPa, GPaToeVAng
export GradientWorkspace
export BodyEmbedding, CellEmbedding, BodyEmbeddingGradient, CellEmbeddingGradient
export two_body_view, three_body_view, backprop!
export _apply_embedding_batch, _apply_embedding_cell

# Meta calculator/modifier API used in workflows
export MetaCalculator,
    AbstractModifier,
    ExtraDimensionPenaltyModifier,
    CellRelaxationModifier,
    AdditiveCalculatorModifier,
    get_modifier,
    get_modifier_parameter,
    set_modifier_parameter!,
    get_augmented_positions,
    set_augmented_positions!

# Timer controls
export enable_eddp_timer!, disable_eddp_timer!, reset_eddp_timer!

function __init__()
    reset_timer!(to)
end

end # module
