# Calculator Modifiers Module
#
# This file includes all modifier implementations for the MetaCalculator pipeline.

# Include abstract interface
include("abstract.jl")

# Include specific modifiers
include("dimension_penalty.jl")
include("cell_relaxation.jl")
include("additive_calc.jl")

# Export abstract type
export AbstractModifier

# Export modifier types
export ExtraDimensionPenaltyModifier
export CellRelaxationModifier
export AdditiveCalculatorModifier

# Export interface functions
export pre_calculate!
export post_calculate!
export modifies_positions
export requires_forces
export get_parameter
export set_parameter!

# Export modifier-specific functions (for CellRelaxationModifier)
export get_augmented_positions
export set_augmented_positions!
export deformgradient
