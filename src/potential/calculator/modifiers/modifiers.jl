# Calculator Modifiers Module
#
# This file includes all modifier implementations for the MetaCalculator pipeline.

include("abstract.jl")
include("dimension_penalty.jl")
include("volume_constraint.jl")
include("additive_calc.jl")

export AbstractModifier
export ExtraDimensionPenaltyModifier
export VolumeConstraintModifier
export AdditiveCalculatorModifier
export pre_calculate!
export post_calculate!
export requires_forces
export get_parameter
export set_parameter!
