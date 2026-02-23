#=
Core evaluation layer.

This layer groups functionality needed for:
- feature generation
- gradient propagation
- NN interface abstraction/backends
- calculator-level energy/force/stress evaluation
=#

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
