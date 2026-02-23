#=
Functional workflow layer.

This layer groups user-facing workflows built on the core evaluator:
- dataset preprocessing and training
- relaxation and random structure search workflows
- builder/orchestration tools
- analysis and plotting helpers
=#

# Training and data handling
include("lmsolve.jl")
include("preprocessing.jl")
include("eddpf90.jl")
include("training.jl")

# Utilities and workflows
include("tools.jl")
include("functional/functional.jl")

# Optimizers and optimisation interfaces
include("optimizer/abstract.jl")
include("optimizer/utils.jl")
include("optimizer/tpsd.jl")
include("optimizer/fire.jl")
include("opt.jl")

# Records, automation and orchestration
include("records.jl")
include("quickbuild.jl")
include("link/link.jl")
include("hyper.jl")

# Analysis and plotting
include("minsep.jl")
include("plotting/recipes.jl")
include("lazy_res.jl")
include("model_structures.jl")
include("plotting.jl")
