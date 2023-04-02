#=
Functional routines for using the model for different tasks
=#

# Transformation of the structures
include("transforms.jl")
# Interface with the `buildcell` program
include("buildcell.jl")
# Geometry optimisation
include("relax.jl")
# Perform random structure searching
include("rss.jl")
