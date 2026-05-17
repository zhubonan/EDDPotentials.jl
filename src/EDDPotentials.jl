module EDDPotentials

import CellBase
using Reexport
@reexport using CellBase: read_castep, read_res, write_res, read_cell, write_xyz, read_stru, write_stru
@reexport using StatsBase: mean
using TimerOutputs

const to = TimerOutput()
# Default to have the timer disabled
disable_timer!(to)

"""
    enable_eddp_timer!()

Enable internal timing collection for EDDP operations.
"""
enable_eddp_timer!() = enable_timer!(to)
"""
    disable_eddp_timer!()

Disable internal timing collection for EDDP operations.
"""
disable_eddp_timer!() = disable_timer!(to)
"""
    reset_eddp_timer!()

Reset internal timing counters.
"""
reset_eddp_timer!() = reset_timer!(to)

@reexport using Glob: glob


# abstract types
abstract type EDDPotentialsOption end

# end of abstract types

include("potential/Potential.jl")
@reexport using .Potential

include("training/Training.jl")
@reexport using .Training

include("simulation/Simulation.jl")
@reexport using .Simulation

const USE_CUDA = Ref{Bool}(false)

# Stub overridden by EDDPotentialsGPU extension
function train_sgd_gpu! end

# Public package namespaces and their source ownership.

include("search/Search.jl")
include("workflow/Workflow.jl")
include("analysis/Analysis.jl")

@reexport using .Search
@reexport using .Workflow
@reexport using .Analysis

function __init__()
    reset_eddp_timer!()
end

end # module
