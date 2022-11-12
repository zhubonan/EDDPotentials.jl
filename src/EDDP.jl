module EDDP

greet() = print("Hello World!")
using CellBase
using CellBase: read_castep
import CellBase
using StatsBase: mean
export mean
using TimerOutputs
export reset_timer!, enable_timer!, disable_timer!

const to = TimerOutput() 
# Default to have the timer disabled
disable_timer!(to)


# External
include("nnls.jl")
include("feature.jl")
include("gradient.jl")
include("nn/interface.jl")
include("nntools.jl")
include("lmsolve.jl")
include("preprocessing.jl")
include("eddpf90.jl")
include("training.jl")
include("calculator.jl")
include("tools.jl")
include("opt.jl")
include("iterative_build.jl")

include("link/link.jl")

export Lattice, reciprocal, cellmat, cellvecs, cellpar, wrap!, volume, frac_pos, lattice
export distance_between, distance_squared_between, displace!, clip, Cell, supercell
export distance_matrix, laplacian_matrix, nmodules, find_modules
export get_cell, get_forces, get_energy, get_stress, VariableLatticeFilter, CellFeature, CellWorkSpace, CellCalculator
export get_enthalpy, get_pressure
export get_positions, set_positions!, set_cell!, set_cellmat!
export TrainingOptions, FeatureOptions

function __init__()
    reset_timer!(to)
end

function clear_timer!()
    reset_timer!(to)
end

end # module
