module EDDP
using Reexport

@reexport using CellBase
using CellBase: read_castep
import CellBase
using StatsBase: mean
export mean
using TimerOutputs
export reset_timer!, enable_timer!, disable_timer!

const to = TimerOutput()
# Default to have the timer disabled
disable_timer!(to)

# Re-export `glob`
using Glob
export glob


# External
include("nnls/nnls.jl")

# Feature generation
include("feature.jl")
include("embedding.jl")
include("embedding_rules.jl")
include("gradient.jl")

# NN interface
include("nn/interface.jl")
include("nntools.jl")

# Training and data handling
include("utils.jl")
include("lmsolve.jl")
include("preprocessing.jl")
include("eddpf90.jl")
include("training.jl")
include("calculator.jl")
include("tools.jl")
include("opt.jl")
include("records.jl")
include("quickbuild.jl")

# Automated building
include("link/link.jl")
include("link/trainer.jl")

# Plotting and analysis
include("minsep.jl")
include("plotting/recipes.jl")
include("lazy_res.jl")

export Lattice, reciprocal, cellmat, cellvecs, cellpar, wrap!, volume, frac_pos, lattice
export distance_between, distance_squared_between, displace!, clip, Cell, supercell
export distance_matrix, laplacian_matrix, nmodules, find_modules
export get_cell,
    get_forces,
    get_energy,
    get_energy_per_atom,
    get_stress,
    CellFeature,
    NNCalc,
    VariableCellCalc,
    get_energy_std,
    get_energy_std_per_atom
export get_enthalpy, get_pressure
export get_positions, set_positions!, set_cell!, set_cellmat!
export StructureContainer, FeatureContainer, load_from_jld2
export BuilderState, TrainingOption, link!
export Builder,
    walk_forward_tests, load_ensemble, load_features, load_training_dataset, run_rss
export rmse_per_atom, mae_per_atom, max_ae_per_atom
export nfeatures
export TrainingResults
export PhaseDiagram, read_shelx_record, ShelxRecord, ComputedRecord

function __init__()
    reset_timer!(to)
end

function clear_timer!()
    reset_timer!(to)
end

end # module
