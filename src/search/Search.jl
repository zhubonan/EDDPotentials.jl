module Search

using ..Potential
import ..Potential:
    AbstractNNInterface,
    GPaToeVAng,
    eVAngToGPa,
    glob_allow_abs,
    _make_core_repulsion,
    get_label,
    stem,
    update_metadata!,
    is_unique_fvec
using ..Simulation
import ..EDDPotentials: EDDPotentialsOption
using CellBase
using LinearAlgebra
import CellBase:
    add_dimensions,
    read_res,
    write_res,
    set_cellmat!,
    set_positions!,
    remove_dimensions

include("buildcell.jl")
include("transforms.jl")
include("hot.jl")
include("hyper.jl")
include("rss.jl")
include("minsep.jl")
include("model_structures.jl")

run_rss(args...; kwargs...) = _run_rss(args...; kwargs...)

export HotAIRSSOption,
    HotAIRSSResult,
    RssPruneOption,
    SepecieSeparation,
    bP,
    build_one,
    build_random_structures,
    build_and_relax,
    build_and_relax_one,
    check_energy_threshold!,
    compute_specie_separations,
    gather_minsep_stats,
    get_specie_pairs,
    hot_airss!,
    model_structures,
    rattle_cell!,
    run_rss,
    shake_res,
    scheduled_penalty_relax!,
    hyper_gen_relax,
    HyperRoundDiagnostics

end # module Search
