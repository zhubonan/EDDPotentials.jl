module Analysis

using ..Potential
import ..Potential:
    ManualFluxBackPropInterface,
    glob_allow_abs,
    get_elemental_energy,
    eVAngToGPa
using ..Training
using ..Training:
    _load_structures_from_path,
    _parse_path_slice
using ..Workflow
using CellBase
import CellBase
using Printf
using LinearAlgebra
using StatsBase: mean, countmap
import Base

include("evaluation.jl")
include("records.jl")
include("plotting_recipes.jl")
include("lazy_res.jl")
include("plotting.jl")

export AbstractRecord,
    ShelxTITL,
    ShelxRecord,
    ComputedRecord,
    Simplex,
    PhaseDiagram,
    EvaluationConfig,
    EnergyMetrics,
    ForceMetrics,
    StressMetrics,
    LayerSaturationReport,
    AggregatedMetrics,
    benchmark_prediction,
    compare_energy,
    compare_forces,
    compare_stress,
    detect_saturation,
    energy_absolute_error,
    energy_ae_per_atom,
    evaluate,
    find_simplex,
    get_2d_plot_data,
    get_decomposition,
    get_e_above_hull,
    get_ternary_hulldata,
    load_evaluation_data,
    read_shelx_record,
    record_comp,
    record_energy,
    record_id,
    record_reduced_comp,
    plot_ternary,
    plot_binary,
    plot_binary_plotly,
    plot_minsep,
    plot_hist2d,
    plot_scatter,
    plot_std_vs_error,
    as_dataframe,
    select_refine

end # module Analysis
