using Printf
using LinearAlgebra: norm, dot, tr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

"""
    EvaluationConfig

Controls how reference data is extracted from `Cell` objects for evaluation.
"""
struct EvaluationConfig
    "Metadata key for total energy (eV)"
    energy_key::Symbol
    "Arrays key for reference forces (eV/Angstrom)"
    force_key::Symbol
    "Metadata key for reference stress"
    stress_key::Symbol
    "Threshold for |activation| above which a neuron is considered saturated"
    saturation_threshold::Float64
    "Number of timing samples per structure for benchmarking"
    n_timing_samples::Int
end

function EvaluationConfig(;
    energy_key=:dft_energy,
    force_key=:dft_forces,
    stress_key=:dft_stress,
    saturation_threshold=0.01,
    n_timing_samples=5,
)
    EvaluationConfig(energy_key, force_key, stress_key, saturation_threshold, n_timing_samples)
end

# ---------------------------------------------------------------------------
# Per-structure metric structs
# ---------------------------------------------------------------------------

"""
    EnergyMetrics

Per-structure energy comparison between prediction and reference.
"""
struct EnergyMetrics
    "Number of atoms"
    natoms::Int
    "Reference total energy (eV)"
    e_ref::Float64
    "Predicted total energy (eV)"
    e_pred::Float64
end

energy_absolute_error(m::EnergyMetrics) = abs(m.e_pred - m.e_ref)
energy_ae_per_atom(m::EnergyMetrics) = energy_absolute_error(m) / m.natoms

"""
    ForceMetrics

Per-structure force comparison.
"""
struct ForceMetrics
    "Number of atoms"
    natoms::Int
    "Number of spatial dimensions"
    ndims::Int
    "RMSE of force components (eV/Angstrom)"
    component_rmse::Float64
    "MAE of force components (eV/Angstrom)"
    component_mae::Float64
    "Maximum absolute error across all force components"
    max_error::Float64
    "Cosine similarity of the flattened force vectors"
    cosine_similarity::Float64
end

"""
    StressMetrics

Per-structure stress comparison.
"""
struct StressMetrics
    "RMSE across all 9 components of the 3x3 stress tensor (eV/Angstrom^3)"
    component_rmse::Float64
    "Absolute error in hydrostatic pressure (GPa)"
    pressure_error::Float64
end

"""
    LayerSaturationReport

Saturation statistics for one dense layer of the neural network.
"""
struct LayerSaturationReport
    "Layer index (1-based)"
    layer_index::Int
    "Number of neurons in this layer"
    n_neurons::Int
    "Fraction of (neuron, sample) pairs with |activation| > threshold"
    saturated_fraction::Float64
    "Mean absolute pre-activation value"
    mean_abs_preactivation::Float64
end

# ---------------------------------------------------------------------------
# Aggregated metrics
# ---------------------------------------------------------------------------

"""
    AggregatedMetrics

Summary statistics across all evaluated structures.
"""
struct AggregatedMetrics
    n_structures::Int
    total_natoms::Int
    # Energy
    energy_rmse_per_atom::Float64
    energy_mae_per_atom::Float64
    energy_max_ae_per_atom::Float64
    energy_r2::Float64
    # Forces (NaN if unavailable)
    force_component_rmse::Float64
    force_component_mae::Float64
    force_max_error::Float64
    force_mean_cosine_similarity::Float64
    # Stress (NaN if unavailable)
    stress_component_rmse::Float64
    stress_pressure_rmse::Float64
    # Saturation
    layer_saturation_fractions::Vector{Float64}
    saturation_threshold::Float64
    # Timing
    ms_per_atom::Float64
    # Per-structure breakdown
    energy_metrics::Vector{EnergyMetrics}
    force_metrics::Vector{Union{ForceMetrics,Nothing}}
    stress_metrics::Vector{Union{StressMetrics,Nothing}}
end

# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

"""
    _extract_reference_energy(cell, config) -> Float64

Extract reference energy from cell metadata. Falls back through
`config.energy_key` → `:enthalpy` → `:energy`. Throws if none found.
"""
function _extract_reference_energy(cell, config::EvaluationConfig)
    # Try the configured key first
    if haskey(cell.metadata, config.energy_key)
        return Float64(cell.metadata[config.energy_key])
    end
    # Fallbacks for EDDP-native data
    if config.energy_key != :enthalpy && haskey(cell.metadata, :enthalpy)
        return Float64(cell.metadata[:enthalpy])
    end
    if haskey(cell.metadata, :energy)
        return Float64(cell.metadata[:energy])
    end
    error(
        "Cell is missing energy metadata (tried :$(config.energy_key), :enthalpy, :energy)"
    )
end

"""
    _extract_reference_forces(cell, config) -> Union{Matrix{Float64}, Nothing}

Extract reference forces. Checks `arrays` first, then `metadata`.
Returns `nothing` if not present.
"""
function _extract_reference_forces(cell, config::EvaluationConfig)
    # Try the configured key in arrays
    if haskey(cell.arrays, config.force_key)
        return Float64.(cell.arrays[config.force_key])
    end
    # Common fallbacks in arrays
    for key in (:forces, :force)
        if haskey(cell.arrays, key)
            return Float64.(cell.arrays[key])
        end
    end
    # Normalized location in metadata
    if haskey(cell.metadata, :force)
        return Float64.(cell.metadata[:force])
    end
    nothing
end

"""
    _extract_reference_stress(cell, config) -> Union{Matrix{Float64}, Nothing}

Extract and normalize reference stress to a 3x3 matrix (eV/Angstrom^3).
Handles Voigt 6-vector, 9-element row-major, and existing 3x3 matrix.
Returns `nothing` if not present.
"""
function _extract_reference_stress(cell, config::EvaluationConfig)
    raw = get(cell.metadata, config.stress_key, nothing)
    if raw === nothing
        for key in (:virial, :stress)
            raw = get(cell.metadata, key, nothing)
            raw !== nothing && break
        end
    end
    raw === nothing && return nothing

    val = Float64.(raw)

    if val isa AbstractVector && length(val) == 6
        # Voigt notation: xx, yy, zz, yz, xz, xy
        return [
            val[1] val[6] val[5]
            val[6] val[2] val[4]
            val[5] val[4] val[3]
        ]
    elseif val isa AbstractVector && length(val) == 9
        return reshape(val, 3, 3)
    elseif val isa AbstractMatrix && size(val) == (3, 3)
        return val
    end
    @warn "Cannot parse stress of size $(size(val)) from metadata"
    nothing
end

# ---------------------------------------------------------------------------
# Per-structure comparison functions
# ---------------------------------------------------------------------------

# Internal: compute energy+forces+stress in a single calculate! call.
# Returns (energy, forces_matrix, stress_matrix).
function _predict_cell!(calc, cell)
    set_cell!(calc, cell)
    calculate!(calc; forces=true, rebuild_nl=true)
    e = sum(calc.eng) + get_elemental_energy(calc.cell, calc.elemental_energies)
    f = copy(calc.forces)
    s = copy(calc.stress)
    (Float64(e), Float64.(f), Float64.(s))
end

# Internal: compute ForceMetrics from pre-computed prediction and reference.
function _compute_force_metrics(f_pred, f_ref)
    ndims, nat = size(f_ref)
    @assert size(f_pred) == (ndims, nat)

    diff = f_pred .- f_ref
    component_rmse = sqrt(mean(abs2, diff))
    component_mae = mean(abs, diff)
    max_error = maximum(abs, diff)

    f_pred_flat = vec(f_pred)
    f_ref_flat = vec(f_ref)
    norm_pred = norm(f_pred_flat)
    norm_ref = norm(f_ref_flat)
    cos_sim = if norm_pred > 0 && norm_ref > 0
        dot(f_pred_flat, f_ref_flat) / (norm_pred * norm_ref)
    else
        NaN
    end

    ForceMetrics(nat, ndims, component_rmse, component_mae, max_error, cos_sim)
end

# Internal: compute StressMetrics from pre-computed prediction and reference.
function _compute_stress_metrics(s_pred, s_ref)
    diff = s_pred .- s_ref
    component_rmse = sqrt(mean(abs2, diff))
    pressure_error = abs(eVAngToGPa((tr(s_pred) - tr(s_ref)) / 3))
    StressMetrics(component_rmse, pressure_error)
end

"""
    compare_energy(calc, cell, config=EvaluationConfig()) -> EnergyMetrics

Compute predicted energy and compare with reference from cell metadata.
"""
function compare_energy(calc, cell, config::EvaluationConfig=EvaluationConfig())
    e_pred = Float64(eddp_energy(calc, cell))
    e_ref = _extract_reference_energy(cell, config)
    EnergyMetrics(length(cell), e_ref, e_pred)
end

"""
    compare_forces(calc, cell, config=EvaluationConfig()) -> Union{ForceMetrics, Nothing}

Compute predicted forces and compare with reference.
Returns `nothing` if the cell has no reference forces.
"""
function compare_forces(calc, cell, config::EvaluationConfig=EvaluationConfig())
    f_ref = _extract_reference_forces(cell, config)
    f_ref === nothing && return nothing
    f_pred = Float64.(eddp_forces(calc, cell))

    ndims, nat = size(f_ref)
    @assert size(f_pred) == (ndims, nat)

    diff = f_pred .- f_ref
    component_rmse = sqrt(mean(abs2, diff))
    component_mae = mean(abs, diff)
    max_error = maximum(abs, diff)

    # Cosine similarity of the flattened force vectors
    f_pred_flat = vec(f_pred)
    f_ref_flat = vec(f_ref)
    norm_pred = norm(f_pred_flat)
    norm_ref = norm(f_ref_flat)
    cos_sim = if norm_pred > 0 && norm_ref > 0
        dot(f_pred_flat, f_ref_flat) / (norm_pred * norm_ref)
    else
        NaN
    end

    ForceMetrics(nat, ndims, component_rmse, component_mae, max_error, cos_sim)
end

"""
    compare_stress(calc, cell, config=EvaluationConfig()) -> Union{StressMetrics, Nothing}

Compute predicted stress and compare with reference.
Returns `nothing` if the cell has no reference stress data.
"""
function compare_stress(calc, cell, config=EvaluationConfig=EvaluationConfig())
    s_ref = _extract_reference_stress(cell, config)
    s_ref === nothing && return nothing
    s_pred = Float64.(eddp_stress(calc, cell))

    diff = s_pred .- s_ref
    component_rmse = sqrt(mean(abs2, diff))
    pressure_error = abs(eVAngToGPa((tr(s_pred) - tr(s_ref)) / 3))

    StressMetrics(component_rmse, pressure_error)
end

# ---------------------------------------------------------------------------
# Saturation detection
# ---------------------------------------------------------------------------

"""
    detect_saturation(calc::EDDPCalculator, cell; threshold=0.1) -> Vector{LayerSaturationReport}

Run a forward pass on `cell` and report saturation statistics for each hidden
layer (excluding the final linear output layer).

A neuron is considered "saturated" when the local gradient of its activation
function `|dσ/dx| < threshold`. For `leaky_tanh`, the maximum gradient is
`1 + 1e-6 ≈ 1.0`, so `threshold=0.01` means the gradient has dropped below
1% of its maximum — the neuron is barely responsive to input changes.

The last Dense layer (identity activation, gradient always 1.0) is skipped
since it has no saturation behavior.
"""
function detect_saturation(calc, cell; threshold=0.01)
    # Trigger a forward pass (energy-only)
    eddp_energy(calc, cell)
    itf = calc.nninterface
    _detect_saturation_from_interface(itf; threshold)
end

function _detect_saturation_from_interface(itf; threshold=0.01)
    if !(itf isa ManualFluxBackPropInterface)
        return LayerSaturationReport[]
    end
    if isempty(itf.gchains)
        return LayerSaturationReport[]
    end
    cg = itf.gchains[itf.last_id]
    nlayers = length(cg.layers)
    reports = LayerSaturationReport[]
    for (i, layer) in enumerate(cg.layers)
        if !isa(layer, EDDPotentialsCore.DenseGradient)
            continue
        end
        # Skip the last layer (typically identity/linear — no saturation)
        chain = itf.chain
        if i == nlayers && i <= length(chain.layers) && chain.layers[i].σ === identity
            continue
        end
        # Compute local gradient dσ/dx for each (pre-activation, output) pair
        wx = layer.wx   # pre-activation (W*x + b)
        out = layer.out  # post-activation σ.(wx)
        n_total = length(wx)
        n_saturated = 0
        for j in eachindex(wx)
            g = layer.gσ(wx[j], out[j])
            if abs(g) < threshold
                n_saturated += 1
            end
        end
        push!(
            reports,
            LayerSaturationReport(
                i, size(out, 1), n_saturated / n_total, mean(abs, wx)
            ),
        )
    end
    reports
end

# ---------------------------------------------------------------------------
# Timing / benchmarking
# ---------------------------------------------------------------------------

"""
    benchmark_prediction(calc, cell; n_samples=5) -> Float64

Time energy+forces prediction over `n_samples` runs for a single cell.
Returns milliseconds per atom.
"""
function benchmark_prediction(calc, cell; n_samples=5)
    # Warmup
    eddp_energy(calc, cell)
    eddp_forces(calc, cell)

    times = Float64[]
    for _ in 1:n_samples
        t = @elapsed begin
            eddp_energy(calc, cell)
            eddp_forces(calc, cell)
        end
        push!(times, t)
    end
    mean_ms = mean(times) * 1000
    mean_ms / length(cell)
end

"""
    benchmark_prediction(calc, cells::AbstractVector{<:Cell}; n_samples=5) -> Float64

Benchmark prediction across multiple cells. Returns aggregate milliseconds
per atom (total wall time / total atoms).
"""
function benchmark_prediction(calc, cells::AbstractVector{<:Cell}; n_samples=5)
    isempty(cells) && return NaN

    # Warmup on first cell
    eddp_energy(calc, cells[1])
    eddp_forces(calc, cells[1])

    total_time = 0.0
    total_natoms = 0
    for _ in 1:n_samples
        for cell in cells
            t = @elapsed begin
                eddp_energy(calc, cell)
                eddp_forces(calc, cell)
            end
            total_time += t
            total_natoms += length(cell)
        end
    end
    (total_time * 1000) / total_natoms
end

# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

"""
    _r2_score(y_true, y_pred) -> Float64

Standard R² (coefficient of determination).
"""
function _r2_score(y_true::AbstractVector, y_pred::AbstractVector)
    ss_res = sum((y_true .- y_pred) .^ 2)
    ss_tot = sum((y_true .- mean(y_true)) .^ 2)
    ss_tot == 0 ? NaN : 1.0 - ss_res / ss_tot
end

# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

"""
    evaluate(calc, cells; config=EvaluationConfig(), compute_forces=true,
             compute_stress=true, compute_timing=true, compute_saturation=true)
    -> AggregatedMetrics

Evaluate an `EDDPCalculator` against reference data stored in Cell metadata/arrays.

Returns an `AggregatedMetrics` struct with per-structure breakdowns and
summary statistics.
"""
function evaluate(
    calc,
    cells::AbstractVector{<:Cell};
    config=EvaluationConfig(),
    compute_forces=true,
    compute_stress=true,
    compute_timing=true,
    compute_saturation=true,
)
    n = length(cells)
    e_metrics = Vector{EnergyMetrics}(undef, n)
    f_metrics = Vector{Union{ForceMetrics,Nothing}}(undef, n)
    s_metrics = Vector{Union{StressMetrics,Nothing}}(undef, n)
    sat_fractions_by_layer = Vector{Vector{Float64}}()
    ms_per_atom = NaN

    for i in eachindex(cells)
        cell = cells[i]
        need_forces = compute_forces || compute_stress

        # Single calculate! call per cell
        if need_forces
            e_pred, f_pred, s_pred = _predict_cell!(calc, cell)
        else
            e_pred = Float64(eddp_energy(calc, cell))
            f_pred = nothing
            s_pred = nothing
        end

        # Energy (always)
        e_ref = _extract_reference_energy(cell, config)
        e_metrics[i] = EnergyMetrics(length(cell), e_ref, e_pred)

        # Forces
        f_metrics[i] = if compute_forces
            f_ref = _extract_reference_forces(cell, config)
            f_ref === nothing ? nothing : _compute_force_metrics(f_pred, f_ref)
        else
            nothing
        end

        # Stress
        s_metrics[i] = if compute_stress
            s_ref = _extract_reference_stress(cell, config)
            s_ref === nothing ? nothing : _compute_stress_metrics(s_pred, s_ref)
        else
            nothing
        end

        # Saturation (accumulated) — reuse the forward pass from _predict_cell!
        if compute_saturation
            sat = _detect_saturation_from_interface(
                calc.nninterface; threshold=config.saturation_threshold
            )
            if !isempty(sat)
                if isempty(sat_fractions_by_layer)
                    for _ in sat
                        push!(sat_fractions_by_layer, Float64[])
                    end
                end
                for (li, report) in enumerate(sat)
                    push!(sat_fractions_by_layer[li], report.saturated_fraction)
                end
            end
        end
    end

    # Timing
    if compute_timing
        ms_per_atom = benchmark_prediction(calc, cells; n_samples=config.n_timing_samples)
    end

    # --- Aggregate ---

    total_nat = sum(m.natoms for m in e_metrics)
    e_errors_pa = [energy_ae_per_atom(m) for m in e_metrics]
    energy_rmse_pa = sqrt(mean(e_errors_pa .^ 2))
    energy_mae_pa = mean(e_errors_pa)
    energy_max_ae_pa = maximum(e_errors_pa)

    e_ref_pa = [m.e_ref / m.natoms for m in e_metrics]
    e_pred_pa = [m.e_pred / m.natoms for m in e_metrics]
    energy_r2 = _r2_score(e_ref_pa, e_pred_pa)

    # Forces
    valid_f = filter(!isnothing, f_metrics)
    force_component_rmse = isempty(valid_f) ? NaN : sqrt(mean(m.component_rmse^2 for m in valid_f))
    force_component_mae = isempty(valid_f) ? NaN : mean(m.component_mae for m in valid_f)
    force_max_err = isempty(valid_f) ? NaN : maximum(m.max_error for m in valid_f)
    force_cos_sim = isempty(valid_f) ? NaN : mean(m.cosine_similarity for m in valid_f)

    # Stress
    valid_s = filter(!isnothing, s_metrics)
    stress_rmse = isempty(valid_s) ? NaN : sqrt(mean(m.component_rmse^2 for m in valid_s))
    stress_pressure_rmse = isempty(valid_s) ? NaN : sqrt(mean(m.pressure_error^2 for m in valid_s))

    # Saturation
    layer_sat_fracs = [mean(fracs) for fracs in sat_fractions_by_layer]

    AggregatedMetrics(
        n, total_nat,
        energy_rmse_pa, energy_mae_pa, energy_max_ae_pa, energy_r2,
        force_component_rmse, force_component_mae, force_max_err, force_cos_sim,
        stress_rmse, stress_pressure_rmse,
        layer_sat_fracs, config.saturation_threshold,
        ms_per_atom,
        e_metrics, f_metrics, s_metrics,
    )
end

"""
    evaluate(calc, sc::StructureContainer; config=EvaluationConfig(energy_key=:enthalpy,
             force_key=:REF_forces), kwargs...) -> AggregatedMetrics

Evaluate an `EDDPCalculator` directly against the structures in a
`StructureContainer`. The default reference keys match `StructureContainer`'s
normalised data convention: energies in `cell.metadata[:enthalpy]` and forces
in `cell.arrays[:REF_forces]`.
"""
function evaluate(
    calc,
    sc::StructureContainer;
    config=EvaluationConfig(energy_key=:enthalpy, force_key=:REF_forces),
    kwargs...,
)
    evaluate(calc, sc.structures; config, kwargs...)
end

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

"""
    load_evaluation_data(paths) -> Vector{Cell}

Load structures from file paths for evaluation. Accepts glob patterns
and the same `@slice` syntax as `StructureContainer`.
"""
function load_evaluation_data(paths::AbstractVector)
    cells = Cell[]
    for path in paths
        filepath, _ = _parse_path_slice(path)
        if contains(filepath, "*") || contains(filepath, "?")
            resolved = glob_allow_abs(filepath)
        else
            resolved = [filepath]
        end
        for fp in resolved
            append!(cells, _load_structures_from_path(fp))
        end
    end
    cells
end

load_evaluation_data(path::AbstractString) = load_evaluation_data([path])

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", m::AggregatedMetrics)
    println(io, "EDDP Evaluation Report")
    println(io, "="^50)
    @printf(io, "Structures:       %d  (total atoms: %d)\n", m.n_structures, m.total_natoms)
    println(io)
    println(io, "Energy (eV/atom):")
    @printf(io, "  RMSE:           %.6f\n", m.energy_rmse_per_atom)
    @printf(io, "  MAE:            %.6f\n", m.energy_mae_per_atom)
    @printf(io, "  Max AE:         %.6f\n", m.energy_max_ae_per_atom)
    @printf(io, "  R2:             %.6f\n", m.energy_r2)

    if !isnan(m.force_component_rmse)
        println(io)
        println(io, "Forces (eV/Angstrom):")
        @printf(io, "  RMSE:           %.6f\n", m.force_component_rmse)
        @printf(io, "  MAE:            %.6f\n", m.force_component_mae)
        @printf(io, "  Max error:      %.6f\n", m.force_max_error)
        @printf(io, "  Cosine sim:     %.6f\n", m.force_mean_cosine_similarity)
    end

    if !isnan(m.stress_component_rmse)
        println(io)
        println(io, "Stress:")
        @printf(io, "  RMSE:           %.6f eV/Ang^3\n", m.stress_component_rmse)
        @printf(io, "  Pressure RMSE:  %.6f GPa\n", m.stress_pressure_rmse)
    end

    if !isempty(m.layer_saturation_fractions)
        println(io)
        @printf(io, "NN Saturation (|dσ/dx| < %.4f):\n", m.saturation_threshold)
        for (i, frac) in enumerate(m.layer_saturation_fractions)
            @printf(io, "  Layer %d: %.2f%% saturated\n", i, frac * 100)
        end
    end

    if !isnan(m.ms_per_atom)
        println(io)
        @printf(io, "Performance:      %.4f ms/atom\n", m.ms_per_atom)
    end
end
