include("utils.jl")

using Flux: Chain, Dense
using EDDPotentials
using EDDPotentials:
    EvaluationConfig,
    EnergyMetrics,
    ForceMetrics,
    StressMetrics,
    LayerSaturationReport,
    AggregatedMetrics,
    _extract_reference_energy,
    _extract_reference_forces,
    _extract_reference_stress,
    _r2_score,
    energy_ae_per_atom,
    compare_energy,
    compare_forces,
    compare_stress,
    load_evaluation_data
using EDDPotentials.Potential: ManualFluxBackPropInterface, DenseGradient
using CellBase
using Test

# ---------------------------------------------------------------------------
# Helper: create an EDDPCalculator for testing
# ---------------------------------------------------------------------------

function _make_eddp_calc()
    cell = _lco_cell()
    cf = _generate_cf(cell)
    itf = ManualFluxBackPropInterface(
        Chain(Dense(rand(5, nfeatures(cf))), Dense(rand(1, 5))),
    )
    ee = Dict(:Li => -1.0, :Co => -2.0, :O => -0.5)
    EDDPCalculator(cf, itf, ee)
end

function _make_simple_eddp_calc()
    cell = _h2_cell()
    cf = _generate_cf(cell)
    itf = ManualFluxBackPropInterface(
        Chain(Dense(rand(5, nfeatures(cf))), Dense(rand(1, 5))),
    )
    ee = Dict(:H => -0.5)
    EDDPCalculator(cf, itf, ee)
end

function _cell_with_refs(; natoms=2, e_ref=-10.0, f_ref=nothing, s_ref=nothing)
    cell = _h2_cell()
    cell.metadata[:dft_energy] = e_ref
    if f_ref !== nothing
        cell.arrays[:dft_forces] = f_ref
    end
    if s_ref !== nothing
        cell.metadata[:dft_stress] = s_ref
    end
    cell
end

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@testset "EvaluationConfig" begin
    c = EvaluationConfig()
    @test c.energy_key == :dft_energy
    @test c.force_key == :dft_forces
    @test c.stress_key == :dft_stress
    @test c.saturation_threshold == 0.01
    @test c.n_timing_samples == 5

    c2 = EvaluationConfig(energy_key=:enthalpy, force_key=:force, n_timing_samples=3)
    @test c2.energy_key == :enthalpy
    @test c2.n_timing_samples == 3
end

@testset "EnergyMetrics" begin
    m = EnergyMetrics(4, -10.0, -10.5)
    @test m.natoms == 4
    @test m.e_ref == -10.0
    @test m.e_pred == -10.5
    @test energy_ae_per_atom(m) ≈ 0.125
end

@testset "_extract_reference_energy" begin
    config = EvaluationConfig()
    cell = Cell(Lattice(4.0, 4.0, 4.0), [:H], [[0.0, 0.0, 0.0]])

    # dft_energy (default key)
    cell.metadata[:dft_energy] = -5.0
    @test _extract_reference_energy(cell, config) == -5.0

    # Fallback to :enthalpy
    delete!(cell.metadata, :dft_energy)
    cell.metadata[:enthalpy] = -6.0
    @test _extract_reference_energy(cell, config) == -6.0

    # Fallback to :energy
    delete!(cell.metadata, :enthalpy)
    cell.metadata[:energy] = -7.0
    @test _extract_reference_energy(cell, config) == -7.0

    # Error when nothing
    delete!(cell.metadata, :energy)
    @test_throws ErrorException _extract_reference_energy(cell, config)

    # Custom key
    config2 = EvaluationConfig(energy_key=:my_energy)
    cell.metadata[:my_energy] = -8.0
    @test _extract_reference_energy(cell, config2) == -8.0
end

@testset "_extract_reference_forces" begin
    config = EvaluationConfig()
    cell = Cell(Lattice(4.0, 4.0, 4.0), [:H], [[0.0, 0.0, 0.0]])

    # Nothing when absent
    @test _extract_reference_forces(cell, config) === nothing

    # In arrays with default key
    f = [0.1 0.2; 0.3 0.4; 0.5 0.6]
    cell.arrays[:dft_forces] = f
    result = _extract_reference_forces(cell, config)
    @test result ≈ f

    # Fallback to :forces
    delete!(cell.arrays, :dft_forces)
    cell.arrays[:forces] = f .+ 1.0
    result = _extract_reference_forces(cell, config)
    @test result ≈ f .+ 1.0
end

@testset "_extract_reference_stress" begin
    config = EvaluationConfig()
    cell = Cell(Lattice(4.0, 4.0, 4.0), [:H], [[0.0, 0.0, 0.0]])

    # Nothing when absent
    @test _extract_reference_stress(cell, config) === nothing

    # Voigt 6-vector
    voigt = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # xx, yy, zz, yz, xz, xy
    cell.metadata[:dft_stress] = voigt
    s = _extract_reference_stress(cell, config)
    @test s[1, 1] ≈ 1.0   # xx
    @test s[2, 2] ≈ 2.0   # yy
    @test s[3, 3] ≈ 3.0   # zz
    @test s[2, 3] ≈ 4.0   # yz
    @test s[1, 3] ≈ 5.0   # xz
    @test s[1, 2] ≈ 6.0   # xy
    @test s[3, 2] ≈ 4.0   # yz (symmetric)
    @test s[3, 1] ≈ 5.0   # xz (symmetric)
    @test s[2, 1] ≈ 6.0   # xy (symmetric)

    # 9-element vector (row-major)
    delete!(cell.metadata, :dft_stress)
    full9 = collect(1.0:9.0)
    cell.metadata[:stress] = full9
    s2 = _extract_reference_stress(cell, config)
    @test size(s2) == (3, 3)
    @test s2[1, 1] ≈ 1.0
    @test s2[3, 3] ≈ 9.0

    # 3x3 matrix
    delete!(cell.metadata, :stress)
    mat = reshape(collect(1.0:9.0), 3, 3)
    cell.metadata[:virial] = mat
    s3 = _extract_reference_stress(cell, config)
    @test s3 ≈ mat
end

@testset "compare_energy" begin
    calc = _make_simple_eddp_calc()
    cell = _cell_with_refs(e_ref=-10.0)
    config = EvaluationConfig()
    m = compare_energy(calc, cell, config)
    @test m.natoms == 2
    @test m.e_ref == -10.0
    @test isa(m.e_pred, Float64)
        @test isfinite(m.e_pred)
        @test energy_ae_per_atom(m) ≈ abs(m.e_pred - m.e_ref) / m.natoms
end

@testset "compare_forces" begin
    calc = _make_simple_eddp_calc()
    config = EvaluationConfig()

    # Without reference forces
    cell = _cell_with_refs()
    @test compare_forces(calc, cell, config) === nothing

    # With reference forces
    cell = _cell_with_refs(f_ref=rand(3, 2) .* 0.1)
    m = compare_forces(calc, cell, config)
    @test m !== nothing
    @test m isa ForceMetrics
    @test m.natoms == 2
    @test m.ndims == 3
    @test m.component_rmse >= 0
    @test m.component_mae >= 0
    @test m.max_error >= 0
    @test m.component_rmse >= m.component_mae
    @test m.max_error >= m.component_mae
    @test isfinite(m.component_rmse)
end

@testset "compare_stress" begin
    calc = _make_simple_eddp_calc()
    config = EvaluationConfig()

    # Without reference stress
    cell = _cell_with_refs()
    @test compare_stress(calc, cell, config) === nothing

    # With reference stress (Voigt)
    cell = _cell_with_refs(s_ref=rand(6) .* 0.01)
    m = compare_stress(calc, cell, config)
    @test m !== nothing
    @test m isa StressMetrics
    @test m.component_rmse >= 0
    @test m.pressure_error >= 0
    @test isfinite(m.component_rmse)
end

@testset "detect_saturation" begin
    calc = _make_simple_eddp_calc()
    cell = _h2_cell()

    # Default threshold is 0.01 (gradient-based)
    reports = detect_saturation(calc, cell)
    @test isa(reports, Vector{LayerSaturationReport})
    # The test model has 2 Dense layers: hidden (leaky_tanh) + output (identity)
    # The output layer (identity, gradient always 1.0) should be skipped
    @test length(reports) == 1
    for r in reports
        @test r.saturated_fraction >= 0.0
        @test r.saturated_fraction <= 1.0
        @test r.n_neurons > 0
        @test r.mean_abs_preactivation >= 0
    end

    # With a very strict threshold, should report more saturation
    reports_strict = detect_saturation(calc, cell; threshold=0.5)
    @test length(reports_strict) == 1
    # Stricter threshold means more neurons are "saturated"
    @test reports_strict[1].saturated_fraction >= reports[1].saturated_fraction
end

@testset "_r2_score" begin
    # Perfect prediction
    y = [1.0, 2.0, 3.0, 4.0]
    @test _r2_score(y, y) ≈ 1.0

    # Constant prediction (predict mean)
    @test _r2_score(y, fill(mean(y), 4)) ≈ 0.0

    # Bad prediction
    @test _r2_score(y, reverse(y)) < 1.0
end

@testset "benchmark_prediction single cell" begin
    calc = _make_simple_eddp_calc()
    cell = _h2_cell()
    ms = benchmark_prediction(calc, cell; n_samples=2)
    @test ms > 0
    @test isa(ms, Float64)
end

@testset "benchmark_prediction multiple cells" begin
    calc = _make_simple_eddp_calc()
    cells = [_h2_cell(), _h2_cell(5.0), _h2_cell(3.5)]
    ms = benchmark_prediction(calc, cells; n_samples=2)
    @test ms > 0
    @test isa(ms, Float64)
end

@testset "evaluate" begin
    calc = _make_simple_eddp_calc()

    # Create a small test set with reference data
    cells = Cell[]
    for i in 1:3
        c = _h2_cell(4.0 + 0.1 * i)
        c.metadata[:dft_energy] = -10.0 - i
        c.arrays[:dft_forces] = rand(3, 2) .* 0.1
        c.metadata[:dft_stress] = rand(6) .* 0.01
        push!(cells, c)
    end

    result = evaluate(calc, cells)
    @test result.n_structures == 3
    @test result.total_natoms == 6
    @test result.energy_rmse_per_atom >= 0
    @test result.energy_mae_per_atom >= 0
    @test result.energy_max_ae_per_atom >= 0
    @test result.energy_r2 <= 1.0
    @test !isnan(result.force_component_rmse)
    @test !isnan(result.stress_component_rmse)
    @test !isnan(result.ms_per_atom)
    @test length(result.energy_metrics) == 3
    @test length(result.force_metrics) == 3
    @test length(result.stress_metrics) == 3
end

@testset "evaluate with missing force/stress" begin
    calc = _make_simple_eddp_calc()

    # Cells with only energy
    cells = Cell[]
    for i in 1:2
        c = _h2_cell(4.0 + 0.1 * i)
        c.metadata[:dft_energy] = -10.0 - i
        push!(cells, c)
    end

    result = evaluate(calc, cells; compute_timing=false, compute_saturation=false)
    @test result.n_structures == 2
    @test isnan(result.force_component_rmse)
    @test isnan(result.stress_component_rmse)
    @test all(isnothing, result.force_metrics)
    @test all(isnothing, result.stress_metrics)
end

@testset "AggregatedMetrics display" begin
    calc = _make_simple_eddp_calc()
    cells = Cell[]
    for i in 1:2
        c = _h2_cell(4.0 + 0.1 * i)
        c.metadata[:dft_energy] = -10.0 - i
        c.arrays[:dft_forces] = rand(3, 2) .* 0.1
        c.metadata[:dft_stress] = rand(6) .* 0.01
        push!(cells, c)
    end
    result = evaluate(calc, cells; compute_timing=false, compute_saturation=false)

    # Just check it doesn't error and produces output
    s = sprint(show, MIME"text/plain"(), result)
    @test occursin("EDDP Evaluation Report", s)
    @test occursin("Energy", s)
    @test occursin("RMSE", s)
    @test occursin("Forces", s)
    @test occursin("Stress", s)
    # Saturation only appears when compute_saturation=true with data
end

@testset "evaluate with EDDP-native keys" begin
    # Test using enthalpy/force keys (as used in EDDP training data)
    calc = _make_simple_eddp_calc()
    config = EvaluationConfig(energy_key=:enthalpy, force_key=:force)

    cell = _h2_cell()
    cell.metadata[:enthalpy] = -10.0
    cell.metadata[:force] = rand(3, 2) .* 0.1

    result = evaluate(calc, [cell]; config, compute_timing=false, compute_saturation=false)
    @test result.n_structures == 1
    @test !isnan(result.force_component_rmse)
end

@testset "evaluate StructureContainer" begin
    calc = _make_simple_eddp_calc()

    cells = typeof(_h2_cell())[]
    energies = Float64[]
    for i in 1:2
        cell = _h2_cell(4.0 + 0.1 * i)
        energy = -10.0 - i
        cell.metadata[:enthalpy] = energy
        cell.arrays[:REF_forces] = rand(3, 2) .* 0.1
        push!(cells, cell)
        push!(energies, energy)
    end
    sc = StructureContainer(cells, energies)

    result = evaluate(calc, sc; compute_timing=false, compute_saturation=false)
    @test result.n_structures == 2
    @test !isnan(result.force_component_rmse)
end

@testset "load_evaluation_data paths" begin
    mktempdir() do dir
        cell1 = _h2_cell()
        cell2 = _h2_cell(4.2)
        CellBase.write_res(joinpath(dir, "eval_1.res"), cell1)
        CellBase.write_res(joinpath(dir, "eval_2.res"), cell2)

        cells = load_evaluation_data(joinpath(dir, "eval_*.res"))
        @test length(cells) == 2
    end
end
