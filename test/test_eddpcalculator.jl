using EDDPotentials
using EDDPotentials: NNCalc, eddp_energy, eddp_forces, eddp_stress, nfeatures
using EDDPotentials.EDDPotentialsCore:
    LinearInterface,
    ManualFluxBackPropInterface,
    EnsembleNNInterface,
    save_model_artifact,
    forward!,
    feature_vector
using Flux: Chain, Dense, tanh
using CellBase
using Test

isdefined(Main, :_h2_cell) || include("utils.jl")

@testset "EDDPCalculator" begin

    # Build a simple model for testing
    cell = _h2_cell()
    cf = CellFeature([:H]; p2=[2.0, 3.0], p3=Float64[], q3=Float64[])
    model = LinearInterface(collect(range(0.1, length=nfeatures(cf), step=0.1)))
    elemental_energies = Dict(:H => -1.2)

    # Reference energy via the existing API
    ref_calc = NNCalc(deepcopy(cell), cf, model; elemental_energies, core=nothing)
    ref_energy = eddp_energy(ref_calc)
    ref_forces = eddp_forces(ref_calc)
    ref_stress = eddp_stress(ref_calc)

    @testset "load_calculator from JSON" begin
        mktempdir() do dir
            path = joinpath(dir, "model.json")
            save_model_artifact(path, model; feature_spec=cf, elemental_energies)

            calc = load_calculator(path)
            @test calc isa EDDPCalculator
            @test calc._nncalc === nothing  # no cell bound yet
        end
    end

    @testset "Two-argument prediction" begin
        mktempdir() do dir
            path = joinpath(dir, "model.json")
            save_model_artifact(path, model; feature_spec=cf, elemental_energies)
            calc = load_calculator(path; core_repulsion=nothing)

            e = eddp_energy(calc, deepcopy(cell))
            @test e ≈ ref_energy

            f = eddp_forces(calc, deepcopy(cell))
            @test f ≈ ref_forces

            s = eddp_stress(calc, deepcopy(cell))
            @test s ≈ ref_stress

            e_per = eddp_energy_per_atom(calc, deepcopy(cell))
            @test e_per ≈ ref_energy / length(cell)
        end
    end

    @testset "Cell binding and single-arg methods" begin
        mktempdir() do dir
            path = joinpath(dir, "model.json")
            save_model_artifact(path, model; feature_spec=cf, elemental_energies)
            calc = load_calculator(path; core_repulsion=nothing)

            set_cell!(calc, deepcopy(cell))
            @test calc._nncalc !== nothing
            @test length(get_cell(calc)) == length(cell)

            e = eddp_energy(calc)
            @test e ≈ ref_energy

            f = eddp_forces(calc)
            @test f ≈ ref_forces

            s = eddp_stress(calc)
            @test s ≈ ref_stress
        end
    end

    @testset "Unbound calculator errors" begin
        mktempdir() do dir
            path = joinpath(dir, "model.json")
            save_model_artifact(path, model; feature_spec=cf, elemental_energies)
            calc = load_calculator(path)

            @test_throws ErrorException eddp_energy(calc)
            @test_throws ErrorException eddp_forces(calc)
            @test_throws ErrorException get_cell(calc)
        end
    end

    @testset "Workspace reuse" begin
        mktempdir() do dir
            path = joinpath(dir, "model.json")
            save_model_artifact(path, model; feature_spec=cf, elemental_energies)
            calc = load_calculator(path; core_repulsion=nothing)

            # First prediction (creates NNCalc)
            e1 = eddp_energy(calc, deepcopy(cell))
            nnc1 = calc._nncalc
            @test nnc1 !== nothing

            # Second prediction with same-size cell (should reuse workspace)
            e2 = eddp_energy(calc, deepcopy(cell))
            @test e2 ≈ e1
            @test calc._nncalc === nnc1  # same NNCalc object

            # Prediction with different atom count (should create new NNCalc)
            cell3 = Cell(Lattice(4.0, 4.0, 4.0), [:H, :H, :H],
                         [0.0 0.1 0.2; 0.0 0.0 0.1; 0.1 0.0 0.0])
            e3 = eddp_energy(calc, cell3)
            @test calc._nncalc !== nnc1  # new NNCalc created

            # Prediction again with 2-atom cell (should create yet another new NNCalc)
            e4 = eddp_energy(calc, deepcopy(cell))
            @test e4 ≈ e1
        end
    end

    @testset "EDDPCalculator from ModelArtifact" begin
        mktempdir() do dir
            path = joinpath(dir, "model.json")
            save_model_artifact(path, model; feature_spec=cf, elemental_energies)
            artifact = load_model_artifact(path)

            calc = EDDPCalculator(artifact; core_repulsion=nothing)
            @test calc isa EDDPCalculator

            e = eddp_energy(calc, deepcopy(cell))
            @test e ≈ ref_energy
        end
    end

    @testset "extract_nncalc" begin
        mktempdir() do dir
            path = joinpath(dir, "model.json")
            save_model_artifact(path, model; feature_spec=cf, elemental_energies)
            calc = load_calculator(path; core_repulsion=nothing)

            @test_throws ErrorException extract_nncalc(calc)

            set_cell!(calc, deepcopy(cell))
            nnc = extract_nncalc(calc)
            @test nnc isa NNCalc
            @test eddp_energy(nnc) ≈ ref_energy
        end
    end

    @testset "Optimization via set_cell!" begin
        mktempdir() do dir
            path = joinpath(dir, "model.json")
            save_model_artifact(path, model; feature_spec=cf, elemental_energies)
            calc = load_calculator(path; core_repulsion=nothing)

            # Displace atoms from equilibrium
            test_cell = deepcopy(cell)
            set_positions!(test_cell, positions(test_cell) .+ 0.1)
            set_cell!(calc, test_cell)

            # Run a few optimization steps
            res = opt_tpsd(calc; itermax=5)
            # Just verify it runs without error and returns something
            @test res !== nothing
        end
    end

    @testset "Ensemble model" begin
        model1 = LinearInterface(collect(range(0.1, length=nfeatures(cf), step=0.1)))
        model2 = LinearInterface(collect(range(0.2, length=nfeatures(cf), step=0.1)))
        ensemble = EnsembleNNInterface((model1, model2), [0.25, 0.75])

        mktempdir() do dir
            path = joinpath(dir, "ensemble.json")
            save_model_artifact(
                path, ensemble;
                feature_spec=cf,
                elemental_energies,
            )

            calc = load_calculator(path; core_repulsion=nothing)
            e = eddp_energy(calc, deepcopy(cell))
            @test e isa Float64

            # Also test energy_std
            e_std = eddp_energy_std(calc, deepcopy(cell))
            @test e_std >= 0
        end
    end
end
