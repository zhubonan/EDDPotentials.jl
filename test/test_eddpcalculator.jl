using EDDPotentials
using EDDPotentials:
    EDDPCalculator,
    eddp_energy,
    eddp_forces,
    eddp_stress,
    eddp_pressure,
    eddp_pressure_gpa,
    eddp_pressure_GPa,
    nfeatures,
    CoreRepulsion,
    create_calculator,
    predict,
    eddp_energy_per_atom,
    eddp_energy_std
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

    cell = _h2_cell()
    cf = CellFeature([:H]; p2=[2.0, 3.0], p3=Float64[], q3=Float64[])
    model = LinearInterface(collect(range(0.1, length=nfeatures(cf), step=0.1)))
    elemental_energies = Dict(:H => -1.2)

    ref_calc = EDDPCalculator(deepcopy(cell), cf, model; elemental_energies, core_repulsion=nothing)
    ref_energy = eddp_energy(ref_calc)
    ref_forces = eddp_forces(ref_calc)
    ref_stress = eddp_stress(ref_calc)

    @testset "load_calculator from JSON" begin
        mktempdir() do dir
            path = joinpath(dir, "model.json")
            save_model_artifact(path, model; feature_spec=cf, elemental_energies)

            calc = load_calculator(path)
            @test calc isa EDDPCalculator
            @test calc.cell === nothing
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

            p = eddp_pressure(calc, deepcopy(cell))
            @test p ≈ (ref_stress[1, 1] + ref_stress[2, 2] + ref_stress[3, 3]) / 3

            p_gpa = eddp_pressure_gpa(calc, deepcopy(cell))
            @test p_gpa ≈ eddp_pressure_gpa(ref_calc)

            p_GPa = eddp_pressure_GPa(calc, deepcopy(cell))
            @test p_GPa ≈ p_gpa

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
            @test calc.cell !== nothing
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

            e1 = eddp_energy(calc, deepcopy(cell))
            ws1 = calc.workspace
            @test ws1 !== nothing

            e2 = eddp_energy(calc, deepcopy(cell))
            @test e2 ≈ e1
            @test calc.workspace === ws1

            cell3 = Cell(Lattice(4.0, 4.0, 4.0), [:H, :H, :H],
                         [0.0 0.1 0.2; 0.0 0.0 0.1; 0.1 0.0 0.0])
            e3 = eddp_energy(calc, cell3)
            @test calc.workspace !== ws1

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

    @testset "Optimization via set_cell!" begin
        mktempdir() do dir
            path = joinpath(dir, "model.json")
            save_model_artifact(path, model; feature_spec=cf, elemental_energies)
            calc = load_calculator(path; core_repulsion=nothing)

            test_cell = deepcopy(cell)
            set_positions!(test_cell, positions(test_cell) .+ 0.1)
            set_cell!(calc, test_cell)

            res = opt_tpsd(calc; itermax=5)
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

            e_std = eddp_energy_std(calc, deepcopy(cell))
            @test e_std >= 0
        end
    end

    @testset "Cache invalidation" begin
        calc = EDDPCalculator(deepcopy(cell), cf, model; elemental_energies, core_repulsion=nothing)

        e1 = eddp_energy(calc)
        f1 = copy(eddp_forces(calc))
        @test calc._energy_calculated
        @test calc._forces_calculated

        # Position change: should invalidate and recalculate
        pos = positions(get_cell(calc))
        pos[1, 1] += 0.05
        set_positions!(calc, pos)
        @test calc._forces_calculated
        e2 = eddp_energy(calc)
        @test e2 != e1
        @test calc._energy_calculated

        # Lattice change: should invalidate
        cmat = cellmat(get_cell(calc))
        cmat[1, 1] *= 1.01
        set_cellmat!(calc, cmat)
        e3 = eddp_energy(calc)
        @test calc._energy_calculated

        # Species change: should invalidate via set_cell!
        cf_multi = CellFeature([:H, :He]; p2=[2.0], p3=Float64[], q3=Float64[])
        model_multi = LinearInterface(collect(range(0.1, length=nfeatures(cf_multi), step=0.1)))
        cell_multi = Cell(Lattice(6.0, 6.0, 6.0), [:H, :H],
                          [0.0 1.0; 0.0 0.0; 0.0 0.0])
        calc_multi = EDDPCalculator(cell_multi, cf_multi, model_multi; elemental_energies=Dict{Symbol,Float64}(), core_repulsion=nothing)
        e_a = eddp_energy(calc_multi)
        cell_modified = deepcopy(get_cell(calc_multi))
        CellBase.species(cell_modified)[1] = :He
        set_cell!(calc_multi, cell_modified)
        @test calc_multi._forces_calculated == false
        e_b = eddp_energy(calc_multi)
        @test calc_multi._energy_calculated
    end

    @testset "Force caching — energy-only then force" begin
        calc = EDDPCalculator(deepcopy(cell), cf, model; elemental_energies, core_repulsion=nothing)

        e1 = eddp_energy(calc; forces=false)
        @test calc._energy_calculated
        @test calc._forces_calculated == false

        f1 = eddp_forces(calc)
        @test calc._forces_calculated
    end

    @testset "core=nothing backward compat" begin
        calc = EDDPCalculator(deepcopy(cell), cf, model; elemental_energies, core=nothing)
        @test calc.core_repulsion === nothing

        calc2 = EDDPCalculator(deepcopy(cell), cf, model; elemental_energies)
        @test calc2.core_repulsion isa CoreRepulsion
    end

    @testset "create_calculator independence" begin
        model_calc = EDDPCalculator(cf, model, elemental_energies; core_repulsion=nothing)
        @test model_calc.cell === nothing

        bound = create_calculator(model_calc, deepcopy(cell))
        @test bound.cell !== nothing

        # Model-level calculator must remain unbound
        @test model_calc.cell === nothing
        @test model_calc.workspace === nothing

        # Bound calculator must give correct energy independently
        e_model = eddp_energy(ref_calc)
        e_bound = eddp_energy(bound)
        @test e_bound ≈ e_model

        # Modify the bound calculator — model-level must be unaffected
        pos = positions(get_cell(bound))
        pos[1, 1] += 0.1
        set_positions!(bound, pos)
        e_bound2 = eddp_energy(bound)
        # Model-level calc is still unbound
        @test model_calc.cell === nothing
        # Creating a fresh bound calc from the model should match original
        fresh = create_calculator(model_calc, deepcopy(cell))
        @test eddp_energy(fresh) ≈ e_model
    end

    @testset "predict function" begin
        calc = EDDPCalculator(deepcopy(cell), cf, model; elemental_energies, core_repulsion=nothing)

        result = predict(calc, cell)
        @test result.energy isa Float64
        @test result.forces isa Matrix
        @test result.stress isa Matrix

        result_min = predict(calc, cell; forces=false, stress=false)
        @test !haskey(result_min, :forces)
        @test !haskey(result_min, :stress)
        @test result.energy ≈ result_min.energy
    end

    @testset "eddp_energy_per_atom" begin
        calc = EDDPCalculator(deepcopy(cell), cf, model; elemental_energies, core_repulsion=nothing)
        e = eddp_energy(calc)
        e_pa = eddp_energy_per_atom(calc)
        @test e_pa ≈ e / length(cell)
    end
end
