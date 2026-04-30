using EDDPotentials
using EDDPotentials:
    RandomEmbeddedCellFeature,
    FeatureStore,
    StructureContainer,
    MetaCalculator,
    CellRelaxationModifier,
    is_variable_cell,
    NNCalc,
    get_cell,
    nfeatures,
    feature_vector,
    eddp_energy,
    eddp_enthalpy,
    eddp_forces,
    eddp_stress,
    train!,
    split,
    ginit,
    fd_desc,
    stress_gradient
using EDDPotentials.EDDPotentialsCore:
    ManualFluxBackPropInterface,
    LinearInterface,
    flux_mlp_model,
    _need_calc
using Test
using Flux
using CellBase: positions, species

@testset "Random Embedded Feature" begin
    @testset "Constructor and Channel Scaling" begin
        cell = _h2o_cell()
        embedded = RandomEmbeddedCellFeature([:H, :O]; nembed=3, seed=7, p2=2:4, p3=2:3, q3=2:3)
        direct = feature_vector(embedded, cell)
        @test size(direct) == (nfeatures(embedded), length(cell))
        @test nfeatures(embedded) == 3 + 3 * 9 + 4 * 18
        @test length(embedded.pair_channels) == 9
        @test length(embedded.triplet_channels) == 18

        embedded2 = RandomEmbeddedCellFeature([:H, :O]; nembed=3, seed=7, p2=2:4, p3=2:3, q3=2:3)
        embedded3 = RandomEmbeddedCellFeature([:H, :O]; nembed=3, seed=8, p2=2:4, p3=2:3, q3=2:3)
        @test embedded.embedding == embedded2.embedding
        @test embedded.embedding != embedded3.embedding
    end

    @testset "Native Embedded Size Independence" begin
        cell = _lco_cell()
        embedded = RandomEmbeddedCellFeature([:Li, :Co, :O]; nembed=2, seed=3, p2=[2], p3=[2], q3=[2])
        direct = feature_vector(embedded, cell)
        @test size(direct) == (nfeatures(embedded), length(cell))

        many_elements = Symbol.("E" .* string.(1:89))
        embedded_many = RandomEmbeddedCellFeature(
            many_elements;
            nembed=2,
            seed=3,
            p2=[2],
            p3=[2],
            q3=[2],
        )
        @test EDDPotentials.feature_size(embedded) == EDDPotentials.feature_size(embedded_many)
    end

    @testset "Large Element Set Construction" begin
        elems = [:H, :He, :Li, :Be, :B, :C, :N, :O, :F, :Ne]
        pos = hcat([[0.7 * (i - 1), 0.2 * (-1)^i, 0.1 * (i % 3)] for i = 1:length(elems)]...)
        cell = Cell(Lattice(20.0, 20.0, 20.0), elems, pos)
        embedded = RandomEmbeddedCellFeature(elems; nembed=4, seed=13, p2=[2], p3=[2], q3=[2])

        direct = feature_vector(embedded, cell)
        @test size(direct) == (nfeatures(embedded), length(cell))
        @test EDDPotentials.feature_size(embedded) == (4, 16, 40)

        many_elements = Symbol.("E" .* string.(1:89))
        embedded_many = RandomEmbeddedCellFeature(
            many_elements;
            nembed=4,
            seed=13,
            p2=[2],
            p3=[2],
            q3=[2],
        )
        @test EDDPotentials.feature_size(embedded_many) == (4, 16, 40)
    end

    @testset "Descriptor Gradient Finite Difference" begin
        cell = _h2o_cell()
        embedded = RandomEmbeddedCellFeature([:H, :O]; nembed=3, seed=17, p2=2:2, p3=2:2, q3=2:2)

        diff, gvec_ji = fd_desc(embedded, cell)
        @test maximum(abs.(diff - gvec_ji)) < 1e-7

        diff_strain, virial = stress_gradient(embedded, cell)
        @test maximum(abs.(diff_strain + virial)) < 1e-4
    end

    @testset "Multi-Species Descriptor Gradient Finite Difference" begin
        elems = [:H, :He, :Li, :Be, :B, :C, :N, :O, :F, :Ne]
        pos = hcat([[0.7 * (i - 1), 0.2 * (-1)^i, 0.1 * (i % 3)] for i = 1:length(elems)]...)
        cell = Cell(Lattice(20.0, 20.0, 20.0), elems, pos)
        embedded = RandomEmbeddedCellFeature(elems; nembed=4, seed=19, p2=2:2, p3=2:2, q3=2:2)

        diff, gvec_ji = fd_desc(embedded, cell)
        @test maximum(abs.(diff - gvec_ji)) < 1e-7

        diff_strain, virial = stress_gradient(embedded, cell)
        @test maximum(abs.(diff_strain + virial)) < 1e-4
    end

    @testset "Calculator Integration" begin
        cell = _h2_cell()
        embedded = RandomEmbeddedCellFeature(
            unique(species(cell));
            nembed=2,
            seed=5,
            p2=[2],
            q3=[2, 3],
            p3=[2, 3],
        )

        function _test_forces_fd(calc, amp=1e-7, atol=1e-5)
            ftmp = copy(eddp_forces(calc))
            etmp = eddp_energy(calc)
            positions(get_cell(calc))[1] += amp
            @test _need_calc(calc, true)
            tmp = (eddp_energy(calc) - etmp) / amp
            @test -tmp ≈ ftmp[1] atol = atol
        end

        function _test_forces_fd_vc(calc; amp=1e-7, atol=1e-5, idx=1)
            if !is_variable_cell(calc)
                calc = MetaCalculator(calc, [CellRelaxationModifier(get_cell(calc))])
            end
            ftmp = copy(eddp_forces(calc))
            etmp = eddp_enthalpy(calc)
            pos = copy(get_positions(calc))
            pos[idx] += amp
            set_positions!(calc, pos)
            @test calc.energy_dirty
            tmp = (eddp_enthalpy(calc) - etmp) / amp
            @test -tmp ≈ ftmp[idx] atol = atol
        end

        lin = LinearInterface(rand(nfeatures(embedded)))
        calc = NNCalc(cell, embedded, lin; core=nothing)
        @test isa(eddp_energy(calc), Float64)
        @test all(isapprox.(sum(eddp_forces(calc), dims=2), 0.0, atol=1e-10))
        @test size(eddp_stress(calc)) == (3, 3)
        _test_forces_fd(calc)
        _test_forces_fd_vc(calc)

        mbp = ManualFluxBackPropInterface(embedded, 5)
        calc2 = NNCalc(cell, embedded, mbp; core=nothing)
        @test isa(eddp_energy(calc2), Float64)
        @test all(isapprox.(sum(eddp_forces(calc2), dims=2), 0.0, atol=1e-10))
        _test_forces_fd(calc2)

        model = flux_mlp_model(embedded, 5)
        @test size(model(rand(nfeatures(embedded), 2))) == (1, 2)
    end

    @testset "Multi-Species Finite Difference" begin
        cell = _lco_cell()
        embedded = RandomEmbeddedCellFeature(
            unique(species(cell));
            nembed=2,
            seed=11,
            p2=[2],
            q3=[2, 3],
            p3=[2, 3],
        )

        function _test_forces_fd(calc, amp=1e-7, atol=1e-5)
            ftmp = copy(eddp_forces(calc))
            etmp = eddp_energy(calc)
            positions(get_cell(calc))[1] += amp
            @test _need_calc(calc, true)
            tmp = (eddp_energy(calc) - etmp) / amp
            @test -tmp ≈ ftmp[1] atol = atol
        end

        function _test_forces_fd_vc(calc; amp=1e-7, atol=1e-5, idx=1)
            if !is_variable_cell(calc)
                calc = MetaCalculator(calc, [CellRelaxationModifier(get_cell(calc))])
            end
            ftmp = copy(eddp_forces(calc))
            etmp = eddp_enthalpy(calc)
            pos = copy(get_positions(calc))
            pos[idx] += amp
            set_positions!(calc, pos)
            @test calc.energy_dirty
            tmp = (eddp_enthalpy(calc) - etmp) / amp
            @test -tmp ≈ ftmp[idx] atol = atol
        end

        lin = LinearInterface(rand(nfeatures(embedded)))
        calc = NNCalc(cell, embedded, lin; core=nothing)
        _test_forces_fd(calc)
        _test_forces_fd_vc(calc)

        mbp = ManualFluxBackPropInterface(embedded, 5)
        calc2 = NNCalc(_lco_cell(), embedded, mbp; core=nothing)
        _test_forces_fd(calc2)
    end

    @testset "FeatureStore and Serialization" begin
        cells = [_h2o_cell(4.0, s) for s in (0.95, 1.0, 1.05, 1.1)]
        energies = [0.1, 0.2, 0.25, 0.35]
        sc = StructureContainer(cells, energies)
        embedded = RandomEmbeddedCellFeature([:H, :O]; nembed=2, seed=9, p2=2:3, p3=2:2, q3=2:2)

        fc = FeatureStore(sc, embedded; show_progress=false)
        @test length(get_data(fc)) == length(sc)
        @test size(fc[1], 1) == nfeatures(embedded)

        fc_train, fc_val = split(fc, 3, 1; standardize=true)
        model = ManualFluxBackPropInterface(
            Chain(Dense(ginit(4, nfeatures(embedded))), Dense(ginit(1, 4))),
            xt=fc_train.xt,
            yt=fc_train.yt,
        )
        train!(model, fc_train, fc_val; max_epochs=2, show_progress=false, earlystop=0)
        @test isfinite(sum(model(fc_train[1])))

        mktempdir() do dir
            path = joinpath(dir, "embedded_fc.jld2")
            EDDPotentials.save_feature_store(fc, path)
            fc_loaded = EDDPotentials.load_feature_store(path)
            @test fc_loaded.feature == embedded
            @test allclose(get_data(fc_loaded)[1], fc[1], atol=1e-12)
        end
    end

    @testset "Learnable Embedding Gradient" begin
        using EDDPotentials.EDDPotentialsCore:
            grad_embedding!,
            recompute_features!,
            get_workspace,
            suggest_rcut,
            paramvector,
            nparams,
            forward!,
            backward!,
            gradinp!,
            setparamvector!,
            gradparam!

        cell = _h2o_cell(5.0)
        cf = RandomEmbeddedCellFeature([:H, :O]; nembed=3, seed=42, p2=2:3, p3=2:2, q3=2:2)

        # Create a simple NN model
        model = ManualFluxBackPropInterface(cf, 4)
        nn_params = paramvector(model)

        # Compute features and NN output
        rcut = suggest_rcut(cf; shell=0.0)
        nl = CellBase.NeighbourList(cell, rcut, 500; savevec=true)
        wk = get_workspace(cf, nl, false)
        recompute_features!(wk.fvec, cf, cell, wk, nl)
        out = forward!(model, wk.fvec)
        energy = sum(out)

        # Backward pass to get dE/dfvec
        backward!(model)
        nat = natoms(cell)
        nf = nfeatures(cf)
        gv = zeros(Float64, nf, nat)
        gradinp!(gv, model)

        # Analytical embedding gradient
        gE = zeros(Float64, size(cf.embedding))
        grad_embedding!(gE, cf, cell, nl, gv)

        # Finite-difference verification
        eps = 1e-7
        gE_fd = zeros(Float64, size(cf.embedding))
        for s in 1:size(cf.embedding, 1)
            for d in 1:size(cf.embedding, 2)
                # Perturb +eps
                cf.embedding[s, d] += eps
                recompute_features!(wk.fvec, cf, cell, wk, nl)
                out_plus = forward!(model, wk.fvec)
                e_plus = sum(out_plus)

                # Perturb -eps
                cf.embedding[s, d] -= 2eps
                recompute_features!(wk.fvec, cf, cell, wk, nl)
                out_minus = forward!(model, wk.fvec)
                e_minus = sum(out_minus)

                # Restore
                cf.embedding[s, d] += eps

                gE_fd[s, d] = (e_plus - e_minus) / (2eps)
            end
        end

        # Restore features with original embedding
        recompute_features!(wk.fvec, cf, cell, wk, nl)

        @test allclose(gE, gE_fd, atol=1e-5)
    end
end
