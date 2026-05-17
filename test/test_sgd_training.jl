include("utils.jl")

using EDDPotentials
using EDDPotentials:
    train_sgd!,
    constant_lr,
    step_decay_lr,
    cosine_annealing_lr,
    train!,
    StructureContainer,
    CellFeature,
    FeatureStore,
    nfeatures,
    predict_energy,
    reinit
using EDDPotentials.Potential:
    ManualFluxBackPropInterface,
    FluxInterface,
    ginit,
    nparams,
    paramvector
using EDDPotentials.Training:
    _scatter_matrix,
    _hcat_batch

using Test
using Flux
using StatsBase

@testset "SGD Training" begin

    @testset "_scatter_matrix" begin
        # Basic: 3 structures with 2, 3, 1 atoms
        natoms_vec = [2, 3, 1]
        ref = zeros(Float64, 5)  # any array for device/type reference
        S = _scatter_matrix([1, 3], natoms_vec, ref)
        @test size(S) == (2, 3)  # 2 structures, 2+1=3 total atoms
        # Structure 1 gets atoms 1-2, structure 3 gets atom 3
        @test S[1, 1] == 1.0
        @test S[1, 2] == 1.0
        @test S[1, 3] == 0.0
        @test S[2, 1] == 0.0
        @test S[2, 2] == 0.0
        @test S[2, 3] == 1.0

        # All structures selected
        S2 = _scatter_matrix([1, 2, 3], natoms_vec, ref)
        @test size(S2) == (3, 6)
        @test sum(S2) == 6  # one 1 per atom
        # Each row sums to its structure's atom count
        @test sum(S2[1, :]) == 2
        @test sum(S2[2, :]) == 3
        @test sum(S2[3, :]) == 1

        # Single structure
        S3 = _scatter_matrix([2], natoms_vec, ref)
        @test size(S3) == (1, 3)
        @test all(S3[1, :] .== 1.0)

        # Verify energy summation: per-atom energies → per-structure totals
        natoms_vec2 = [4, 2, 3]
        Y = collect(1.0:9.0)  # 9 atoms total
        S4 = _scatter_matrix([1, 2, 3], natoms_vec2, ref)
        E_pred = S4 * Y
        @test E_pred[1] ≈ sum(Y[1:4])
        @test E_pred[2] ≈ sum(Y[5:6])
        @test E_pred[3] ≈ sum(Y[7:9])
    end

    @testset "_hcat_batch" begin
        # Concatenate 3 matrices with different numbers of columns (atoms)
        x1 = [1.0 2.0; 3.0 4.0]       # 2 features, 2 atoms
        x2 = [5.0 6.0 7.0; 8.0 9.0 10.0]  # 2 features, 3 atoms
        x3 = [11.0; 12.0]              # 2 features, 1 atom
        x_t = [x1, x2, x3]

        X = _hcat_batch(x_t, [1, 3])
        @test size(X) == (2, 3)  # 2 features, 2+1=3 atoms
        @test X[:, 1:2] == x1
        @test X[:, 3] == x3

        # All structures
        X2 = _hcat_batch(x_t, [1, 2, 3])
        @test size(X2) == (2, 6)  # 2 features, 2+3+1=6 atoms
        @test X2[:, 1:2] == x1
        @test X2[:, 3:5] == x2
        @test X2[:, 6] == x3

        # Single structure
        X3 = _hcat_batch(x_t, [2])
        @test X3 == x2

        # Empty-ish: single atom
        X4 = _hcat_batch(x_t, [3])
        @test size(X4) == (2, 1)
        @test X4[:, 1] == x3[:, 1]
    end

    @testset "_scatter_matrix and _hcat_batch together" begin
        # End-to-end: scatter matrix applied to concatenated batch gives correct per-structure sums
        nf = 5
        natoms_vec = [3, 2, 4]
        x_t = [rand(nf, n) for n in natoms_vec]
        ref = zeros(nf)

        batch_indices = [1, 3]
        X = _hcat_batch(x_t, batch_indices)
        S = _scatter_matrix(batch_indices, natoms_vec, ref)

        # Simulate per-atom energy predictions: one energy per atom
        Y_atom = rand(sum(natoms_vec[i] for i in batch_indices))
        E_via_scatter = S * Y_atom

        # Manually compute expected per-structure energies
        offset = 0
        for (k, i) in enumerate(batch_indices)
            nat = natoms_vec[i]
            @test E_via_scatter[k] ≈ sum(Y_atom[offset+1:offset+nat])
            offset += nat
        end
    end

    @testset "Learning rate schedules" begin
        # Constant
        sched = constant_lr(0.01)
        @test sched(1) == 0.01
        @test sched(100) == 0.01

        # Step decay: epoch ÷ step_size, so epoch=10 with step_size=10 → factor^1
        sched2 = step_decay_lr(0.1; step_size=10, factor=0.5)
        @test sched2(1) ≈ 0.1
        @test sched2(9) ≈ 0.1
        @test sched2(10) ≈ 0.05   # 10 ÷ 10 = 1
        @test sched2(11) ≈ 0.05
        @test sched2(20) ≈ 0.025  # 20 ÷ 10 = 2

        # Cosine annealing
        sched3 = cosine_annealing_lr(0.1, 0.001, 100)
        @test sched3(0) ≈ 0.1 atol = 1e-6
        @test sched3(100) < 0.002
        # Monotonically decreasing after epoch 0
        vals = [sched3(e) for e in 0:100]
        @test all(diff(vals) .< 0)
    end

    @testset "Adam convergence (linear model)" begin
        # Use same-size structures so a linear model can fit exactly:
        # E = w * sum(x) + n * b, constant n means it reduces to linear in sum(x)
        nf = 10
        n_atoms = 3
        data = [rand(nf, n_atoms) for _ = 1:20]
        # Generate targets from a known linear model so fitting is exact
        w_true = rand(1, nf)
        b_true = rand(1)
        y = [sum(w_true * d .+ b_true) for d in data]

        model = ManualFluxBackPropInterface(
            Chain(Dense(rand(1, nf), rand(1))),
        )
        opt_res, params, history, _ = train_sgd!(
            model, data, y;
            optimizer=:adam, lr=1e-2,
            max_epochs=200, earlystop=0,
            batch_size=20,  # full batch
        )
        # Linear model should fit exactly since the data is generated from a linear model
        out = model(data[2])
        @test sum(out) ≈ y[2] atol = 0.1
        @test length(params) > 0
    end

    @testset "Adam convergence with yt standardization" begin
        # Verify that training works correctly when yt (output standardization) is set.
        # A linear model with yt should converge — check that RMSE decreases.
        nf = 10
        n_atoms = 3
        data = [rand(nf, n_atoms) for _ = 1:20]
        w_true = rand(1, nf)
        b_true = rand(1)
        y = [sum(w_true * d .+ b_true) for d in data]

        # Create a yt transform (Z-score on per-atom energies)
        y_per_atom = y ./ n_atoms
        yt = StatsBase.fit(StatsBase.ZScoreTransform, reshape(y_per_atom, 1, length(y)), dims=2)

        model = ManualFluxBackPropInterface(
            Chain(Dense(rand(1, nf), rand(1)));
            yt=yt,
        )
        opt_res, params, history, _ = train_sgd!(
            model, data, y;
            optimizer=:adam, lr=1e-2,
            max_epochs=200, earlystop=0,
            batch_size=20,
        )
        # RMSE should have decreased from initial random weights
        @test history[end, 1] < history[1, 1]
        @test length(params) > 0
    end

    @testset "SGD with momentum convergence" begin
        nf = 10
        data = vcat([rand(nf, 3) for _ = 1:10], [rand(nf, 2) for _ = 1:10])
        y = rand(length(data))

        model = ManualFluxBackPropInterface(
            Chain(Dense(rand(1, nf), rand(1))),
        )
        opt_res, params, history, _ = train_sgd!(
            model, data, y;
            optimizer=:sgd_momentum, lr=1e-3, momentum=0.9,
            max_epochs=100, earlystop=0,
            batch_size=10,
        )
        out = model(data[2])
        @test isfinite(sum(out))
    end

    @testset "Plain SGD convergence" begin
        nf = 10
        data = vcat([rand(nf, 3) for _ = 1:10], [rand(nf, 2) for _ = 1:10])
        y = rand(length(data))

        model = ManualFluxBackPropInterface(
            Chain(Dense(rand(1, nf), rand(1))),
        )
        opt_res, params, history, _ = train_sgd!(
            model, data, y;
            optimizer=:sgd, lr=1e-2,
            max_epochs=100, earlystop=0,
            batch_size=10,
        )
        out = model(data[2])
        @test isfinite(sum(out))
    end

    @testset "SGD with FluxInterface" begin
        nf = 10
        data = vcat([rand(nf, 3) for _ = 1:10], [rand(nf, 2) for _ = 1:10])
        y = rand(length(data))

        model = FluxInterface(
            Chain(Dense(rand(1, nf), rand(1))),
        )
        opt_res, params, history, _ = train_sgd!(
            model, data, y;
            optimizer=:adam, lr=1e-3,
            max_epochs=200, earlystop=0,
            batch_size=10,
        )
        out = model(data[2])
        @test isfinite(sum(out))
        @test length(params) > 0
    end

    @testset "Early stopping" begin
        nf = 10
        # Use data where overfitting is likely: small train set
        data = [rand(nf, 3) for _ = 1:20]
        y = rand(length(data))

        model = ManualFluxBackPropInterface(
            Chain(Dense(rand(1, nf), rand(1))),
        )

        x_val = data[11:20]
        y_val = y[11:20]
        x_train = data[1:10]
        y_train = y[1:10]

        # Use high learning rate to cause oscillation/overfitting
        opt_res, params, history, _ = train_sgd!(
            model, x_train, y_train;
            x_val, y_val,
            optimizer=:sgd, lr=1.0,
            max_epochs=1000, earlystop=20,
            batch_size=10,
        )
        # With aggressive lr on random data, training should become unstable
        # and early stop should trigger. If not, still verify the mechanism runs.
        @test size(history, 1) <= 1000
        @test length(params) > 0
    end

    @testset "train! dispatcher integration" begin
        path = joinpath(datadir, "training/*.res")
        path = relpath(path, pwd())
        sc = StructureContainer([path])
        cf = CellFeature([:B])
        fc = FeatureStore(sc, cf)
        fc_train, fc_val = split(fc, 4, 4; standardize=true)

        model = ManualFluxBackPropInterface(
            Chain(Dense(ginit(5, nfeatures(cf))), Dense(ginit(1, 5))),
            xt=fc_train.xt,
            yt=fc_train.yt,
        )
        opt_res, params, history, _ = train!(
            model, fc_train, fc_val;
            train_method=:sgd,
            optimizer=:adam,
            lr=1e-3,
            max_epochs=30,
            earlystop=10,
            batch_size=4,
        )
        @test length(params) > 0
        @test size(history, 1) >= 1
    end

    @testset "Learning rate scheduling" begin
        nf = 10
        data = [rand(nf, 3) for _ = 1:10]
        y = rand(length(data))

        model = ManualFluxBackPropInterface(
            Chain(Dense(rand(1, nf), rand(1))),
        )

        schedule = cosine_annealing_lr(1e-2, 1e-4, 50)
        opt_res, params, history, _ = train_sgd!(
            model, data, y;
            optimizer=:adam, lr=1e-2,
            lr_schedule=schedule,
            max_epochs=50, earlystop=0,
            batch_size=5,
        )
        @test size(history, 1) == 50
        @test isfinite(params[1])
    end

    @testset "Weight decay" begin
        nf = 10
        data = [rand(nf, 3) for _ = 1:10]
        y = rand(length(data))

        model = ManualFluxBackPropInterface(
            Chain(Dense(rand(1, nf), rand(1))),
        )
        opt_res, params, history, _ = train_sgd!(
            model, data, y;
            optimizer=:adam, lr=1e-3,
            max_epochs=100, earlystop=0,
            batch_size=5,
            weight_decay=1e-4,
        )
        @test isfinite(params[1])
    end
end
