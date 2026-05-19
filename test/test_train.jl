using EDDPotentials
using EDDPotentials.Potential: ManualFluxBackPropInterface, LinearInterface, AbstractNNInterface, FluxInterface, ginit, compute_objectives, compute_objectives_diff, nparams, setup_fj, setup_fj_batched, paramvector
using EDDPotentials.Training:
    levenberg_marquardt,
    levenberg_marquardt_batched,
    _apply_damping_full!,
    _solve_normal_full!

using Test
using Flux
using StatsBase
using NLSolversBase
using Random

isdefined(Main, :datadir) || include("utils.jl")

function _train_rattled_family_sc()
    base = _h2o_cell()
    labels = [
        "train-parent-a",
        "train-parent-a-shake-1",
        "train-parent-a-force-shake-1",
        "train-parent-b",
        "train-parent-b-shake-1",
        "train-parent-b-shake-2",
        "train-parent-c",
        "train-parent-c-shake-1",
        "train-parent-c-force-shake-1",
    ]
    cells = typeof(base)[deepcopy(base) for _ in labels]
    for (i, cell) in enumerate(cells)
        cell.metadata[:label] = labels[i]
        set_positions!(cell, get_positions(cell) .+ 0.01 * i)
    end
    StructureContainer(cells, -12.5 .+ 0.01 .* collect(eachindex(cells)), labels)
end

function _train_parent_sets(tm)
    parent(label) = begin
        result = match(r"^(.+?)(?:-force)?-shake-\d+$", label)
        isnothing(result) ? label : result.captures[1]
    end
    map(
        fc -> Set(parent.(fc.labels)),
        (tm.train_results.fc, tm.val_results.fc, tm.test_results.fc),
    )
end

function _capture_training_stdout(f)
    mktemp() do _, io
        redirect_stdout(io) do
            f()
        end
        seekstart(io)
        read(io, String)
    end
end

function _assert_training_startup_header(text, routine, log_file; extras=String[])
    @test occursin("==== Training Startup ====", text)
    @test occursin("routine = $routine", text)
    @test occursin("model_type = ", text)
    @test occursin("target_parameters = ", text)
    @test occursin("train_samples = ", text)
    @test occursin("validation_samples = ", text)
    @test occursin("julia_threads = ", text)
    @test occursin("options:", text)
    for extra in extras
        @test occursin(extra, text)
    end
    log_text = read(log_file, String)
    @test occursin("routine = $routine", log_text)
    for extra in extras
        @test occursin(extra, log_text)
    end
end

@testset "Training" begin
    @testset "Full LM damping modes" begin
        JJ = [2.0 0.1; 0.1 3.0]
        DtD = [0.25, 4.0]

        JJ_fletcher = copy(JJ)
        _apply_damping_full!(JJ_fletcher, 10.0, DtD, :fletcher)
        @test JJ_fletcher[1, 1] == 4.5
        @test JJ_fletcher[2, 2] == 43.0
        @test JJ_fletcher[1, 2] == JJ[1, 2]

        JJ_floor = copy(JJ)
        _apply_damping_full!(JJ_floor, 10.0, DtD, :fletcher_floor)
        @test JJ_floor[1, 1] == 12.0
        @test JJ_floor[2, 2] == 43.0
        @test JJ_floor[1, 2] == JJ[1, 2]

        @test_throws ArgumentError _apply_damping_full!(copy(JJ), 10.0, DtD, :unknown)
    end

    @testset "Full LM strict Cholesky" begin
        A = [0.0 1.0; 1.0 0.0]
        rhs = [1.0, 2.0]

        delta_chol = fill(-99.0, 2)
        @test !_solve_normal_full!(delta_chol, copy(A), rhs, :cholesky)
        @test delta_chol == fill(-99.0, 2)

        delta_direct = zeros(2)
        @test _solve_normal_full!(delta_direct, copy(A), rhs, :direct)
        @test delta_direct ≈ [2.0, 1.0]

        delta_qr = zeros(2)
        @test _solve_normal_full!(delta_qr, copy(A), rhs, :qr)
        @test delta_qr ≈ [2.0, 1.0]

        function f_rankdef!(fvec, p)
            fvec[1] = p[1] - 1.0
            fvec
        end
        function j_rankdef!(jmat, p)
            jmat[1, 1] = 1.0
            jmat[1, 2] = 0.0
            jmat
        end
        function fj_rankdef!(fvec, jmat, p)
            f_rankdef!(fvec, p)
            j_rankdef!(jmat, p)
            fvec, jmat
        end
        p0 = [0.0, 5.0]
        od = OnceDifferentiable(
            f_rankdef!,
            j_rankdef!,
            fj_rankdef!,
            p0,
            zeros(1),
            inplace=true,
        )
        opt_res = levenberg_marquardt(
            od,
            p0;
            lambda=0.0,
            maxIter=3,
            earlystop=0,
            keep_best=false,
            damping_mode=:fletcher_floor,
            solver=:cholesky,
        )
        @test opt_res.minimizer[1] ≈ 1.0 atol = 1e-8
    end

    @testset "Training startup headers" begin
        Random.seed!(1234)
        nf = 4
        data = [rand(nf, 2), rand(nf, 3), rand(nf, 2), rand(nf, 3)]
        y = rand(length(data))

        mktempdir() do dir
            lm_log = joinpath(dir, "lm.log")
            model = ManualFluxBackPropInterface(Chain(Dense(rand(1, nf), rand(1))))
            lm_out = _capture_training_stdout() do
                train_lm!(
                    model,
                    data,
                    y;
                    max_epochs=1,
                    earlystop=0,
                    show_progress=false,
                    log_file=lm_log,
                    g_tol=1e-9,
                )
            end
            _assert_training_startup_header(
                lm_out,
                "train_lm!",
                lm_log;
                extras=[
                    "max_epochs = 1",
                    "show_progress = false",
                    "g_tol = 1.0e-9",
                    "solver = :cholesky",
                ],
            )

            lm_floor_log = joinpath(dir, "lm-floor.log")
            model = ManualFluxBackPropInterface(Chain(Dense(rand(1, nf), rand(1))))
            lm_floor_out = _capture_training_stdout() do
                train_lm!(
                    model,
                    data,
                    y;
                    max_epochs=1,
                    earlystop=0,
                    show_progress=false,
                    log_file=lm_floor_log,
                    damping_mode=:fletcher_floor,
                )
            end
            _assert_training_startup_header(
                lm_floor_out,
                "train_lm!",
                lm_floor_log;
                extras=[
                    "damping_mode = :fletcher_floor",
                    "lambda = 100000.0",
                ],
            )

            lm_floor_override_log = joinpath(dir, "lm-floor-override.log")
            model = ManualFluxBackPropInterface(Chain(Dense(rand(1, nf), rand(1))))
            lm_floor_override_out = _capture_training_stdout() do
                train_lm!(
                    model,
                    data,
                    y;
                    max_epochs=1,
                    earlystop=0,
                    show_progress=false,
                    log_file=lm_floor_override_log,
                    damping_mode=:fletcher_floor,
                    lambda=123.0,
                )
            end
            _assert_training_startup_header(
                lm_floor_override_out,
                "train_lm!",
                lm_floor_override_log;
                extras=[
                    "damping_mode = :fletcher_floor",
                    "lambda = 123.0",
                ],
            )

            batched_log = joinpath(dir, "batched.log")
            model = ManualFluxBackPropInterface(Chain(Dense(rand(1, nf), rand(1))))
            batched_out = _capture_training_stdout() do
                train_lm_batched!(
                    model,
                    data,
                    y;
                    batch_size=length(data),
                    max_epochs=1,
                    earlystop=0,
                    show_progress=false,
                    threaded=false,
                    log_file=batched_log,
                    x_tol=1e-8,
                )
            end
            _assert_training_startup_header(
                batched_out,
                "train_lm_batched!",
                batched_log;
                extras=[
                    "batch_size = 4",
                    "threaded = false",
                    "x_tol = 1.0e-8",
                ],
            )

            sgd_log = joinpath(dir, "sgd.log")
            model = ManualFluxBackPropInterface(Chain(Dense(rand(1, nf), rand(1))))
            sgd_out = _capture_training_stdout() do
                train_sgd!(
                    model,
                    data,
                    y;
                    batch_size=2,
                    max_epochs=1,
                    earlystop=0,
                    show_progress=false,
                    log_file=sgd_log,
                    startup_marker=:sgd_header,
                )
            end
            _assert_training_startup_header(
                sgd_out,
                "train_sgd!",
                sgd_log;
                extras=[
                    "batch_size = 2",
                    "optimizer = :adam",
                    "startup_marker = :sgd_header",
                ],
            )

            structures = [_h2o_cell(), deepcopy(_h2o_cell())]
            y_emb = rand(length(structures))
            cf = RandomEmbeddedCellFeature([:H, :O]; nembed=2, seed=11, p2=[2], p3=[2], q3=[2])
            model = ManualFluxBackPropInterface(Chain(Dense(rand(1, nfeatures(cf)), rand(1))))

            emb_log = joinpath(dir, "embedding.log")
            emb_out = _capture_training_stdout() do
                EDDPotentials.Training.train_lm_with_embedding!(
                    model,
                    structures,
                    cf,
                    y_emb;
                    max_epochs=1,
                    earlystop=0,
                    show_progress=false,
                    log_file=emb_log,
                    normalize_embedding=true,
                    damping_mode=:fletcher_floor,
                )
            end
            _assert_training_startup_header(
                emb_out,
                "train_lm_with_embedding!",
                emb_log;
                extras=[
                    "normalize_embedding = true",
                    "max_epochs = 1",
                    "damping_mode = :fletcher_floor",
                    "lambda = 100000.0",
                ],
            )

            cf_override = RandomEmbeddedCellFeature([:H, :O]; nembed=2, seed=13, p2=[2], p3=[2], q3=[2])
            model = ManualFluxBackPropInterface(Chain(Dense(rand(1, nfeatures(cf_override)), rand(1))))
            emb_override_log = joinpath(dir, "embedding-override.log")
            emb_override_out = _capture_training_stdout() do
                EDDPotentials.Training.train_lm_with_embedding!(
                    model,
                    structures,
                    cf_override,
                    y_emb;
                    max_epochs=1,
                    earlystop=0,
                    show_progress=false,
                    log_file=emb_override_log,
                    damping_mode=:fletcher_floor,
                    lambda=123.0,
                )
            end
            _assert_training_startup_header(
                emb_override_out,
                "train_lm_with_embedding!",
                emb_override_log;
                extras=[
                    "damping_mode = :fletcher_floor",
                    "lambda = 123.0",
                ],
            )

            cf_batched = RandomEmbeddedCellFeature([:H, :O]; nembed=2, seed=12, p2=[2], p3=[2], q3=[2])
            model = ManualFluxBackPropInterface(Chain(Dense(rand(1, nfeatures(cf_batched)), rand(1))))
            emb_batched_log = joinpath(dir, "embedding-batched.log")
            emb_batched_out = _capture_training_stdout() do
                EDDPotentials.Training.train_lm_batched_with_embedding!(
                    model,
                    structures,
                    cf_batched,
                    y_emb;
                    batch_size=length(structures),
                    max_epochs=1,
                    earlystop=0,
                    show_progress=false,
                    log_file=emb_batched_log,
                    solver=:direct,
                )
            end
            _assert_training_startup_header(
                emb_batched_out,
                "train_lm_batched_with_embedding!",
                emb_batched_log;
                extras=[
                    "batch_size = 2",
                    "solver = :direct",
                ],
            )
        end
    end

    @testset "NN objectives" begin
        Random.seed!(1234)
        nf = 1000
        data = vcat([rand(nf, 3) for _ = 1:10], [rand(nf, 2) for _ = 1:10])

        f = zeros(length(data))
        y = rand(length(data))
        yt = StatsBase.fit(StatsBase.ZScoreTransform, reshape(y, 1, length(y)), dims=2)

        model = ManualFluxBackPropInterface(
            Chain(Dense(rand(1, nf), rand(1), tanh)),
            yt=yt,
        )
        compute_objectives(f, model, data, f)
        @test any(f != 0.0)

        jmat = rand(length(data), nparams(model))
        compute_objectives_diff(f, jmat, model, data, y)

        @test any(jmat != 0.0)

        # Setting up and minimising
        f!, j!, fj! = setup_fj(model, data, y)
        p0 = paramvector(model)

        # Manual diff
        jmat = zeros(length(y), length(p0))
        j!(jmat, p0)
        dp = 1e-4
        out = similar(y)
        f!(out, p0)
        p0[1] += dp
        out2 = similar(y)
        f!(out2, p0)
        val = (out2[1] - out[1]) / dp
        @test abs(val - jmat[1]) < 1e-7

        # Optimisation 
        od2 = OnceDifferentiable(
            f!,
            j!,
            fj!,
            p0,
            zeros(eltype(data[1]), length(data)),
            inplace=true,
        )
        opt_res = levenberg_marquardt(od2, p0; show_trace=false, damping_mode=:fletcher_floor, solver=:cholesky)

        j!(jmat, opt_res.minimizer)
        @test all(isapprox.(jmat, 0.0, atol=1e-6))


        # Test with training config
        model =
            ManualFluxBackPropInterface(Chain(Dense(rand(1, nf), rand(1))))

        train_lm!(model, data, y; damping_mode=:fletcher_floor, solver=:direct)
        out = model(data[2])
        # Check we have successfully performed the fit
        @test sum(out) ≈ y[2] atol = 1e-5

        # Highly overfitted linear case....
        nf = 1000
        data = vcat([rand(nf, 3) for _ = 1:10], [rand(nf, 2) for _ = 1:10])
        y = size.(data, 2) ./ 2
        model = LinearInterface(rand(1, nf))

        opt_res, _, _ = train_lm!(model, data, y; earlystop=0)
    end

    @testset "Ensemble" begin
        path = joinpath(datadir, "training/*.res")
        path = relpath(path, pwd())
        sc = StructureContainer([path])
        cf = CellFeature([:B])
        fc = FeatureStore(sc, cf)

        fc_train, fc_val = split(fc, 4, 4; standardize=true)
        models = []
        model = ManualFluxBackPropInterface(
            Chain(Dense(ginit(5, nfeatures(fc.feature))), Dense(ginit(1, 5))),
            xt=fc_train.xt,
            yt=fc_train.yt,
        )
        model_ = reinit(model)
        @test any(paramvector(model_) .!= paramvector(model))
        for _ = 1:5
            model_ = reinit(model)
            res = train!(model_, fc_train, fc_val)
            push!(models, model_)
        end

        emod = create_ensemble(models, fc_train)
        out = predict_energy(emod, fc_train[1])
        @test isa(out, Real)

    end


    @testset "Flux" begin
        nf = 1000
        data = vcat([rand(nf, 3) for _ = 1:10], [rand(nf, 2) for _ = 1:10])

        f = zeros(length(data))
        y = rand(length(data))
        yt = StatsBase.fit(StatsBase.ZScoreTransform, reshape(y, 1, length(y)), dims=2)

        model = FluxInterface(Chain(Dense(rand(1, nf), rand(1))))
        compute_objectives(f, model, data, f)
        @test any(f != 0.0)

        jmat = rand(length(data), nparams(model))
        compute_objectives_diff(f, jmat, model, data, y)
        @test any(jmat != 0.0)

        # Setting up and minimising
        f!, j!, fj! = setup_fj(model, data, y)
        p0 = paramvector(model)

        # Manual finite diff
        jmat = zeros(length(y), length(p0))
        j!(jmat, p0)
        dp = 1e-4
        out = similar(y)
        f!(out, p0)
        p0[1] += dp
        out2 = similar(y)
        f!(out2, p0)
        val = (out2[1] - out[1]) / dp
        @test abs(val - jmat[1]) < 1e-7

        # Optimisation 
        od2 = OnceDifferentiable(
            f!,
            j!,
            fj!,
            p0,
            zeros(eltype(data[1]), length(data)),
            inplace=true,
        )
        opt_res = levenberg_marquardt(od2, p0; show_trace=false)
        #@test_broken opt_res.g_converged
        @test opt_res.x_converged
    end

    @testset "Batched LM" begin
        nf = 1000
        data = vcat([rand(nf, 3) for _ = 1:10], [rand(nf, 2) for _ = 1:10])
        y = rand(length(data))

        # Test setup_fj_batched with a linear model (no saturation issues)
        model = ManualFluxBackPropInterface(
            Chain(Dense(rand(1, nf), rand(1))),
        )
        fj_batch!, f_full! = setup_fj_batched(model, data, y)

        batch_idx = [1, 3, 5, 7]
        B = length(batch_idx)
        f_b = zeros(B)
        j_b = zeros(B, nparams(model))
        fj_batch!(f_b, j_b, paramvector(model), batch_idx)
        @test any(f_b .!= 0.0)
        @test any(j_b .!= 0.0)

        # Compare batch results against full-dataset computation
        f_full = zeros(length(data))
        compute_objectives(f_full, model, data, y)
        for (li, gi) in enumerate(batch_idx)
            @test abs(f_b[li] - f_full[gi]) < 1e-10
        end
    end

    @testset "train! dispatch lm_batched" begin
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
        # batch_size must be >= nparams for well-conditioned normal equations
        # Use batch_size = n_data to effectively test the dispatch path
        opt_res, params, history, funcs = train!(
            model, fc_train, fc_val;
            train_method=:lm_batched,
            batch_size=length(fc_train),
            max_epochs=30,
            earlystop=10,
        )
        @test length(params) > 0
        # history is only populated when callback is active (earlystop > 0)
        @test size(history, 1) >= 1
    end

    @testset "train_model" begin
        path = joinpath(datadir, "training/*.res")
        path = relpath(path, pwd())
        cf = CellFeature([:B])

        # Load full dataset once, create a small subset of 6 structures for fast tests
        sc_full = StructureContainer([path])
        sc_small = StructureContainer(sc_full.structures[1:6], sc_full.H[1:6])
        mktempdir() do small_dir
            for (i, s) in enumerate(sc_small.structures)
                CellBase.write_res(joinpath(small_dir, "t$i.res"), s)
            end
            small_path = joinpath(small_dir, "t*.res")

            # Train once with minimal epochs and reuse across subtests that only need a model object
            base_tm = train_model(small_path, cf;
                val_ratio=0.2, test_ratio=0.2, nnodes=[3], max_epochs=2, earlystop=5, verbose=false)

            @testset "from paths with explicit spec" begin
                tm = train_model(small_path, cf;
                    val_ratio=0.2, test_ratio=0.2, nnodes=[3], max_epochs=2, earlystop=5, verbose=false)
                @test tm isa TrainedModel
                @test tm.model isa ManualFluxBackPropInterface
                @test tm.feature_spec == cf
                @test tm.val_results isa TrainingResults
                @test tm.test_results isa TrainingResults
                @test length(tm.test_results.fc) > 0
            end

            @testset "from StructureContainer" begin
                tm = train_model(sc_small, cf; val_ratio=0.2, test_ratio=0.2, nnodes=[3], max_epochs=2, earlystop=5, verbose=false)
                @test tm isa TrainedModel
                @test tm.val_results isa TrainingResults
                @test tm.test_results isa TrainingResults
            end

            @testset "from FeatureStore" begin
                fc = FeatureStore(sc_small, cf)
                tm = train_model(fc; val_ratio=0.2, test_ratio=0.2, nnodes=[3], max_epochs=2, earlystop=5, verbose=false)
                @test tm isa TrainedModel
                @test tm.val_results isa TrainingResults
                @test tm.test_results isa TrainingResults
            end

            @testset "predict on TrainedModel" begin
                cell = sc_small.structures[1]
                result = EDDPotentials.predict(base_tm, cell)
                @test haskey(result, :energy)
                @test haskey(result, :forces)
                @test haskey(result, :stress)
                @test isfinite(result.energy)
                @test all(isfinite, result.forces)
                @test all(isfinite, result.stress)
                result_min = EDDPotentials.predict(base_tm, cell; forces=false, stress=false)
                @test !haskey(result_min, :forces)
                @test !haskey(result_min, :stress)
            end

            @testset "predict on EDDPCalculator" begin
                calc = EDDPCalculator(base_tm)
                cell = sc_small.structures[1]
                result = EDDPotentials.predict(calc, cell)
                @test haskey(result, :energy)
                @test haskey(result, :forces)
                @test isfinite(result.energy)
            end

            @testset "save/load round-trip" begin
                mktempdir() do dir
                    p = joinpath(dir, "tm.json")
                    save_model_artifact(p, base_tm)
                    artifact = load_model_artifact(p)
                    calc = EDDPCalculator(artifact)
                    cell = sc_small.structures[1]
                    energy_before = EDDPotentials.predict(base_tm, cell; forces=false, stress=false).energy
                    energy_after = EDDPotentials.predict(calc, cell; forces=false, stress=false).energy
                    @test energy_before ≈ energy_after
                end
            end

            @testset "val_ratio=0 throws" begin
                @test_throws ArgumentError train_model(small_path, cf; val_ratio=0.0)
            end

            @testset "test_ratio=0 throws" begin
                @test_throws ArgumentError train_model(small_path, cf; test_ratio=0.0)
            end

            @testset "tiny ratio yields zero structures throws" begin
                @test_throws ArgumentError train_model(small_path, cf; val_ratio=0.01)
                @test_throws ArgumentError train_model(small_path, cf; test_ratio=0.01)
            end

            @testset "pressure_gpa passthrough" begin
                tm = train_model(small_path, cf; val_ratio=0.2, test_ratio=0.2, nnodes=[3], max_epochs=2,
                                 earlystop=5, pressure_gpa=2.0, verbose=false)
                @test tm isa TrainedModel
            end

            @testset "show TrainedModel" begin
                io = IOBuffer()
                show(io, MIME("text/plain"), base_tm)
                s = String(take!(io))
                @test occursin("TrainedModel", s)
                @test occursin("RMSE", s)
                @test occursin("Validation", s)
                @test occursin("Test", s)
            end

            @testset "save_model_artifact from TrainedModel" begin
                mktempdir() do dir
                    p = joinpath(dir, "model.json")
                    saved = save_model_artifact(p, base_tm)
                    @test saved == p
                    @test isfile(p)
                    loaded = load_model(p)
                    @test loaded isa AbstractNNInterface
                end
            end

            @testset "auto-infer feature spec from paths" begin
                tm = train_model(small_path; val_ratio=0.2, test_ratio=0.2, nnodes=[3], max_epochs=2,
                                 earlystop=5, verbose=false)
                @test tm isa TrainedModel
                @test :B in tm.feature_spec.elements
                @test tm.val_results isa TrainingResults
                @test tm.test_results isa TrainingResults
            end

            @testset "auto-infer feature spec from StructureContainer" begin
                tm = train_model(sc_small; val_ratio=0.2, test_ratio=0.2, nnodes=[3], max_epochs=2,
                                 earlystop=5, verbose=false)
                @test tm isa TrainedModel
                @test :B in tm.feature_spec.elements
            end

            @testset "auto-infer with custom feature params" begin
                tm = train_model(small_path; val_ratio=0.2, test_ratio=0.2, nnodes=[3], max_epochs=2,
                                 earlystop=5, p2=2:6, rcut2=5.0, verbose=false)
                @test tm isa TrainedModel
                @test collect(tm.feature_spec.elements) == [:B]
                @test !isempty(tm.feature_spec.p2)
                @test tm.feature_spec.rcut2 ≈ 5.0
            end

            @testset "seed reproducibility" begin
                tm1 = train_model(small_path, cf; val_ratio=0.2, test_ratio=0.2, nnodes=[3],
                                   max_epochs=2, earlystop=5, seed=123, verbose=false)
                tm2 = train_model(small_path, cf; val_ratio=0.2, test_ratio=0.2, nnodes=[3],
                                   max_epochs=2, earlystop=5, seed=123, verbose=false)
                @test tm1.train_results.fc.labels == tm2.train_results.fc.labels
                @test tm1.val_results.fc.labels == tm2.val_results.fc.labels
                @test tm1.test_results.fc.labels == tm2.test_results.fc.labels
            end

            @testset "respect_shakes for train_model StructureContainer" begin
                rattled = _train_rattled_family_sc()
                cf_rattle = CellFeature(unique(species(rattled.structures[1]));
                    p2=[2, 3], p3=[2], q3=[1], rcut2=4.0, rcut3=3.5)
                tm = train_model(
                    rattled,
                    cf_rattle;
                    val_ratio=0.34,
                    test_ratio=0.34,
                    respect_shakes=true,
                    nnodes=[3],
                    max_epochs=1,
                    earlystop=2,
                    verbose=false,
                    show_progress=false,
                )
                train_parents, val_parents, test_parents = _train_parent_sets(tm)
                @test isempty(intersect(train_parents, val_parents))
                @test isempty(intersect(train_parents, test_parents))
                @test isempty(intersect(val_parents, test_parents))
            end

            @testset "respect_shakes for train_model FeatureStore" begin
                rattled = _train_rattled_family_sc()
                cf_rattle = CellFeature(unique(species(rattled.structures[1]));
                    p2=[2, 3], p3=[2], q3=[1], rcut2=4.0, rcut3=3.5)
                fs = FeatureStore(rattled, cf_rattle; show_progress=false)
                tm = train_model(
                    fs;
                    val_ratio=0.34,
                    test_ratio=0.34,
                    respect_shakes=true,
                    nnodes=[3],
                    max_epochs=1,
                    earlystop=2,
                    verbose=false,
                    show_progress=false,
                )
                train_parents, val_parents, test_parents = _train_parent_sets(tm)
                @test isempty(intersect(train_parents, val_parents))
                @test isempty(intersect(train_parents, test_parents))
                @test isempty(intersect(val_parents, test_parents))
            end

            @testset "pre-split StructureContainers with feature_spec" begin
                sc_train, sc_val, sc_test = split(sc_small, 4, 1, 1; standardize=false)
                tm = train_model(sc_train, sc_val, sc_test, cf;
                    nnodes=[3], max_epochs=2, earlystop=5, verbose=false)
                @test tm isa TrainedModel
                @test length(tm.train_results.fc) == 4
                @test length(tm.val_results.fc) == 1
                @test length(tm.test_results.fc) == 1
            end

            @testset "pre-split StructureContainers auto-infer" begin
                sc_train, sc_val, sc_test = split(sc_small, 4, 1, 1; standardize=false)
                tm = train_model(sc_train, sc_val, sc_test;
                    nnodes=[3], max_epochs=2, earlystop=5, verbose=false)
                @test tm isa TrainedModel
                @test :B in tm.feature_spec.elements
            end

            @testset "pre-split paths with feature_spec" begin
                sc_train, sc_val, sc_test = split(sc_small, 4, 1, 1; standardize=false)
                mktempdir() do dir
                    for (i, s) in enumerate(sc_train.structures)
                        CellBase.write_res(joinpath(dir, "train_$i.res"), s)
                    end
                    for (i, s) in enumerate(sc_val.structures)
                        CellBase.write_res(joinpath(dir, "val_$i.res"), s)
                    end
                    for (i, s) in enumerate(sc_test.structures)
                        CellBase.write_res(joinpath(dir, "test_$i.res"), s)
                    end
                    tm = train_model(
                        joinpath(dir, "train_*.res"),
                        joinpath(dir, "val_*.res"),
                        joinpath(dir, "test_*.res"),
                        cf;
                        nnodes=[3], max_epochs=2, earlystop=5, verbose=false)
                    @test tm isa TrainedModel
                end
            end

            @testset "pre-split paths auto-infer" begin
                sc_train, sc_val, sc_test = split(sc_small, 4, 1, 1; standardize=false)
                mktempdir() do dir
                    for (i, s) in enumerate(sc_train.structures)
                        CellBase.write_res(joinpath(dir, "train_$i.res"), s)
                    end
                    for (i, s) in enumerate(sc_val.structures)
                        CellBase.write_res(joinpath(dir, "val_$i.res"), s)
                    end
                    for (i, s) in enumerate(sc_test.structures)
                        CellBase.write_res(joinpath(dir, "test_$i.res"), s)
                    end
                    tm = train_model(
                        joinpath(dir, "train_*.res"),
                        joinpath(dir, "val_*.res"),
                        joinpath(dir, "test_*.res");
                        nnodes=[3], max_epochs=2, earlystop=5, verbose=false)
                    @test tm isa TrainedModel
                    @test :B in tm.feature_spec.elements
                end
            end

            @testset "pre-split FeatureStores" begin
                fc = FeatureStore(sc_small, cf)
                fc_train, fc_val, fc_test = split(fc, 4, 1, 1; standardize=true)
                tm = train_model(fc_train, fc_val, fc_test;
                    nnodes=[3], max_epochs=2, earlystop=5, verbose=false)
                @test tm isa TrainedModel
                @test length(tm.train_results.fc) == 4
                @test length(tm.val_results.fc) == 1
                @test length(tm.test_results.fc) == 1
            end

            @testset "elemental_energies passthrough" begin
                e0s = Dict(:B => -5.0)
                tm = train_model(small_path, cf;
                    val_ratio=0.2, test_ratio=0.2, nnodes=[3], max_epochs=2,
                    earlystop=5, elemental_energies=e0s, verbose=false)
                @test tm.elemental_energies[:B] ≈ -5.0
            end
        end
    end
end
