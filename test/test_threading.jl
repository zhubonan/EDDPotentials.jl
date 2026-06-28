using EDDPotentials
using EDDPotentials.Potential: ManualFluxBackPropInterface, nparams, setup_fj, setup_fj_batched, setup_fj_batched_threaded, paramvector, ThreadBufferPool, compute_objectives, compute_objectives_diff, compute_objectives_threaded, rmse_per_atom_threaded

using Test
using Flux
using StatsBase

@testset "Threading" begin

    # Helper: generate synthetic data with mixed batch sizes
    function _make_data(; nf=100, n3=15, n2=10)
        data = vcat([rand(nf, 3) for _ = 1:n3], [rand(nf, 2) for _ = 1:n2])
        y = rand(length(data))
        data, y
    end

    @testset "ThreadBufferPool" begin
        nf = 100
        model = ManualFluxBackPropInterface(
            Chain(Dense(rand(1, nf), rand(1))),
        )
        nslots = 4
        pool = ThreadBufferPool(model, nslots)

        @test length(pool.jtmps) == nslots
        @test length(pool.gchains) == nslots
        @test pool.chain === EDDPotentials.Potential.get_flux_model(model)
        @test length(pool.jtmps[1]) == nparams(model)
    end

    @testset "ThreadBufferPool with yt transform" begin
        nf = 100
        data, y = _make_data(; nf)
        yt = StatsBase.fit(StatsBase.ZScoreTransform, reshape(y, 1, length(y)), dims=2)
        xt_len = 10
        xt = ZScoreTransform(xt_len, 2, randn(xt_len), rand(xt_len) .+ 0.5)

        model = ManualFluxBackPropInterface(
            Chain(Dense(rand(1, nf), rand(1))),
            xt=xt,
            yt=yt,
        )
        pool = ThreadBufferPool(model, 4)

        @test pool.yt_scale ≈ yt.scale[1]
        @test pool.yt === yt
        @test pool.xt === xt
    end

    @testset "ThreadBufferPool grows for chunk granularity" begin
        nf = 100
        data, y = _make_data(; nf)
        model = ManualFluxBackPropInterface(
            Chain(Dense(rand(1, nf), rand(1))),
        )

        _, f_full_thr!, pool = setup_fj_batched_threaded(model, data, y; ngps=2)
        @test pool.ngps == 2

        f_thr = zeros(length(data))
        f_full_thr!(f_thr, paramvector(model))
        @test any(f_thr .!= 0.0)

        small_pool = ThreadBufferPool(model, 1; ngps=2)
        compute_objectives_threaded(f_thr, small_pool, data, y)
        if Threads.nthreads() > 1 && length(data) > Threads.nthreads() * 5
            @test length(small_pool.gchains) >=
                length(EDDPotentials.Potential._chunk_ranges(length(data); ngps=small_pool.ngps))
        else
            @test length(small_pool.gchains) == 1
        end
    end

    @testset "_chunk_ranges" begin
        # Even division
        chunks = EDDPotentials.Potential._chunk_ranges(20; ngps=2)
        @test sum(length, chunks) == 20
        @test all(c -> !isempty(c), chunks)

        # Uneven division — remainder chunk
        chunks = EDDPotentials.Potential._chunk_ranges(23; ngps=2)
        @test sum(length, chunks) == 23

        # Too few data points — one element per chunk
        chunks = EDDPotentials.Potential._chunk_ranges(3; ngps=100)
        @test length(chunks) == 3
        @test sum(length, chunks) == 3
    end

    @testset "setup_fj_batched_threaded matches sequential" begin
        for with_transforms in (false, true)
            desc = with_transforms ? "with xt/yt" : "no transforms"
            @testset "$desc" begin
                nf = 100
                data, y = _make_data(; nf)

                xt_val = nothing
                yt_val = nothing
                if with_transforms
                    yt_val = StatsBase.fit(
                        StatsBase.ZScoreTransform,
                        reshape(y, 1, length(y)), dims=2,
                    )
                    xt_len = 10
                    xt_val = ZScoreTransform(xt_len, 2, randn(xt_len), rand(xt_len) .+ 0.5)
                end

                model_seq = ManualFluxBackPropInterface(
                    Chain(Dense(rand(1, nf), rand(1))),
                    xt=xt_val,
                    yt=yt_val,
                )
                model_thr = ManualFluxBackPropInterface(
                    Chain(Dense(rand(1, nf), rand(1))),
                    xt=xt_val,
                    yt=yt_val,
                )
                # Give both models identical weights
                p0 = paramvector(model_seq)
                EDDPotentials.Potential.setparamvector!(model_thr, p0)

                fj_batch_seq!, f_full_seq! = setup_fj_batched(model_seq, data, y)
                fj_batch_thr!, f_full_thr!, _ = setup_fj_batched_threaded(model_thr, data, y; ngps=2)

                # Test a subset batch
                batch_idx = collect(1:10)
                B = length(batch_idx)
                f_seq = zeros(B)
                j_seq = zeros(B, nparams(model_seq))
                fj_batch_seq!(f_seq, j_seq, p0, batch_idx)

                f_thr = zeros(B)
                j_thr = zeros(B, nparams(model_thr))
                fj_batch_thr!(f_thr, j_thr, p0, batch_idx)

                @test maximum(abs.(f_seq .- f_thr)) < 1e-10
                @test maximum(abs.(j_seq .- j_thr)) < 1e-10

                # Test f_full! closure
                f_full_seq = zeros(length(data))
                f_full_thr = zeros(length(data))
                f_full_seq!(f_full_seq, p0)
                f_full_thr!(f_full_thr, p0)
                @test maximum(abs.(f_full_seq .- f_full_thr)) < 1e-10

                # Test fvec-only path (jmat=nothing)
                f_seq2 = zeros(B)
                f_thr2 = zeros(B)
                fj_batch_seq!(f_seq2, nothing, p0, batch_idx)
                fj_batch_thr!(f_thr2, nothing, p0, batch_idx)
                @test maximum(abs.(f_seq2 .- f_thr2)) < 1e-10

                # Test jmat-only path (fvec=nothing)
                j_seq2 = zeros(B, nparams(model_seq))
                j_thr2 = zeros(B, nparams(model_thr))
                fj_batch_seq!(nothing, j_seq2, p0, batch_idx)
                fj_batch_thr!(nothing, j_thr2, p0, batch_idx)
                @test maximum(abs.(j_seq2 .- j_thr2)) < 1e-10
            end
        end
    end

    @testset "setup_fj_batched_threaded with weights" begin
        nf = 100
        data, y = _make_data(; nf)
        weights = rand(length(data)) .+ 0.5

        model_seq = ManualFluxBackPropInterface(
            Chain(Dense(rand(1, nf), rand(1))),
        )
        model_thr = ManualFluxBackPropInterface(
            Chain(Dense(rand(1, nf), rand(1))),
        )
        p0 = paramvector(model_seq)
        EDDPotentials.Potential.setparamvector!(model_thr, p0)

        fj_seq!, _ = setup_fj_batched(model_seq, data, y, weights)
        fj_thr!, _, _ = setup_fj_batched_threaded(model_thr, data, y, weights; ngps=2)

        batch_idx = collect(1:10)
        B = length(batch_idx)

        f_seq = zeros(B); j_seq = zeros(B, nparams(model_seq))
        f_thr = zeros(B); j_thr = zeros(B, nparams(model_thr))
        fj_seq!(f_seq, j_seq, p0, batch_idx)
        fj_thr!(f_thr, j_thr, p0, batch_idx)

        @test maximum(abs.(f_seq .- f_thr)) < 1e-10
        @test maximum(abs.(j_seq .- j_thr)) < 1e-10
    end

    @testset "compute_objectives_diff threaded matches sequential" begin
        nf = 100
        data, y = _make_data(; nf)

        model = ManualFluxBackPropInterface(
            Chain(Dense(rand(1, nf), rand(1))),
        )
        p0 = paramvector(model)

        # Sequential
        f_seq = zeros(length(data))
        j_seq = zeros(length(data), nparams(model))
        compute_objectives_diff(f_seq, j_seq, model, data, y; jtmp=j_seq[1, :])

        # Re-set parameters and compute via the internal threaded function
        EDDPotentials.Potential.setparamvector!(model, p0)
        f_thr = zeros(length(data))
        j_thr = zeros(length(data), nparams(model))
        EDDPotentials.Potential._compute_objectives_diff_threaded(
            f_thr, j_thr, model, data, y; jtmp=j_thr[1, :], ngps=2,
        )

        @test maximum(abs.(f_seq .- f_thr)) < 1e-10
        @test maximum(abs.(j_seq .- j_thr)) < 1e-10
    end

    @testset "train_lm_batched! threaded=true converges" begin
        nf = 100
        data = vcat([rand(nf, 3) for _ = 1:10], [rand(nf, 2) for _ = 1:10])
        y = size.(data, 2) ./ 2  # simple target: n_atoms / 2

        model = ManualFluxBackPropInterface(
            Chain(Dense(rand(1, nf), rand(1))),
        )
        train_lm_batched!(
            model, data, y;
            threaded=true,
            batch_size=length(data),
            max_epochs=20,
            earlystop=0,
        )
        out = model(data[2])
        @test sum(out) ≈ y[2] atol = 1e-4
    end

    @testset "train_lm_batched! threaded vs non-threaded produce same fit" begin
        nf = 100
        data = vcat([rand(nf, 3) for _ = 1:10], [rand(nf, 2) for _ = 1:10])
        y = rand(length(data))

        model_seq = ManualFluxBackPropInterface(
            Chain(Dense(rand(1, nf), rand(1))),
        )
        p0 = paramvector(model_seq)

        model_thr = ManualFluxBackPropInterface(
            Chain(Dense(rand(1, nf), rand(1))),
        )
        EDDPotentials.Potential.setparamvector!(model_thr, copy(p0))

        train_lm_batched!(
            model_seq, data, y;
            threaded=false,
            batch_size=length(data),
            max_epochs=10,
            earlystop=0,
        )
        train_lm_batched!(
            model_thr, data, y;
            threaded=true,
            batch_size=length(data),
            max_epochs=10,
            earlystop=0,
        )

        # Both should converge to similar parameters
        p_seq = paramvector(model_seq)
        p_thr = paramvector(model_thr)
        @test maximum(abs.(p_seq .- p_thr)) < 1e-8
    end

    @testset "threaded finite-difference Jacobian check" begin
        nf = 50
        data = [rand(nf, 3) for _ = 1:5]
        y = rand(length(data))

        model = ManualFluxBackPropInterface(
            Chain(Dense(rand(1, nf), rand(1))),
        )
        fj_thr!, _, _ = setup_fj_batched_threaded(model, data, y; ngps=1)

        p0 = paramvector(model)
        batch_idx = collect(1:length(data))
        B = length(batch_idx)

        fvec = zeros(B)
        jmat = zeros(B, nparams(model))
        fj_thr!(fvec, jmat, p0, batch_idx)

        # Finite difference check on first parameter
        dp = 1e-5
        fvec2 = zeros(B)
        p1 = copy(p0)
        p1[1] += dp
        fj_thr!(fvec2, nothing, p1, batch_idx)

        fd_jac = (fvec2 .- fvec) ./ dp
        @test abs(fd_jac[1] - jmat[1, 1]) < 1e-4
    end

    @testset "threaded with small dataset (sequential fallback)" begin
        # Dataset too small to trigger threading — should fall back to sequential
        nf = 50
        data = [rand(nf, 3) for _ = 1:3]
        y = rand(3)

        model = ManualFluxBackPropInterface(
            Chain(Dense(rand(1, nf), rand(1))),
        )
        fj_thr!, _, _ = setup_fj_batched_threaded(model, data, y)

        p0 = paramvector(model)
        fvec = zeros(3)
        jmat = zeros(3, nparams(model))
        # Should not error even when falling back to sequential
        fj_thr!(fvec, jmat, p0, collect(1:3))
        @test any(fvec .!= 0.0)
        @test any(jmat .!= 0.0)
    end

    @testset "compute_objectives_threaded matches sequential" begin
        for with_transforms in (false, true)
            desc = with_transforms ? "with xt/yt" : "no transforms"
            @testset "$desc" begin
                nf = 100
                data, y = _make_data(; nf)

                xt_val = nothing
                yt_val = nothing
                if with_transforms
                    yt_val = StatsBase.fit(
                        StatsBase.ZScoreTransform,
                        reshape(y, 1, length(y)), dims=2,
                    )
                    xt_len = 10
                    xt_val = ZScoreTransform(xt_len, 2, randn(xt_len), rand(xt_len) .+ 0.5)
                end

                model = ManualFluxBackPropInterface(
                    Chain(Dense(rand(1, nf), rand(1))),
                    xt=xt_val,
                    yt=yt_val,
                )

                # Sequential
                f_seq = zeros(length(data))
                compute_objectives(f_seq, model, data, y)

                # Threaded via pool
                pool = ThreadBufferPool(model, 4)
                f_thr = zeros(length(data))
                compute_objectives_threaded(f_thr, pool, data, y)

                @test maximum(abs.(f_seq .- f_thr)) < 1e-10
            end
        end
    end

    @testset "compute_objectives_threaded with weights" begin
        nf = 100
        data, y = _make_data(; nf)
        weights = rand(length(data)) .+ 0.5

        model = ManualFluxBackPropInterface(
            Chain(Dense(rand(1, nf), rand(1))),
        )

        f_seq = zeros(length(data))
        compute_objectives(f_seq, model, data, y, weights)

        pool = ThreadBufferPool(model, 4)
        f_thr = zeros(length(data))
        compute_objectives_threaded(f_thr, pool, data, y, weights)

        @test maximum(abs.(f_seq .- f_thr)) < 1e-10
    end

    @testset "rmse_per_atom_threaded matches sequential" begin
        for with_transforms in (false, true)
            desc = with_transforms ? "with xt/yt" : "no transforms"
            @testset "$desc" begin
                nf = 100
                data, y = _make_data(; nf)
                natoms = [size(v, 2) for v in data]

                xt_val = nothing
                yt_val = nothing
                if with_transforms
                    yt_val = StatsBase.fit(
                        StatsBase.ZScoreTransform,
                        reshape(y, 1, length(y)), dims=2,
                    )
                    xt_len = 10
                    xt_val = ZScoreTransform(xt_len, 2, randn(xt_len), rand(xt_len) .+ 0.5)
                end

                model = ManualFluxBackPropInterface(
                    Chain(Dense(rand(1, nf), rand(1))),
                    xt=xt_val,
                    yt=yt_val,
                )

                # Sequential
                rmse_seq = EDDPotentials.rmse_per_atom(model, data, y, natoms)

                # Threaded
                pool = ThreadBufferPool(model, 4)
                rmse_thr = rmse_per_atom_threaded(pool, data, y, natoms)

                @test abs(rmse_seq - rmse_thr) < 1e-10
            end
        end
    end

    @testset "f_full! threaded closure with pool" begin
        nf = 100
        data, y = _make_data(; nf)

        model = ManualFluxBackPropInterface(
            Chain(Dense(rand(1, nf), rand(1))),
        )
        p0 = paramvector(model)

        _, f_full_thr!, pool = setup_fj_batched_threaded(model, data, y; ngps=2)

        # f_full! should now use threaded compute_objectives internally
        f_thr = zeros(length(data))
        f_full_thr!(f_thr, p0)

        # Compare with sequential compute_objectives
        f_seq = zeros(length(data))
        compute_objectives(f_seq, model, data, y)

        @test maximum(abs.(f_seq .- f_thr)) < 1e-10
    end
end
