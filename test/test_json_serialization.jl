using EDDPotentials
using EDDPotentials: EDDPCalculator, eddp_energy, feature_vector
using EDDPotentials.Potential:
    BodyEmbedding,
    CellEmbedding,
    EnsembleNNInterface,
    JSON,
    feature_size,
    LinearInterface,
    ManualFluxBackPropInterface,
    forward!,
    load_feature_artifact,
    load_model_artifact,
    save_as_jld2,
    save_feature_json,
    save_model_artifact
using Flux: Chain, Dense, tanh
using StatsBase: fit, ZScoreTransform
using Test

isdefined(Main, :_h2_cell) || include("utils.jl")

feature_dtype(cf::CellFeature) = !isempty(cf.p2) ? eltype(cf.p2) :
                                 !isempty(cf.p3) ? eltype(cf.p3) :
                                 Float64
feature_dtype(cf::RandomEmbeddedCellFeature) = eltype(cf.embedding)

function feature_equivalent(a::CellFeature, b::CellFeature)
    a == b || return false
    feature_dtype(a) == feature_dtype(b)
end

function feature_equivalent(a::RandomEmbeddedCellFeature, b::RandomEmbeddedCellFeature)
    a == b || return false
    feature_dtype(a) == feature_dtype(b)
end

function assert_model_roundtrip(path, model, feature_spec, cell; elemental_energies=Dict{Symbol,Float64}(), metadata=nothing)
    input_before = feature_vector(feature_spec, cell)
    output_before = forward!(model, input_before)

    save_model_artifact(path, model; feature_spec, elemental_energies, metadata)
    artifact = load_model_artifact(path)

    @test feature_equivalent(artifact.feature_spec, feature_spec)
    @test artifact.elemental_energies == elemental_energies

    input_after = feature_vector(artifact.feature_spec, cell)
    @test input_after == input_before
    @test forward!(artifact.model, input_after) == output_before

    artifact
end

@testset "JSON Serialization" begin
    @testset "Feature Artifact" begin
        cell = _h2o_cell()
        feature = RandomEmbeddedCellFeature(
            [:H, :O];
            T=Float32,
            nembed=2,
            seed=9,
            p2=2:3,
            p3=2:2,
            q3=2:2,
        )
        elemental_energies = Dict(:H => -1.5, :O => -8.0)
        expected = feature_vector(feature, cell)

        mktempdir() do dir
            path = joinpath(dir, "feature.json")
            save_feature_json(path, feature; elemental_energies)
            artifact = load_feature_artifact(path)

            @test feature_equivalent(artifact.feature_spec, feature)
            @test artifact.elemental_energies == elemental_energies
            @test eltype(artifact.feature_spec.embedding) == Float32
            @test feature_vector(artifact.feature_spec, cell) == expected
        end
    end

    @testset "Linear Model Artifact" begin
        cell = _h2_cell()
        cf = CellFeature([:H]; p2=[2.0, 3.0], p3=Float64[], q3=Float64[])
        model = LinearInterface(collect(range(0.1, length=nfeatures(cf), step=0.1)))
        elemental_energies = Dict(:H => -1.2)

        mktempdir() do dir
            path = joinpath(dir, "linear.json")
            artifact = assert_model_roundtrip(
                path,
                model,
                cf,
                cell;
                elemental_energies,
            )

            @test artifact.model isa LinearInterface
            @test eltype(artifact.model.param) == eltype(model.param)
            @test load_model(path) isa LinearInterface
        end
    end

    @testset "Manual Backprop Artifact" begin
        cell = _h2o_cell()
        cf = CellFeature([:H, :O]; p2=[2.0, 3.0], p3=[2.0], q3=[2.0])
        xt = fit(ZScoreTransform, rand(nfeatures(cf) - feature_size(cf)[1], 6), dims=2)
        yt = fit(ZScoreTransform, rand(1, 6), dims=2)
        chain = Chain(Dense(rand(4, nfeatures(cf)), rand(4), tanh), Dense(rand(1, 4), rand(1)))
        model = ManualFluxBackPropInterface(chain; xt, yt)
        elemental_energies = Dict(:H => -1.1, :O => -7.7)

        mktempdir() do dir
            path = joinpath(dir, "manual.json")
            artifact = assert_model_roundtrip(
                path,
                model,
                cf,
                cell;
                elemental_energies,
            )

            @test artifact.model isa ManualFluxBackPropInterface
            @test artifact.model.xt.mean == xt.mean
            @test artifact.model.xt.scale == xt.scale
            @test artifact.model.yt.mean == yt.mean
            @test artifact.model.yt.scale == yt.scale
            @test eltype(artifact.model.chain.layers[1].weight) == eltype(model.chain.layers[1].weight)
            @test eltype(artifact.model.chain.layers[2].weight) == eltype(model.chain.layers[2].weight)
        end
    end

    @testset "Embedded Model Artifact" begin
        cell = _h2o_cell()
        cf = CellFeature([:H, :O]; p2=[2.0, 3.0], p3=Float64[], q3=Float64[])
        embed = CellEmbedding(Float32, cf, 2, 2)
        embed.two_body.weight .= reshape(
            Float32.(collect(1:length(embed.two_body.weight))),
            size(embed.two_body.weight),
        )
        embed.three_body.weight .= reshape(
            Float32.(collect(1:length(embed.three_body.weight))),
            size(embed.three_body.weight),
        )
        ninp = size(embed(rand(Float32, nfeatures(cf), 1)), 1)
        chain = Chain(
            embed,
            Dense(rand(Float32, 3, ninp), rand(Float32, 3), tanh),
            Dense(rand(Float32, 1, 3), rand(Float32, 1)),
        )
        model = ManualFluxBackPropInterface(chain)

        mktempdir() do dir
            path = joinpath(dir, "embedded.json")
            artifact = assert_model_roundtrip(
                path,
                model,
                cf,
                cell;
                elemental_energies=Dict(:H => -1.0, :O => -7.0),
            )

            @test artifact.model isa ManualFluxBackPropInterface
            @test artifact.model.chain.layers[1] isa CellEmbedding
            @test artifact.model.chain.layers[1].two_body.weight == embed.two_body.weight
            @test artifact.model.chain.layers[1].three_body.weight == embed.three_body.weight
            @test eltype(artifact.model.chain.layers[1].two_body.weight) == Float32
            @test eltype(artifact.model.chain.layers[1].three_body.weight) == Float32
            @test eltype(artifact.model.chain.layers[2].weight) == eltype(model.chain.layers[2].weight)
            @test eltype(artifact.model.chain.layers[3].weight) == eltype(model.chain.layers[3].weight)
        end
    end

    @testset "Random Embedded Model Artifact" begin
        cell = _h2o_cell()
        cf = RandomEmbeddedCellFeature(
            [:H, :O];
            T=Float32,
            nembed=2,
            seed=11,
            p2=2:3,
            p3=2:2,
            q3=2:2,
        )
        model = LinearInterface(Float32.(collect(range(0.1, length=nfeatures(cf), step=0.05))))

        mktempdir() do dir
            path = joinpath(dir, "embedded_feature.json")
            artifact = assert_model_roundtrip(
                path,
                model,
                cf,
                cell;
                elemental_energies=Dict(:H => -1.0, :O => -8.0),
            )

            @test artifact.model isa LinearInterface
            @test eltype(artifact.feature_spec.embedding) == Float32
            @test eltype(artifact.model.param) == Float32
        end
    end

    @testset "Ensemble Artifact" begin
        cell = _h2_cell()
        cf = CellFeature([:H]; p2=[2.0, 3.0], p3=Float64[], q3=Float64[])
        model1 = LinearInterface(collect(range(0.1, length=nfeatures(cf), step=0.1)))
        model2 = LinearInterface(collect(range(0.2, length=nfeatures(cf), step=0.1)))
        ensemble = EnsembleNNInterface((model1, model2), [0.25, 0.75])
        elemental_energies = Dict(:H => -1.2)

        mktempdir() do dir
            path = joinpath(dir, "ensemble.json")
            artifact = assert_model_roundtrip(
                path,
                ensemble,
                cf,
                cell;
                elemental_energies,
                metadata=Dict("builder_uuid" => "abc123", "train_labels" => ["a", "b"]),
            )

            @test artifact.model isa EnsembleNNInterface
            @test artifact.metadata["builder_uuid"] == "abc123"
            @test artifact.metadata["train_labels"] == ["a", "b"]
            @test eltype(artifact.model.models[1].param) == eltype(model1.param)
            @test eltype(artifact.model.models[2].param) == eltype(model2.param)
        end
    end

    @testset "Legacy JLD2 Round-Trips" begin
        # Basic model
        model = LinearInterface([0.5, 0.6, 0.7])
        mktempdir() do dir
            path = joinpath(dir, "legacy.jld2")
            save_as_jld2(path, model)
            artifact = load_model_artifact(path)
            @test artifact.model isa LinearInterface
            @test artifact.feature_spec === nothing
            @test isempty(artifact.elemental_energies)
        end

        # Ensemble round-trip
        cf = CellFeature([:H]; p2=[2.0, 3.0], p3=Float64[], q3=Float64[])
        xt = fit(ZScoreTransform, rand(nfeatures(cf) - feature_size(cf)[1], 6), dims=2)
        yt = fit(ZScoreTransform, rand(1, 6), dims=2)
        model = ManualFluxBackPropInterface(
            Chain(Dense(rand(4, nfeatures(cf)), rand(4), tanh), Dense(rand(1, 4), rand(1)));
            xt,
            yt,
        )
        ensemble = EnsembleNNInterface((model,), [1.0])
        cell = _h2_cell()
        input = feature_vector(cf, cell)
        expected = forward!(ensemble, input)

        mktempdir() do dir
            path = joinpath(dir, "legacy_ensemble.jld2")
            save_as_jld2(path, ensemble)
            artifact = load_model_artifact(path)

            @test artifact.model isa EnsembleNNInterface
            @test artifact.model.models[1] isa ManualFluxBackPropInterface
            @test forward!(artifact.model, input) == expected
        end
    end

    @testset "Invalid Activation" begin
        cf = CellFeature([:H]; p2=[2.0], p3=Float64[], q3=Float64[])
        model = LinearInterface([0.1, 0.2])
        mktempdir() do dir
            path = joinpath(dir, "invalid.json")
            save_model_artifact(path, model; feature_spec=cf)
            payload = JSON.parse(read(path, String))
            payload["model"] = Dict(
                "interface_type" => "manual_flux_backprop",
                "feature_spec" => payload["model"]["feature_spec"],
                "elemental_energies" => payload["model"]["elemental_energies"],
                "metadata" => Dict{String,Any}(),
                "xt" => nothing,
                "yt" => nothing,
                "layers" => [Dict(
                    "layer_type" => "dense",
                    "activation" => "relu",
                    "weight" => [[1.0]],
                    "bias" => [0.0],
                )],
            )
            write(path, JSON.json(payload))
            @test_throws ArgumentError load_model_artifact(path)
        end
    end
end
