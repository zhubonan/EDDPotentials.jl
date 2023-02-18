using EDDP
using Test
using YAML

check_equal(a, b) = a == b
check_equal(a::AbstractArray, b::AbstractArray) = all(a .== b)

@testset "Builder" begin

    @testset "Serialization" begin

        "Test for round-trip dict conversion"
        function test_to_from_dict(obj::T) where {T}
            state_dict = EDDP._todict(obj)
            obj_reconstructed = EDDP._fromdict(T, state_dict)
            for name in fieldnames(T)
                @test check_equal(
                    getproperty(obj, name),
                    getproperty(obj_reconstructed, name),
                )
            end
        end

        function compare_field_equality(a::T, b) where {T}
            all(
                check_equal(getproperty(a, name), getproperty(b, name)) for
                name in fieldnames(T)
            )
        end

        "Test for round-trip yaml conversion"
        function test_to_from_yaml(obj::T) where {T}
            state_dict_ = EDDP._todict(obj)
            state_dict = mktempdir() do temp
                YAML.write_file(joinpath(temp, "test.yaml"), state_dict_)
                YAML.load_file(joinpath(temp, "test.yaml"), dicttype=Dict{Symbol,Any})
            end
            obj_reconstructed = EDDP._fromdict(T, state_dict)
            for name in fieldnames(T)
                @test check_equal(
                    getproperty(obj, name),
                    getproperty(obj_reconstructed, name),
                )
            end
        end

        state = EDDP.BuilderState(seedfile="myseed.jl")
        test_to_from_dict(state)
        test_to_from_yaml(state)

        lm = EDDP.LocalLMTrainer()
        test_to_from_dict(lm)
        test_to_from_yaml(lm)

        rss = EDDP.RssSetting()
        test_to_from_dict(rss)
        test_to_from_yaml(rss)

        cf = EDDP.CellFeature(["H"])
        test_to_from_dict(cf)
        test_to_from_yaml(cf)

        # Compare round trip for the builder
        builder_dict = Dict{Symbol,Any}(
            :cf => Dict{Symbol,Any}(:elements => ["H"]),
            :state => Dict{Symbol,Any}(:seedfile => "seed.cell"),
            :trainer => Dict{Symbol,Any}(:type => "locallm"),
            :cf_embedding => Dict{Symbol,Any}(:n => 3),
        )
        builder = EDDP._fromdict(EDDP.Builder, builder_dict)
        builder_dict2 = EDDP._todict(builder)
        builder2 = EDDP._fromdict(EDDP.Builder, builder_dict2)
        function compare_builder(builder, builder2)
            @test compare_field_equality(builder2.cf, builder.cf)
            @test compare_field_equality(builder2.state, builder.state)
            @test compare_field_equality(builder2.rss, builder.rss)
            @test builder2.cf_embedding.m == builder.cf_embedding.m
            @test builder2.cf_embedding.n == builder.cf_embedding.n
            @test compare_field_equality(builder2.trainer, builder.trainer)
        end
        compare_builder(builder, builder2)

        # YAML test
        loaded_builder = mktempdir() do tempd
            fname = joinpath(tempd, "test.yaml")
            EDDP.save_builder(fname, builder)
            EDDP.Builder(fname)
        end
        # These fields should be set depends on the file path
        @test loaded_builder.state.workdir != builder.state.workdir
        loaded_builder.state.workdir = builder.state.workdir
        @test loaded_builder.state.builder_file_path != builder.state.builder_file_path
        loaded_builder.state.builder_file_path = builder.state.builder_file_path

        compare_builder(builder, loaded_builder)
        global builder, loaded_builder
    end

end
