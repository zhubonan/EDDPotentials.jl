using EDDPotentials
using Test
using TOML
using Configurations

check_equal(a, b) = a == b
check_equal(a::AbstractArray, b::AbstractArray) = all(a .== b)

@testset "Builder" begin

    @testset "Serialization" begin


        "Test for round-trip toml conversion"
        function test_to_from_toml(obj::T) where {T}
            obj_reconstructed = mktempdir() do temp
                open(joinpath(temp, "test.toml"), "w") do f
                    to_toml(f, obj)
                end
                from_toml(T, joinpath(temp, "test.toml"))
            end
            for name in fieldnames(T)
                @test check_equal(
                    getproperty(obj, name),
                    getproperty(obj_reconstructed, name),
                )
            end
        end

        "Test for round-trip toml conversion"
        function test_to_from_toml_builder(obj)

            options = mktempdir() do temp
                EDDPotentials.save_builder(joinpath(temp, "test.toml"), builder)
                o2 = from_toml(EDDPotentials.BuilderOption, joinpath(temp, "test.toml"))
                open(joinpath(temp, "showout"), "w") do fh
                    Base.show(fh, o2)
                end
                o2
            end
            obj_reconstructed = Builder(options)
            for name in fieldnames(Builder)
                @test check_equal(
                    getproperty(obj, name),
                    getproperty(obj_reconstructed, name),
                )
            end
        end



        state = EDDPotentials.BuilderState(seedfile="seed.cell", seedfile_calc="calc.cell")
        test_to_from_toml(state)

        lm = EDDPotentials.TrainingOption()
        test_to_from_toml(lm)

        rss = EDDPotentials.RssSetting()
        test_to_from_toml(rss)

        cf = EDDPotentials.CellFeatureConfig(elements=["H"])
        test_to_from_toml(cf)


        opt = EDDPotentials.BuilderOption(state, cf, nothing, rss, lm, nothing)
        test_to_from_toml(opt)

        # Construct builder from opt
        builder = Builder(opt)
        test_to_from_toml_builder(builder)
    end
end
