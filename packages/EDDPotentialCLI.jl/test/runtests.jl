using Test
using EDDPotentialsCli
using EDDPotentials

@testset "EDDPotentialsCli" begin
    opts = EDDPotentialsCli._get_builder_opt_template("test", "Si", "C")
    @test "Si" in opts.cf.elements
    @test "C" in opts.cf.elements
    @test opts.state.seedfile == "test"

    builder = Builder(opts)
    @test "Si" in builder.options.cf.elements
    @test builder.state.seedfile == "test"
end # EDDPotentialsCli.jl tests
