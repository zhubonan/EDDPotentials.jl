using Test
using EDDPotentialCli
using EDDPotential

@testset "EDDPotentialCli" begin
    opts = EDDPotentialCli._get_builder_opt_template("test", "Si", "C")
    @test "Si" in opts.cf.elements
    @test "C" in opts.cf.elements
    @test opts.state.seedfile == "test"

    builder = Builder(opts)
    @test "Si" in builder.options.cf.elements
    @test builder.state.seedfile == "test"
end # EDDPotentialCli.jl tests
