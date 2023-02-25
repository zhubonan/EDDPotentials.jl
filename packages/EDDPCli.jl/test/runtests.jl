using Test
using EDDPCli
using EDDP

@testset "EDDPCli" begin
    opts = EDDPCli._get_builder_opt_template("test", "Si", "C")
    @test "Si" in opts.cf.elements
    @test "C" in opts.cf.elements
    @test opts.state.seedfile == "test"

    builder = Builder(opts)
    @test "Si" in builder.options.cf.elements
    @test builder.state.seedfile == "test"
end # EDDPCli.jl tests
