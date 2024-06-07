using EDDPotential
using EDDPotentialTools
using CellBase
using PyCall
using Flux

using Test

@testset "ASE interface" begin

    raw"""
    Generate LJ like system: 

    ```math
    E_{at} = f(r)^12 - 2f(r)^6
    ```

    """
    function gen_lj_like(cell)
        elem = unique(species(cell))
        @assert length(elem) == 1 "Only works for single specie Cell for now."
        cf = EDDPotential.CellFeature(elem, p2=[2, 6], p3=[], q3=[])
        chain = Chain(Dense(zeros(1, 3)))
        chain.layers[1].weight[1, 2:3] .= [-2.0, 1.0]
        chain.layers[1].bias .= 0

        model = EDDPotential.ManualFluxBackPropInterface(chain)
        EDDPotential.NNCalc(cell, cf, model)
    end

    ase = pyimport("ase")

    cell = Cell(Lattice(10.0, 10.0, 10.0), [:H, :H, :H, :H], rand(3, 4) .* 4.0)
    calc = gen_lj_like(cell)
    EDDPotential.get_energy(calc)
    atoms, acalc = EDDPotentialTools.get_ase_atoms_and_calculator(calc)


    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    @test get_forces(calc) == transpose(forces)
    @test get_energy(calc) == energy

    # Test geometry optimisation

    mod = pyimport("ase.optimize.bfgs")
    b = mod.BFGS(atoms, logfile=nothing)
    b.run(steps=100)
    @test maximum(atoms.get_forces()) < 1
end


include("test_hull_2d.jl")
include("test_hull_3d.jl")
