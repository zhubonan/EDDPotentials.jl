using EDDPotentials
using EDDPotentials: ComputedRecord, get_e_above_hull, PhaseDiagram
using EDDPotentials.Analysis:
    find_simplex,
    get_coord,
    bary_coords,
    coords_from_bary,
    contains_point
using CellBase
using Test

isdefined(Main, :allclose) || include("utils.jl")

@testset "Records" begin
    records = ComputedRecord[
        ComputedRecord(:O, 0.0),
        ComputedRecord(:H2O, -1.0),
        ComputedRecord(:H, 0.0),
    ]

    phased = PhaseDiagram(records)

    # Compute the distance to the surface defined by the simplex along the last dimension

    #%%
    simplex = phased.simplices[find_simplex(phased, records[1])]
    coord = get_coord(records[1], phased.elements)
    bcord = bary_coords(simplex, coord)
    @test allclose(coords_from_bary(simplex, bcord), coord)
    @test allclose(bcord, [0, 1]; atol=1e-14)

    bcord = bary_coords(simplex, records[2], phased)
    @test allclose(bcord, [1, 0]; atol=1e-14)

    @test contains_point(simplex, coord)

    @test get_e_above_hull(phased, ComputedRecord(Composition(:H2O), 1)) ≈ 2 / 3 atol =
        1e-14
    @test get_e_above_hull(phased, ComputedRecord(Composition(:H), 1)) ≈ 1.0 atol = 1e-14
    @test get_e_above_hull(phased, ComputedRecord(Composition(:O), 0)) ≈ 0.0 atol = 1e-14
end
