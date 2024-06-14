#=
Common functions for helping tests
=#
using EDDPotentials
using CellBase

function _h2_cell(l=4.0, factor=1.0)
    tmp = Float64[
        0 0.1
        0 2.0
        0.1 0
    ] .* factor
    Cell(Lattice(l, l, l), [:H, :H], tmp)
end

function _h2o_cell(l=4.0, factor=1.0)
    tmp = Float64[
        0 0.1 1
        0 1 1
        0.1 0 1
    ] .* factor
    Cell(Lattice(l, l, l), [:H, :H, :O], tmp)
end

function _lco_cell()
    scaled_pos = Float64[
        0.0000000000000000 0.0000000000000000 0.0000000000000000
        0.6666666666666666 0.3333333333333333 0.3333333333333333
        0.3333333333333333 0.6666666666666666 0.6666666666666666
        0.3333333333333333 0.6666666666666665 0.1666666666666667
        0.9999999999999999 0.9999999999999998 0.5000000000000000
        0.6666666666666665 0.3333333333333330 0.8333333333333333
        0.0000000000000000 0.0000000000000000 0.2400068000000000
        0.6666666666666666 0.3333333333333333 0.0933265333333333
        0.6666666666666666 0.3333333333333333 0.5733401333333333
        0.3333333333333333 0.6666666666666666 0.4266598666666666
        0.3333333333333333 0.6666666666666666 0.9066734666666667
        0.0000000000000000 0.0000000000000000 0.7599931999999999
    ]
    cell = Float64[
        1.4056284883992509 -2.4346199584737427 0.0000000000000000
        1.4056284883992509 2.4346199584737427 0.0000000000000000
        0.0000000000000000 0.0000000000000000 13.9094564325679286
    ]
    cell = collect(transpose(cell))
    pos = cell * transpose(scaled_pos)
    pos .+= rand(size(pos)...) * 0.1

    Cell(Lattice(cell), [:Li, :Li, :Li, :Co, :Co, :Co, :O, :O, :O, :O, :O, :O], pos)
end



function _generate_cf(cell::Cell)
    EDDPotentials.CellFeature(unique(species(cell)); p2=[6, 12], q3=[2, 3], p3=[2, 3])
end


function _get_calc()
    cell = _h2_cell()
    cf = _generate_cf(cell)
    itf = EDDPotentials.ManualFluxBackPropInterface(
        Chain(Dense(rand(5, EDDPotentials.nfeatures(cf))), Dense(rand(1, 5))),
    )
    calc = EDDPotentials.NNCalc(cell, cf, itf)
    calc
end

function _get_lco_calc()
    cell = _lco_cell()
    cf = _generate_cf(cell)
    itf = EDDPotentials.ManualFluxBackPropInterface(
        Chain(Dense(rand(5, EDDPotentials.nfeatures(cf))), Dense(rand(1, 5))),
    )
    calc = EDDPotentials.NNCalc(cell, cf, itf)
    calc
end

function allclose(x, y; kwargs...)
    all(isapprox.(x, y; kwargs...))
end

datadir = joinpath(splitdir(@__FILE__)[1], "data")
