#=
Common functions for helping tests
=#
using EDDP
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



function _generate_cf(cell::Cell)
    EDDP.CellFeature(
            unique(species(cell));
            p2=[6, 12],
            q3=[2, 3],
            p3=[2, 3],
        )
end


function _get_calc()
    cell = _h2_cell()
    cf = _generate_cf(cell)
    itf = EDDP.ManualFluxBackPropInterface(
        Chain(Dense(rand(5, EDDP.nfeatures(cf))), Dense(rand(1, 5))),
    )
    calc = EDDP.NNCalc(cell, cf, itf)
    calc
end

function allclose(x, y; kwargs...)
    all(isapprox.(x, y; kwargs...))
end

datadir = joinpath(splitdir(@__FILE__)[1], "data")
