
"""
    glob_allow_abs(path)

Allow path to start with "/" when globbing by converting it to a relative path.
"""
function glob_allow_abs(path)
    if startswith(path, "/")
        path = relpath(path)
    end
    return glob(path)
end

"""
    _split_vector(c, nsplit::Vararg{Real}; shuffle=true, seed=42)

Split a vector by fractions.
"""
function _split_vector(c, nsplit::Vararg{Real}; shuffle=true, seed=42)
    ntot = length(c)
    intsplit = nsplit .* ntot .|> floor .|> Int
    _split_vector(c, intsplit...; shuffle, seed)
end

"""
    _h2_cell(l=4.0, factor=1.0)

Generate a cell containing two hydrogen atoms.
"""
function _h2_cell(l=4.0, factor=1.0)
    tmp = Float64[
        0 0.1
        0 2.0
        0.1 0
    ] .* factor
    Cell(Lattice(l, l, l), [:H, :H], tmp)
end

"""
    _h2o_cell(l=4.0, factor=1.0)

Generate a cell containing two hydrogen atoms and one oxygen atom.
"""
function _h2o_cell(l=4.0, factor=1.0)
    tmp = Float64[
        0 0.1 1
        0 1 1
        0.1 0 1
    ] .* factor
    Cell(Lattice(l, l, l), [:H, :H, :O], tmp)
end

"""
    _lco_cell()

Generate a cell containing 12 Li, 12 Co, and 12 O atoms.
"""
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
