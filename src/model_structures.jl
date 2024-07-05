using Combinatorics: with_replacement_combinations
"""
    model_structures(xranges, sym1::Symbol, sym2::Symbol, rcut)

Generate model structure with pairs
"""
function model_structures(xranges, sym1::Symbol, sym2::Symbol, rcut)
    cells = []
    a = max(xranges[end] + rcut, 15.0)
    for r in xranges
        latt = Lattice(a, a, a)
        symbols = [sym1, sym2]
        positions = [[0.0, 0.0, 0.0], [r, 0.0, 0.0]]
        push!(cells, Cell(latt, symbols, hcat(positions...)))
        cells[end].metadata[:label] = "$sym1-$sym2-$(round(r, digits=3))"
    end
    cells
end

"""
    model_structures(xranges, sym1::Symbol, sym2::Symbol, sym3::Symbol, rcut)

Generate model structure with triplets
"""
function model_structures(xranges, sym1::Symbol, sym2::Symbol, sym3::Symbol, rcut)
    cells = []
    a = max(xranges[end] + rcut, 15.0)
    for r in xranges
        latt = Lattice(a, a, a)
        symbols = [sym1, sym2, sym3]
        positions = [[0.0, 0.0, 0.0], [r, 0.0, 0.0], [0.5 * r, r * sin(pi / 3), 0.0]]
        push!(cells, Cell(latt, symbols, hcat(positions...)))
        cells[end].metadata[:label] = "$sym1-$sym2-$sym3-$(round(r, digits=3))"
    end
    cells
end

"""
    model_structures(xranges, sym1::Symbol, sym2::Symbol, sym3::Symbol, sym4::Symbol, rcut)

Generate model structure with quadruplets
"""
function model_structures(
    xranges,
    sym1::Symbol,
    sym2::Symbol,
    sym3::Symbol,
    sym4::Symbol,
    rcut,
)
    cells = []
    a = max(xranges[end] + rcut, 15)
    for r in xranges
        latt = Lattice(a, a, a)
        symbols = [sym1, sym2, sym3, sym4]
        positions = [[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]]
        pos = hcat(positions...) .* r ./ 2
        push!(cells, Cell(latt, symbols, pos))
        cells[end].metadata[:label] = "$sym1-$sym2-$sym3-$sym4-$(round(r, digits=3))"
    end
    cells
end

"""Center the cell"""
function centre!(cell)
    mat = cellmat(cell)
    lcom = sum(mat, dims=2) ./ 2
    pcom = sum(positions(cell), dims=2) ./ length(cell)
    d = lcom .- pcom
    positions(cell) .+= d
    cell
end

"""
    model_structures(sym;order=2, rmin=0.5, rmax=6.0, npts=30, rcut=6.0)

Generate model structures for a given cell and order.

Arguments:

- `sym`: A sequence of symbols

Keyword Arguments:

- `order`: Order of the model, can be `2`, `3` or `4`.
- `rmin`: Minimum distance.
- `rmax`: Maximum distance.
- `npts`: The number of points to be generated.
"""
function model_structures(sym; order=2, rmin=1.0, rmax=6.0, npts=30, rcut=6.0)
    xs = collect(LinRange(rmin, rmax, npts))
    combs = collect(with_replacement_combinations(sym, order))
    combs, [model_structures(xs, comb..., rcut) for comb in combs]
end

"""
    model_energies(cell, model, cf; order=2, rmin=0.5, rmax=6.0, npts=30, core=CoreRepulsion(1.0))

Generate model structures and compute their energies.
"""
function model_energies(
    cell,
    model,
    cf;
    order=2,
    rmin=0.5,
    rmax=6.0,
    npts=30,
    core=CoreRepulsion(1.0),
)
    xs = collect(LinRange(rmin, rmax, npts))
    combs, structures = model_structures(
        unique(species(cell)),
        order=order,
        rmin=rmin,
        rmax=rmax,
        npts=npts,
        rcut=suggest_rcut(cf),
    )
    energies = [get_energy(NNCalc(x, cf, model; core)) / length(x) for x in structures]
    combs, xs, energies
end
