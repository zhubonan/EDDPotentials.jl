module LJSystems

using CellBase
using EDDPotentials: build_one, get_label, stem
using LinearAlgebra: norm

const AIRSS_DIR = "/home/bonan/appdir/airss-git/examples/1.1"
const SEED_FILE = joinpath(AIRSS_DIR, "Al.cell")

const LJ_SIGMA = 2.0
const LJ_EPSILON = 1.0
const LJ_RCUT = 6.0
const LJ_NMAX = 5000
const FCC_E_PER_ATOM = -7.454

function generate_structures(; n::Int=100, seedfile::String=SEED_FILE,
                              timeout=10, outdir::String="")
    gen_dir = isempty(outdir) ? joinpath(@__DIR__, "lj_generated") : outdir
    mkpath(gen_dir)

    cells = Tuple{String,Cell}[]
    i = 0
    while i < n
        cell = build_one(seedfile; timeout)
        label = get_label(stem(seedfile))
        CellBase.write_cell(joinpath(gen_dir, "$(label).cell"), cell)
        push!(cells, (label, cell))
        i += 1
        i % 10 == 0 && @info "  Generated $i/$n LJ structures"
    end
    @info "Generated $n LJ structures in $gen_dir"
    return cells
end

function load_generated(; n::Int=100, outdir::String="")
    gen_dir = isempty(outdir) ? joinpath(@__DIR__, "lj_generated") : outdir
    if !isdir(gen_dir)
        return Tuple{String,Cell}[]
    end
    all_files = filter(f -> endswith(f, ".cell"), readdir(gen_dir; join=true))
    sort!(all_files)
    n = min(n, length(all_files))
    n == 0 && return Tuple{String,Cell}[]
    indices = round.(Int, range(1, length(all_files); length=n))
    [(basename(f)[1:end-5], CellBase.read_cell(f)) for f in all_files[indices]]
end

function lj_fcc_cell(; a=2.177, nrep=2)
    lat = CellBase.Lattice(a * nrep, a * nrep, a * nrep)
    pos = Float64[]
    for ix in 0:(nrep-1), iy in 0:(nrep-1), iz in 0:(nrep-1)
        append!(pos, [ix * a, iy * a, iz * a])
        append!(pos, [(ix + 0.5) * a, (iy + 0.5) * a, iz * a])
        append!(pos, [(ix + 0.5) * a, iy * a, (iz + 0.5) * a])
        append!(pos, [ix * a, (iy + 0.5) * a, (iz + 0.5) * a])
    end
    nat = div(length(pos), 3)
    pos_mat = reshape(pos, 3, nat)
    cell = Cell(lat, fill(:Al, nat), pos_mat)
    ("FCC-$(nrep)x$(nrep)x$(nrep)", cell)
end

function lj_diamond_cell(; a=3.567, nrep=2)
    lat = CellBase.Lattice(a * nrep, a * nrep, a * nrep)
    basis_frac = Float64[0.0 0.0 0.0; 0.25 0.25 0.25]
    cm = CellBase.cellmat(lat)
    pos_cart = Float64[]
    for ix in 0:(nrep-1), iy in 0:(nrep-1), iz in 0:(nrep-1)
        for b in 1:size(basis_frac, 1)
            fx = (ix + basis_frac[b, 1]) / nrep
            fy = (iy + basis_frac[b, 2]) / nrep
            fz = (iz + basis_frac[b, 3]) / nrep
            cart = cm * [fx, fy, fz]
            append!(pos_cart, cart)
        end
    end
    nat = div(length(pos_cart), 3)
    pos_mat = reshape(pos_cart, 3, nat)
    cell = Cell(lat, fill(:Al, nat), pos_mat)
    ("diamond-$(nrep)x$(nrep)x$(nrep)", cell)
end

function lj_bcc_cell(; a=2.745, nrep=2)
    lat = CellBase.Lattice(a * nrep, a * nrep, a * nrep)
    pos = Float64[]
    for ix in 0:(nrep-1), iy in 0:(nrep-1), iz in 0:(nrep-1)
        append!(pos, [ix * a, iy * a, iz * a])
        append!(pos, [(ix + 0.5) * a, (iy + 0.5) * a, (iz + 0.5) * a])
    end
    nat = div(length(pos), 3)
    pos_mat = reshape(pos, 3, nat)
    cell = Cell(lat, fill(:Al, nat), pos_mat)
    ("BCC-$(nrep)x$(nrep)x$(nrep)", cell)
end

function lj_sc_cell(; a=2.5, nrep=2)
    lat = CellBase.Lattice(a * nrep, a * nrep, a * nrep)
    pos = Float64[]
    for ix in 0:(nrep-1), iy in 0:(nrep-1), iz in 0:(nrep-1)
        append!(pos, [ix * a, iy * a, iz * a])
    end
    nat = div(length(pos), 3)
    pos_mat = reshape(pos, 3, nat)
    cell = Cell(lat, fill(:Al, nat), pos_mat)
    ("SC-$(nrep)x$(nrep)x$(nrep)", cell)
end

end
