module Systems

using CellBase
using EDDPotentials
using EDDPotentials: EDDPotentialsCore, load_model_artifact, CellFeatureConfig, build_one
using EDDPotentials.EDDPotentialsCore: CoreRepulsion
using Unitful: ustrip

const B_EXAMPLE = normpath(joinpath(@__DIR__, "..", "..", "B-eddp-example"))
const SEED_FILE = joinpath(B_EXAMPLE, "B-SEARCH.cell")

function load_boron_calculator(; gen=10)
    model_path = joinpath(B_EXAMPLE, "ensemble-gen$(gen).jld2")

    config = CellFeatureConfig(;
        elements=["B"],
        p2=[2.0, 10.0, 5.0],
        p3=[2.0, 10.0, 5.0],
        q3=[2.0, 10.0, 5.0],
        rcut2=4.0,
        geometry_sequence=true,
    )
    cf = CellFeature(config)

    artifact = load_model_artifact(model_path)
    nn = artifact.model

    elemental_energies = Dict{Symbol,Float64}(:B => -75.5214)

    EDDPCalculator(cf, nn, elemental_energies;
        shell_size=2.0,
        nmax=500,
        core_repulsion=CoreRepulsion(0.5),
    )
end

function select_structures(model_dir::String=B_EXAMPLE; n::Int=5)
    dft_dir = joinpath(model_dir, "gen10-dft")
    all_files = filter(endswith(".res"), readdir(dft_dir))
    base_files = filter(f -> !contains(f, "-shake-"), all_files)

    cells = Tuple{String,Cell}[]
    for fname in base_files
        cell = CellBase.read_res(joinpath(dft_dir, fname))
        push!(cells, (fname, cell))
    end

    sort!(cells; by=((_, c),) -> ustrip(CellBase.volume(c)))

    n = min(n, length(cells))
    if n <= 1
        idx = n == 1 ? [div(length(cells), 2) + 1] : collect(1:length(cells))
        return cells[idx]
    end

    indices = round.(Int, range(1, length(cells); length=n))
    return cells[indices]
end

function generate_structures(; n::Int=100, seedfile::String=SEED_FILE,
                             timeout=10, outdir::String="")
    gen_dir = isempty(outdir) ? joinpath(@__DIR__, "generated") : outdir
    mkpath(gen_dir)

    cells = Tuple{String,Cell}[]
    i = 0
    while i < n
        cell = build_one(seedfile; timeout)
        label = EDDPotentials.get_label(EDDPotentials.stem(seedfile))

        CellBase.write_res(joinpath(gen_dir, "$(label).res"), cell)

        push!(cells, (label, cell))
        i += 1
        i % 10 == 0 && @info "  Generated $i/$n structures"
    end
    @info "Generated $n structures in $gen_dir"
    return cells
end

function load_generated_structures(; n::Int=100, outdir::String="")
    gen_dir = isempty(outdir) ? joinpath(@__DIR__, "generated") : outdir
    if !isdir(gen_dir)
        return Tuple{String,Cell}[]
    end
    all_files = filter(endswith(".res"), readdir(gen_dir; join=true))
    sort!(all_files)
    n = min(n, length(all_files))
    n == 0 && return Tuple{String,Cell}[]
    indices = round.(Int, range(1, length(all_files); length=n))
    [(basename(f)[1:end-4], CellBase.read_res(f)) for f in all_files[indices]]
end

end
