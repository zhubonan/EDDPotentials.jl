#=
Code for MINSEP analysis
=#

using CellBase

"""
    SpecieSeparation

Separation between two species
"""
mutable struct SepecieSeparation
    pair::Pair{Symbol, Symbol}
    minsep::Float64
    meansep::Float64
    count::Int
end

"""
Return a vector of pairs of unique species in the cell
"""
function get_specie_pairs(symbols;permute=false)
    usymbols = unique(symbols)
    sort!(usymbols)
    pairs = Pair{Symbol, Symbol}[]
    for i in 1:length(usymbols)
        for j in i:length(usymbols)
            pair1 = usymbols[i] => usymbols[j]
            push!(pairs, pair1)
            if i != j && permute
                pair2 = usymbols[j] => usymbols[i]
                push!(pairs, pair2)
            end
        end
    end
    pairs
end

"""
    compute_specie_separations(cell::Cell;rcut=6.0, nmax=2000)

Compute the specie-wise minimum and mean separations and return a dictionary covering
all species pairs.
"""
function compute_specie_separations(cell::Cell;rcut=6.0, nmax=2000)
    nl = NeighbourList(cell, rcut, nmax)
    symbols = species(cell)
    usymbols = unique(symbols)
    sort!(usymbols)
    seps = Dict{Pair{Symbol, Symbol}, SepecieSeparation}()
    for pair in get_specie_pairs(species(cell);permute=false)
        val = SepecieSeparation(pair, 999., 0, 0)
        seps[pair] = val
        seps[pair.second => pair.first] = val
    end

    for i in 1:natoms(cell)
        for (j, _, dist) in  eachneighbour(nl, i)
            pair = symbols[i] => symbols[j]
            obj = seps[pair]
            if obj.minsep > dist
                obj.minsep = dist
            end
            # Update the meansep
            obj.meansep = (obj.meansep * obj.count + dist) / (obj.count + 1)
            obj.count += 1
        end
    end
    seps
end

"""
    gather_minsep_stats(minsep_res)

Gather species-wise distances stats from a sequence of computed values.
"""
function gather_minsep_stats(minsep_res)
    # Gather all pairs that have occured
    all_pairs = Set{Pair{Symbol, Symbol}}()
    map( x -> push!.(Ref(all_pairs), keys(x)), minsep_res);
    minsep_stats = Dict{Pair{Symbol, Symbol}, Vector{Float64}}(key => Float64[] for key in all_pairs)
    meansep_stats = Dict{Pair{Symbol, Symbol}, Vector{Float64}}(key => Float64[] for key in all_pairs)

    for pair in all_pairs
        minsep = minsep_stats[pair]
        meansep = meansep_stats[pair]
        for x in minsep_res
            if pair in keys(x)
                push!(minsep, x[pair].minsep)
                push!(meansep, x[pair].meansep)
            end
        end
    end
    minsep_stats, meansep_stats
end

"""
    gather_minsep_stats(cells::Vector{Cell})

Gather species-wise MINSEP stats from a set of cell files.
"""
gather_minsep_stats(c::Vector{T};kwargs...) where {T<:Cell} = gather_minsep_stats([compute_specie_separations(x;kwargs...) for x in c])