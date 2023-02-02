#=
Code for analyse computed results
=#
import Base
import CellBase
using DirectQhull
using CellBase
using LinearAlgebra

struct ComputedRecord
    composition::Composition
    energy::Float64
end

function ComputedRecord(comp_string::Union{AbstractString, Symbol}, energy)
    ComputedRecord(Composition(comp_string), energy)
end

energy_per_atom(c::ComputedRecord) = c.energy / sum(c.composition.counts)
struct Simplex
    coords::Matrix{Float64}
    aug_inv::Matrix{Float64}
end



struct PhaseDiagram
    records::Vector{ComputedRecord}
    min_energy_records::Vector{ComputedRecord}
    min_energy_simplex::Base.IdDict{ComputedRecord,Int}
    min_energy_e_above_hull::Base.IdDict{ComputedRecord,Float64}
    qhull_input::Matrix{Float64}
    simplices::Vector{Simplex}
    simplex_indices::Vector{Vector{UInt32}}
    stable_records::Vector{ComputedRecord}
    elements::Vector{Symbol}
end

function Base.show(io::IO, p::PhaseDiagram)
    elem = join(p.elements, "-")
    println(io, "PhaseDiagram of $(elem):")
    println(io, "Total number of records: $(length(p.records))")
    stable_comps = join([formula(reduce_composition(rec.composition)) for rec in p.stable_records], ", ")
    println(io, "Stable compositions: $(stable_comps)") 
end

"""
"""
function Simplex(coords)
    aug = vcat(coords, ones(eltype(coords), 1, size(coords, 2)))
    Simplex(coords, inv(aug))
end

function bary_coords(s::Simplex, point::Vector)
    p = vcat(point, [1.0])
    s.aug_inv * p
end

"""
    contains_point(s::Simplex, point::Vector, tol=1e-8)

Check if a Simplex contains the given point.
"""
function contains_point(s::Simplex, point::Vector, tol=1e-8)
    all(x -> x > -tol, bary_coords(s, point))
end

"""
    get_coord(comp, elements)

Return the composition coordination for a given composition.
"""
function get_coord(comp, elements)
    out = zeros(length(elements)-1)
    n = sum(comp.counts)
    for j in 2:length(elements)
        out[j-1] = comp[elements[j]] / n
    end
    out
end

get_coord(rec::ComputedRecord, elements) = get_coord(rec.composition, elements)

#%
function PhaseDiagram(records)
    elements = unique(Base.Iterators.flatten(keys(x.composition) for x in records))
    sort!(elements)
    relements = elements[2:end]
    # Number of elements
    delems = length(elements)

    # Get the atomic fractions
    record_by_comp = Dict{Symbol, Vector{ComputedRecord}}()
    for record in records
        reduced_formula = CellBase.formula(reduce_composition(record.composition))
        this_formula = get(record_by_comp, reduced_formula, ComputedRecord[])
        push!(this_formula, record)
        if length(this_formula) == 1
            record_by_comp[reduced_formula] = this_formula
        end
    end

    # Minimum energy records
    min_eng_records_dict = Dict{Symbol, ComputedRecord}()
    for (formula, records) in pairs(record_by_comp)
        engs = map(energy_per_atom, records)
        min_eng_records_dict[formula] = records[argmin(engs)]
    end
    min_eng_records = collect(values(min_eng_records_dict))

    # Prepare the qhull data
    # In the shape of (nelements, n_records + 1)
    # The first element is ignored...
    qhull_points = zeros(Float64, delems, length(min_eng_records) + 1)
    i = 1
    for record in min_eng_records
        comp = record.composition
        n = sum(comp.counts)
        for (j, elem) in enumerate(relements)
            qhull_points[j, i] = comp[elem] / n
        end
        qhull_points[delems, i] = energy_per_atom(record)
        i += 1
    end
    # Add the extra point that is "above" all points
    # This used to select the "visible" facets
    qhull_points[:, i] .= 1 / delems 
    # Make sure it is "above" the existing points
    qhull_points[end, i] = maximum(qhull_points[end, 1:i-1]) + 1.

    # Compute the convex hull
    hull = ConvexHull(qhull_points)

    # Now search for simplex not including the fake point we have introduced
    iextra = size(qhull_points, 2)
    # Find the valid simplices - e.g. the ones not including the extra point that we put in
    valid_simplices = [
        col for col in eachcol(hull.simplices) if !any(x -> x == iextra, col) 
    ]

    # stable entries - those are the ones 
    stable_records_idx = filter(x-> x != iextra, hull.vertices)

    # Vector of (reduced_composition, record)
            
    # Now compute the distance to hull for each entry
    # This can be done by first computing the distance to hull of the lowest energy entry of 
    # each composition, then add the energy differences (per-atom) for the lowest energy entry
    simp = [
        Simplex(qhull_points[1:end-1, pidx]) for pidx in valid_simplices
    ]
    
    # Compute which simplex the point belongs to
    simplex_idx = Base.IdDict{ComputedRecord,Int}() 
    e_above_hull = Base.IdDict{ComputedRecord,Float64}()
    for irec in stable_records_idx
        for (j, s) in enumerate(simp)
            coord = qhull_points[1:end-1, irec]
            if contains_point(s, coord)
                simplex_idx[min_eng_records[irec]] = j 
                # Compute the coords
                bcoords = bary_coords(s, coord)
                # Compute the hull energy
                vertex_energies = qhull_points[end, valid_simplices[j]]
                ehull = dot(bcoords, vertex_energies)
                e_above_hull[min_eng_records[irec]] = qhull_points[end, irec] - ehull
                continue
            end
        end
        @assert !any(x->x == 0, simplex_idx)
    end

    PhaseDiagram(
        records,
        min_eng_records,
        simplex_idx,
        e_above_hull,
        qhull_points,
        simp,
        valid_simplices,
        min_eng_records[stable_records_idx],
        elements
    )
end

function find_simplex(phased, record)
    coord = get_coord(record, phased.elements)
    for (i, s) in enumerate(phased.simplices)
        if contains_point(s, coord)
            return i
        end
    end
    return -1
end

function get_e_above_hull(phased, record)
    # Test if it is an energy that has been seen
    for known in phased.min_energy_records
        if reduce_composition(record.composition) == reduce_composition(known.composition)
            return energy_per_atom(record) - energy_per_atom(known) + phased.min_energy_e_above_hull[known]
        end
    end

    # If not compute the hull energy from scratch
    coord = get_coord(record, phased.elements)
    i  = find_simplex(phased, record)
    bcoords = bary_coords(phased.simplices[i], coord)
    vertex_energies = phased.qhull_input[end, phased.simplex_indices[i]]
    ehull = dot(bcoords, vertex_energies)
    return energy_per_atom(record) - ehull
end

function get_decomposition(phased, record)
    coord = get_coord(record, phased.elements)
    i  = find_simplex(phased, record)
    bcoords = bary_coords(phased.simplices[i], coord)
    Dict(x=> y for (x, y) in zip(phased.min_energy_records[phased.simplex_indices[i]], bcoords))
end

records = [
    ComputedRecord(:O, 0.),
    ComputedRecord(:H2O, -1.),
    ComputedRecord(:H, 0.),
]

phased = PhaseDiagram(records)
    
    # Compute the distance to the surface defined by the simplex along the last dimension

#%%

get_e_above_hull(phased, ComputedRecord(Composition(:H2O), 1))
get_e_above_hull(phased, ComputedRecord(Composition(:H), 1))
get_e_above_hull(phased, ComputedRecord(Composition(:O), 0))