#=
Code for analyse computed results
=#
import Base
import CellBase
using DirectQhull
using CellBase
using LinearAlgebra

"""
Abstract record that can be used to construct a phase diagram
"""
abstract type AbstractRecord end
function record_energy end
function record_comp end
function record_reduced_comp end
function record_id end

struct ComputedRecord <: AbstractRecord
    composition::Composition
    energy::Float64
    record_id::String
    reduced_composition::Composition
    function ComputedRecord(comp::Composition, energy, record_id="")
        new(comp, energy, record_id, reduce_composition(comp))
    end
end

record_energy(c::ComputedRecord) = c.energy
record_comp(c::ComputedRecord) = c.composition
record_reduced_comp(c::ComputedRecord) = c.reduced_composition
record_id(c::ComputedRecord) = c.record_id


function ComputedRecord(comp_string::Union{AbstractString,Symbol}, args...)
    comp = Composition(comp_string)
    ComputedRecord(comp, args...)
end

energy_per_atom(c::AbstractRecord) = record_energy(c) / natoms(record_comp(c))


"""
Type that represents a simplex in N-dimensional space
"""
struct Simplex
    coords::Matrix{Float64}
    aug::Matrix{Float64}
    aug_inv::Matrix{Float64}
end



struct PhaseDiagram{T}
    records::Vector{T}
    formation_energies::Vector{Float64}
    min_energy_records::Vector{T}
    min_energy_simplex::Base.IdDict{T,Int}
    min_energy_e_above_hull::Base.IdDict{T,Float64}
    qhull_input::Matrix{Float64}
    simplices::Vector{Simplex}
    simplex_indices::Vector{Vector{UInt32}}
    stable_records::Vector{T}
    elements::Vector{Symbol}
end

nelem(p::PhaseDiagram) = length(p.elements)

function Base.show(io::IO, p::PhaseDiagram)
    elem = join(p.elements, "-")
    println(io, "PhaseDiagram of $(elem):")
    println(io, "Total number of records: $(length(p.records))")
    stable_comps =
        join([formula(record_reduced_comp(rec)) for rec in p.stable_records], ", ")
    println(io, "Stable compositions: $(stable_comps)")
end

"""
    Simplex(coords)

Construct a simplex from a matrix of the coordinates
"""
function Simplex(coords)
    aug = vcat(coords, ones(eltype(coords), 1, size(coords, 2)))
    Simplex(coords, aug, inv(aug))
end

"""
    bary_coords(s::Simplex, point::Vector)

Obtain barycentric coordinates for a given point
"""
function bary_coords(s::Simplex, point::Vector)
    p = vcat(point, [1.0])
    s.aug_inv * p
end

bary_coords(s, p, phased::PhaseDiagram) = bary_coords(s, get_coord(p, phased))

"""
Convert barycentric coordinate to normal point
"""
function coords_from_bary(s::Simplex, bary::Vector)
    s.aug[1:end-1, :] * bary
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
function get_coord(comp::Composition, elements::Vector{Symbol})
    out = zeros(length(elements) - 1)
    n = sum(comp.counts)
    for j = 2:length(elements)
        out[j-1] = comp[elements[j]] / n
    end
    out
end

"""
    _formation_energies(records, elemental_forms)
Return the formation energy per atom.
"""
function _formation_energies(records, elemental_forms)
    formation_energies = zeros(length(records))
    for (i, record) in enumerate(records)
        ref = 0.0
        comp = record_comp(record)
        for (elem, eng) in pairs(elemental_forms)
            ref += comp[elem] * eng
        end
        form = (record_energy(record) - ref) / natoms(comp)
        formation_energies[i] = form
    end
    formation_energies
end

"""
    get_composition_coord(comp, elements)

Return the composition coordinates
"""
function get_composition_coord(comp, elements)
    n = natoms(comp)
    out = zeros(Float64, length(elements) - 1)
    for j = 1:length(elements)-1
        out[j] = comp[elements[j+1]] / n
    end
    out
end

get_composition_coord(record::AbstractRecord, elements) =
    get_composition_coord(record_comp(record), elements)


get_coord(rec::AbstractRecord, elements::Vector) = get_coord(record_comp(rec), elements)
get_coord(x, phased::PhaseDiagram) = get_coord(x, phased.elements)

#%
function PhaseDiagram(records::Vector{T}) where {T}
    elements = unique(Base.Iterators.flatten(keys(record_comp(x)) for x in records))
    sort!(elements)
    relements = elements[2:end]
    # Number of elements
    delems = length(elements)

    # Get the atomic fractions
    record_by_comp = Dict{Symbol,Vector{T}}()
    for record in records
        reduced_formula = CellBase.formula(reduce_composition(record_comp(record)))
        this_formula = get(record_by_comp, reduced_formula, T[])
        push!(this_formula, record)
        if length(this_formula) == 1
            record_by_comp[reduced_formula] = this_formula
        end
    end

    # Minimum energy records
    min_eng_records_dict = Dict{Symbol,T}()
    for (formula, records) in pairs(record_by_comp)
        engs = map(energy_per_atom, records)
        min_eng_records_dict[formula] = records[argmin(engs)]
    end

    # Compute the formation energies
    elemental_forms = Dict{Symbol,Float64}()
    for (rform, record) in pairs(min_eng_records_dict)
        if length(Composition(rform).species) == 1
            elemental_forms[Symbol(rform)] = energy_per_atom(record)
        end
    end

    # Check we have got all elemental references
    for elem in elements
        if !(elem in keys(elemental_forms))
            throw(ErrorException("Elemental reference missing for $elem"))
        end
    end

    # Compute the formation energies for each record, per atom
    formation_energies = _formation_energies(records, elemental_forms)

    min_eng_records = collect(values(min_eng_records_dict))


    # Prepare the qhull data
    # In the shape of (nelements, n_records + 1)
    # The first element is ignored...
    qhull_points = zeros(Float64, delems, length(min_eng_records) + 1)
    i = 1
    for record in min_eng_records
        eform = formation_energies[findfirst(x -> x == record, records)]
        comp = record_comp(record)
        n = natoms(comp)
        for (j, elem) in enumerate(relements)
            qhull_points[j, i] = comp[elem] / n
        end
        # Use the formation energy as the hull z value
        qhull_points[delems, i] = eform
        i += 1
    end
    # Add the extra point that is "above" all points
    # This used to select the "visible" facets
    qhull_points[:, i] .= 1 / delems
    # Make sure it is "above" the existing points
    qhull_points[end, i] = maximum(qhull_points[end, 1:i-1]) + 1.0

    # Compute the convex hull
    hull = ConvexHull(qhull_points)

    # Now search for simplex not including the fake point we have introduced
    iextra = size(qhull_points, 2)
    # Find the valid simplices - e.g. the ones not including the extra point that we put in
    valid_simplices = Vector{UInt32}[
        col for col in eachcol(hull.simplices) if !any(x -> x == iextra, col)
    ]

    # stable entries - those are the ones 
    stable_records_idx = filter(x -> x != iextra, hull.vertices)

    # Vector of (reduced_composition, record)

    # Now compute the distance to hull for each entry
    # This can be done by first computing the distance to hull of the lowest energy entry of 
    # each composition, then add the energy differences (per-atom) for the lowest energy entry
    simp = [Simplex(qhull_points[1:end-1, pidx]) for pidx in valid_simplices]

    # Compute which simplex the point belongs to
    simplex_idx = Base.IdDict{T,Int}()
    e_above_hull = Base.IdDict{T,Float64}()
    for (irec, rec) in enumerate(min_eng_records)
        for (j, s) in enumerate(simp)
            coord = qhull_points[1:end-1, irec]
            if contains_point(s, coord)
                simplex_idx[rec] = j
                # Compute the coords
                bcoords = bary_coords(s, coord)
                # Compute the hull energy
                vertex_energies = qhull_points[end, valid_simplices[j]]
                ehull = dot(bcoords, vertex_energies)
                e_above_hull[rec] = qhull_points[end, irec] - ehull
                continue
            end
        end
        @assert !any(x -> x == 0, simplex_idx)
    end

    PhaseDiagram(
        records,
        formation_energies,
        min_eng_records,
        simplex_idx,
        e_above_hull,
        qhull_points,
        simp,
        valid_simplices,
        min_eng_records[stable_records_idx],
        elements,
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
        if record_reduced_comp(record) == record_reduced_comp(known)
            return energy_per_atom(record) - energy_per_atom(known) +
                   phased.min_energy_e_above_hull[known]
        end
    end

    # If not compute the hull energy from scratch
    coord = get_coord(record, phased.elements)
    i = find_simplex(phased, record)
    bcoords = bary_coords(phased.simplices[i], coord)
    vertex_energies = phased.qhull_input[end, phased.simplex_indices[i]]
    ehull = dot(bcoords, vertex_energies)
    return energy_per_atom(record) - ehull
end

"""
    get_decomposition(phased, record)

Return the decomposition of a record
"""
function get_decomposition(phased, record)
    coord = get_coord(record, phased.elements)
    i = find_simplex(phased, record)
    bcoords = bary_coords(phased.simplices[i], coord)
    Dict(
        x => y for
        (x, y) in zip(phased.min_energy_records[phased.simplex_indices[i]], bcoords)
    )
end


"""
    PhaseDiagram(sc::StructureContainer)

Construct a `PhaseDiagram` object from a `StructureContainer`.
"""
function PhaseDiagram(sc::StructureContainer)
    records = [
        ComputedRecord(Composition(x), y, get(x.metadata, :label, "")) for
        (x, y) in zip(sc.structures, sc.H)
    ]
    PhaseDiagram(records)
end


"""
Return data needed for constructing a 2D convex hull diagram
"""
function get_2d_plot_data(phased::PhaseDiagram; threshold=0.5)
    @assert nelem(phased) == 2
    hulls = get_e_above_hull.(Ref(phased), phased.records)
    mask = hulls .< threshold

    x = [
        get_composition_coord(record, phased.elements)[1] for record in phased.records[mask]
    ]
    y = phased.formation_energies[mask]

    stable_idx = findall(x -> x in phased.stable_records, phased.records)
    stable_x = [
        get_composition_coord(record, phased.elements)[1] for
        record in phased.records[stable_idx]
    ]
    stable_y = phased.formation_energies[stable_idx]
    stable_y = stable_y[sortperm(stable_x)]
    sort!(stable_x)

    stable_formula =
        [record_reduced_comp(record) |> formula for record in phased.stable_records]
    stable_entry_id = [record_id(record) for record in phased.stable_records]

    (;
        x,
        y,
        stable_formula,
        stable_x,
        stable_y,
        stable_entry_id,
        elements=phased.elements,
        e_above_hull=hulls,
        record_ids=map(x -> record_id(x), phased.records[mask]),
    )
end


"""
    get_ternary_hulldata(phased::PhaseDiagram)

Return the ternary coordinates for all stable and non stable phases 
"""
function get_ternary_hulldata(phased::PhaseDiagram)
    hullbc = phased.qhull_input[1:end-1, 1:end-1]
    hulla = [(1.0 .- sum.(eachcol(hullbc)))...;;]
    hullabc = vcat(hulla, hullbc)

    # Divide into stable and unstable ones 
    stable_mask =
        map(x -> phased.min_energy_records[x] in phased.stable_records, 1:size(hulla, 2))
    unstable_mask = map(!, stable_mask)

    labels = map(x -> formula(record_comp(x)), phased.min_energy_records)
    ehull = [phased.min_energy_e_above_hull[x] for x in phased.min_energy_records]
    (
        abc_stable=hullabc[:, stable_mask],
        labels_stable=labels[stable_mask],
        e_above_hull_stable=ehull[stable_mask],
        abc_unstable=hullabc[:, unstable_mask],
        labels_unstable=labels[unstable_mask],
        e_above_hull_unstable=ehull[unstable_mask],
        elements=phased.elements,
    )
end
