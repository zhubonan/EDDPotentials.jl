#=
Code for analyse computed results
=#
import CellBase
using DirectQhull

struct ComputedRecord
    composition::Composition
    energy::Float64
end

energy_per_atom(c::ComputedRecord) = c.energy / sum(c.composition.counts)



struct PhaseDiagram
    records::Vector{ComputedRecord}
    hull_input::Matrix
    hull::Vector{Float64}
    elements::Vector{Symbol}
end

function PhaseDiagram(records::Vector{ComputedRecord})

    elements = unique(Base.Iterators.flatten(keys(x.composition) for x in records))
    sort!(elements)
    # Number of elements
    delems = length(elements)

    # Get the atomic fractions
    record_by_comp = Dict{Symbol, Vector{ComputedRecord}}()
    for record in records
        reduced_formula = CellBase.formula(reduce_composition(record))
        this_formula = get(record_by_comp, reduced_formula, ComputedRecord[])
        push!(this_formula, record)
        if length(this_formula) == 1
            record_by_comp[reduced_formula] = this_formula
        end
    end

    # Minimum energy records
    min_eng_records = Dict{Symbol, ComputedRecord}()
    for (formula, records) in pairs(record_by_comp)
        engs = map(energy_per_atom, records)
        min_eng_records[formula] = records[argmin(engs)]
    end

    # Prepare the qhull data
    # In the shape of (nelements+1, n_records + 1)
    qhull_points = Matrix{Float64}(delems + 1, length(min_eng_records) + 1)
    i = 1
    for (formula, record) in pairs(min_eng_records)
        comp = Composition(formula)
        n = sum(comp.counts)
        for (j, elem) in enumerate(elements)
            qhull_points[j, i] = comp[elem] / n
        end
        qhull_points[j+1, i] = energy_per_atom(record)
        i += 1
    end
    # Add the extra point that is "above" all points
    # This used to select the "visible" facets
    qhull_points[:, i] .= 1 / delems 
    # Make sure it is "above" the existing points
    qhull_points[end, i] = maximum(qhull_points[end, 1:i-1]) + 1.

    # Compute the convex hull
    convex_hull = ConvexHull(qhull_points, ["QG$i"])

    # 

end
