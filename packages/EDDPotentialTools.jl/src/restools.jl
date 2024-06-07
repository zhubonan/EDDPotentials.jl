#=
Code for analysing data
=#


using DataFrames
using CellBase
using EDDPotential: read_shelx_record, extract_res, ShelxRecord
using CellBase: read_res_many
import CellBase

export as_dataframe

"""
    as_dataframe(records; basic=false)
Construct a `DataFrame` from a vector for ShelxRecord objects.
"""
function as_dataframe(records::Vector{ShelxRecord}; basic=false, sort_enthalpy=true)
    frame = DataFrame(:record => records)

    # Move information stored in metadata as columns
    for col in [
        :label,
        :pressure,
        :volume,
        :enthalpy,
        :spin,
        :abs_spin,
        :natoms,
        :symm,
        :flag1,
        :flag2,
        :flag3,
    ]
        frame[!, col] = map(x -> getproperty(x.titl, col), frame.record)
    end

    # Composition
    frame[!, :composition] = map(x -> x.comp, frame.record)
    frame[!, :formula] = map(x -> formula(x.comp), frame.record)
    if basic
        return frame
    end
    # Sorting
    _enrich_properties(frame)
    if sort_enthalpy
        sort!(frame, [:enthalpy_per_atom])
    end
    frame
end

"""
    as_dataframe(files; basic=false)
Construct a `DataFrame` from a list of (packed) SHELX files.
"""
function as_dataframe(files; basic=false)
    records = read_shelx_record(files)
    as_dataframe(records; basic)
end

"""
    _enrich_properties(frame)

Enrich the properties of the dataframe.

"""
function _enrich_properties(frame)
    frame[!, :nform] = map(CellBase.nform, frame.composition)
    frame[!, :reduced_composition] = map(x -> x.reduced_comp, frame.record)
    frame[!, :reduced_formula] = map(x -> formula(x.reduced_comp), frame.record)

    # Per atom/formula unit quantities
    frame[!, :enthalpy_per_atom] = frame.enthalpy ./ frame.natoms
    frame[!, :enthalpy_per_form] = frame.enthalpy ./ frame.nform

    frame[!, :volume_per_atom] = frame.volume ./ frame.natoms
    frame[!, :volume_per_form] = frame.volume ./ frame.nform
    frame
end



"""
    deduplicate(func, df, cut_off=0.1)

Deduplicate the structures using `func`. `func` take arguments (df, i, j) for computing
the distances. Return a vector of labels for each row.

This uses an one-pass algorithm. Items that have group assigned (other than itself),
will not be used in any further comparison. 

The `cut_off` is scaled by the minimum bond length.
"""
function deduplicate(func, df, cut_off=0.1)
    l = size(df, 1)
    taken = collect(1:l)
    for i = 1:l
        if taken[i] != i
            continue
        end
        for j = i+1:l
            dist = func(df, i, j)
            if dist < cut_off
                taken[j] = i
            end
        end
    end
    taken
end

"""
    evdist(df, i, j)

Compute distance via energy and volume difference ``\\sqrt(\\DeltaE^2 + \\DeltaV^2)``.
"""
function evdist(df, i, j)
    de = df[i, :enthalpy_per_atom] - df[j, :enthalpy_per_atom]
    dv = df[i, :volume_per_atom] - df[j, :volume_per_atom]
    sqrt(de^2 + dv^2)
end

# Gather the structures to refine
"""
    select_refine(frame; dist_func=evdist, dist_cutoff=0.001, top_n=10)

Select records to be used for refinement. 
"""
function select_refine(frame; dist_func=evdist, dist_cutoff=0.001, top_n=10)
    labels = String[]
    for (name, group) in pairs(groupby(frame, :reduced_formula))
        @info "Processing $(name[:reduced_formula])"
        sorted_group = sort(group, :enthalpy_per_atom)
        idx = deduplicate(dist_func, sorted_group, dist_cutoff)
        uidx = unique(idx)
        length(uidx) > top_n && (uidx = uidx[1:top_n])
        selected = filter(x -> x in uidx, idx)
        append!(labels, sorted_group[selected, :label])
    end
    frame[map(x -> x in labels, frame.label), :]
end

"""
    write_res(outdir::AbstractString, df::DataFrame)

Write rows of a `DataFrame` as SHELX files into a target path.
The resulting files will be named as `<record.titl.label>.res`.

Note: 
This can results in undefined behaviour if non-identical records
share the same *label*.
"""
function CellBase.write_res(outdir::AbstractString, df::DataFrame)
    extract_res(df.record; outdir)
end
