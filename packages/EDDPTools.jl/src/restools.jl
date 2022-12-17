#=
Code for analysing data
=#

using DataFrames
using CellBase
using CellBase:read_res_many

"""
    load_res(paths;basic=false)

Load SHELX files into a datafrmae. Paths can be a iterator of file paths or file handles.
"""
function load_res(paths;basic=false)
    data = CellBase.Cell{Float64}[]
    for path in paths
        append!(data, read_res_many(path))
    end
    frame = DataFrame(:cell=>data)

    # Move information stored in metadata as columns
    for col in [:label, :pressure, :volume, :enthalpy, :spin, :abs_spin, :natoms, :symm, :flag1, :flag2, :flag3]
        frame[!, col] = map(x -> x.metadata[col], frame.cell)
    end
    frame[!, :_volume] = frame[!, :volume]
    frame[!, :_natoms] = frame[!, :natoms]
    frame[!, :volume] = map(volume, frame.cell)
    frame[!, :natoms] = map(natoms, frame.cell)

    # Composition
    frame[!, :composition] = Composition.(frame.cell)
    frame[!, :formula] = formula.(frame.composition)
    if basic
        return frame
    end
    _enrich_properties(frame)
end

"""
    _enrich_properties(frame)

Enrich the properties of the dataframe.

"""
function _enrich_properties(frame)
    frame[!, :nform] = map(CellBase.nform, frame.composition)
    frame[!, :reduced_composition] = map(CellBase.reduce_composition, frame.composition)
    frame[!, :reduced_formula] = map(CellBase.formula, frame.reduced_composition)

    # Per atom/formula unit quantities
    frame[!, :enthalpy_per_atom] = frame.enthalpy ./ frame.natoms 
    frame[!, :enthalpy_per_form] = frame.enthalpy ./ frame.nform

    frame[!, :volume_per_atom] = frame.volume ./ frame.natoms 
    frame[!, :volume_per_form] = frame.volume ./ frame.nform
    frame
end