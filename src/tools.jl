#=
Various tool functions for workflow managements
=#

using CellBase: rattle!, reduce, Composition
using ProgressMeter: @showprogress
import CellBase: write_res
using StatsBase
using Base.Threads
using JLD2
using Dates
using UUIDs

"""
    update_metadata!(vc::AbstractCalc, label;symprec=1e-2)

Update the metadata attached to a `Cell`` object
"""
function update_metadata!(vc::AbstractCalc, label; symprec=1e-2)
    this_cell = get_cell(vc)
    # Set metadata
    this_cell.metadata[:enthalpy] = get_enthalpy(vc)
    this_cell.metadata[:volume] = volume(this_cell)
    this_cell.metadata[:pressure] = get_pressure_gpa(vc)
    this_cell.metadata[:label] = label
    symm = CellBase.get_international(this_cell, symprec)
    this_cell.metadata[:symm] = "($(symm))"
    # Write to the file
    vc
end

"""
    write_res(path, vc::VariableCellCalc;symprec=1e-2, label="EDDP")

Write structure in VariableCellCalc as SHELX file.
"""
function write_res(path, vc::VariableCellCalc; symprec=1e-2, label="EDDP")
    update_metadata!(vc, label; symprec)
    write_res(path, get_cell(vc))
end



"""
Check if a feature vector already present in an array of vectors
"""
function is_unique_fvec(all_fvecs, fvec; tol=1e-2, lim=5)
    match = false
    for ref in all_fvecs
        dist = CellBase.fingerprint_distance(ref, fvec; lim)
        if dist < tol
            match = true
            break
        end
    end
    !match
end

ensure_dir(path) = isdir(path) || mkdir(path)

function get_label(seedname)
    dt = Dates.format(now(), "yy-mm-dd-HH-MM-SS")
    suffix = string(uuid4())[end-7:end]
    "$(seedname)-$(dt)-$(suffix)"
end

"Return the *stea* part of a file name"
stem(x::AbstractString) = splitext(splitpath(x)[end])[1]

swapext(fname, new) = splitext(fname)[1] * new



raw"""

Generate LJ like pair-wise interactions

```math
F = \alpha(-2f(r, rc)^a + f(r, rc)^{2a})

The equilibrium position is at ``r_c/2``.
```

Support only single a element for now.
"""
function lj_like_calc(cell::Cell; α=1.0, a=6, rc=3.0)
    elem = unique(species(cell))
    @assert length(elem) == 1 "Only works for single specie Cell for now."
    cf = EDDP.CellFeature(elem, p2=[a, 2a], p3=[], q3=[], rcut2=rc)
    model = EDDP.LinearInterface([0, -2, 1.0] .* α)
    EDDP.NNCalc(cell, cf, model)
end

