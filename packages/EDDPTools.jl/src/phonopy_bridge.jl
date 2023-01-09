
module PhonopyInterface
#=
Use the EDDP model for finite displacements phonon calculation for 
the simple cubic polonium structure
=#

using PyCall
using EDDP
using EDDPTools
using CellBase


function __init__()
    py"""
    from ase.build import bulk
    import numpy as np
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    """
end

"Return a python Phonopy object"
function get_phonopy(cell; kwargs...)
    py"PhonopyAtoms"(unitcell=cell2phonopy(cell), kwargs...)
end

"Convert Cell to PhonopyAtoms"
function cell2phonopy(cell)
    py"PhonopyAtoms"(
        symbols=map(string, species(cell)),
        cell=PyReverseDims(cellmat(cell)),
        scaled_positions=PyReverseDims(get_scaled_positions(cell)),
    )
end

"Convert PhonopyAtoms to cell"
function phonopy2cell(atoms)
    Cell(
        Lattice(py"np.array($(atoms).cell).T"),
        map(Symbol, atoms.get_chemical_symbols()),
        atoms.get_positions() |> transpose |> collect,
    )
end

"""Get forces for a series of supercells"""
function get_phonopy_forces(pyscells, cf, model)
    scells = phonopy2cell.(pyscells)
    ctmp = EDDP.NNCalc(scells[1], cf, model)
    forces = []
    for scell in scells
        EDDP.set_positions!(ctmp, positions(scell))
        push!(forces, get_forces(ctmp) |> transpose |> x -> reshape(x, (1, size(x)...)))
    end
    PyObject(cat(forces..., dims=1))
end
end

using .PhonopyInterface: get_phonopy_forces, phonopy2cell, cell2phonopy, get_phonopy
export get_phonopy_forces, phonopy2cell, cell2phonopy, get_phonopy
