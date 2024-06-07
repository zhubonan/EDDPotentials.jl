
module PhonopyInterface
#=
Use the EDDPotential model for finite displacements phonon calculation for 
the simple cubic polonium structure
=#

using PyCall
using EDDPotential
using EDDPotentialTools
using CellBase
using LinearAlgebra


function __init__()
    py"""
    from ase.build import bulk
    import numpy as np
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.file_IO import write_FORCE_SETS
    """
end

"Return a python Phonopy object"
function get_phonopy(cell; kwargs...)
    py"Phonopy"(unitcell=cell2phonopy(cell); kwargs...)
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
    ctmp = EDDPotential.NNCalc(scells[1], cf, model)
    forces = []
    for scell in scells
        EDDPotential.set_positions!(ctmp, positions(scell))
        push!(forces, get_forces(ctmp) |> transpose |> x -> reshape(x, (1, size(x)...)))
    end
    PyObject(cat(forces..., dims=1))
end

"""
    run_phonon(calc; supercell_matrix, kwargs...)

Run phonon calculation by calling `phonopy`. 
A calculator object containing the input structure with force-minimised should be passed.
Return the `Phonopy` python object. Computed forces and parameters are saved to `out_dir` which
defaults to the current directory.
Then can be used to run further calculations through phonopy command line interface. 

"""
function run_phonon(
    calc;
    out_dir="./phonon",
    phonon_save_name="phonopy_params.yaml",
    force_set_filename="FORCE_SETS",
    supercell_matrix,
    distance=0.01,
    kwargs...,
)
    phonon = get_phonopy(get_cell(calc); supercell_matrix, kwargs...)
    phonon.generate_displacements(; distance=distance)
    scells = phonon.supercells_with_displacements
    cf = calc.cf
    model = calc.nninterface

    pforces = norm.(eachcol(EDDPotential.get_forces(calc)))
    if any(pforces .> 1e-4)
        @warn "Residual forces in the input structure is too large: $(maximum(pforces))!"
    end


    @info "Computing forces for supercell displacements"
    forces = get_phonopy_forces(scells, cf, model)
    phonon.forces = forces
    phonon.produce_force_constants()

    isdir(out_dir) || mkdir(out_dir)
    phonon_save_name = joinpath(out_dir, phonon_save_name)
    force_set_filename = joinpath(out_dir, force_set_filename)
    structure_filename = joinpath(out_dir, "input.res")
    poscar_name = joinpath(out_dir, "POSCAR")

    @info "Input structure written to: $structure_filename."
    write_res(structure_filename, get_cell(calc))
    write_poscar(poscar_name, get_cell(calc))

    @info "Force set file written to: $force_set_filename."
    py"write_FORCE_SETS"(phonon.dataset, force_set_filename)
    @info "Phonopy configuration file written to: $phonon_save_name."
    phonon.save(phonon_save_name)
    return phonon
end

end

using .PhonopyInterface:
    get_phonopy_forces, phonopy2cell, cell2phonopy, get_phonopy, run_phonon
export get_phonopy_forces, phonopy2cell, cell2phonopy, get_phonopy, run_phonon
