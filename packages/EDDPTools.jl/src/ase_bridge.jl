module ASEInterface

using PyCall
using EDDP
using EDDP: ShelxRecord
using Flux
using CellBase
using CellBase: read_res, read_res_many

export get_ase_atoms_and_calculator, atoms_from_cell, view_ase

function __init__()
    py"""
    import ase.calculators.calculator as calc
    import numpy as np
    from typing import List
    from ase.stress import full_3x3_to_voigt_6_stress

    class EDDPCalcBase(calc.FileIOCalculator):
        '''Base class for EDDP.jl calculator'''
        implemented_properties: List[str] = ["energy", "forces", "stress"]
        def __init__(self, atoms, eddp_calc=None, restart=None,
                        ignore_bad_restart_file=calc.Calculator._deprecated,
                        label=None, **kwargs):
            '''File-IO calculator.

            command: str
                Command used to start calculation.
            '''

            calc.Calculator.__init__(self, restart, ignore_bad_restart_file, label,
                                atoms, **kwargs)

            self.eddp_calc = eddp_calc
            self.forces = np.zeros((len(atoms), 3))
            self.stress = np.zeros((3, 3))
            self.energy = None

        def calculate(self, atoms=None, properties=['energy'],
                        system_changes=calc.all_changes):
            calc.Calculator.calculate(self, atoms, properties, system_changes)
            self.sync_atoms_cell(atoms)
            self.run_eddp()
            self.gather_results()


        def sync_atoms_cell(self, atoms):
            raise NotImplementedError


        def _set_positions_eddp2atoms(self):
            raise NotImplementedError

        def _set_positions_atoms2eddp(self):
            raise NotImplementedError

        def _set_cell_eddp2atoms(self):
            raise NotImplementedError

        def _set_cell_atoms2eddp(self):
            raise NotImplementedError

        def run_eddp(self):
            raise NotImplementedError

        def gather_results(self):

            self.results['energy'] = self.energy
            self.results['forces'] = self.forces
            self.results['stress'] = full_3x3_to_voigt_6_stress(self.stress)
    """

    np = py"np"
    global ase
    global EDDPCalc
    global visualize
    @pydef mutable struct EDDPCalc <: py"EDDPCalcBase"

        function set_eddp_calc(self, eddp_calc)
            self.eddp_calc = eddp_calc
        end

        function sync_atoms_cell(self, atoms)
            mat = np.array(atoms.get_cell())
            EDDP.set_cellmat!(self.eddp_calc, collect(transpose(mat)))
            pos = atoms.get_positions()
            EDDP.set_positions!(self.eddp_calc, collect(transpose(pos)))
        end

        function run_eddp(self)
            EDDP.calculate!(self.eddp_calc)
            self.energy = EDDP.get_energy(self.eddp_calc)
            self.forces = PyReverseDims(EDDP.get_forces(self.eddp_calc))
            self.stress = PyReverseDims(EDDP.get_stress(self.eddp_calc))
        end
    end

    ase = pyimport("ase")
    visualize = pyimport("ase.visualize")
end



"""
    atoms_from_cell(cell::Cell)

Return an ase.Atoms object with positions and cell matrix identical to the Cell object.
"""
function atoms_from_cell(cell::Cell)
    ase.Atoms(
        map(string, species(cell)),
        cell=PyReverseDims(get_cellmat(cell)),
        positions=PyReverseDims(get_positions(cell)),
        pbc=true,
    )
end

"""
    get_ase_atoms_and_calculator(calc::EDDP.AbstractCalc)

Setup ASE atoms with calculator using EDDP
"""
function get_ase_atoms_and_calculator(calc::EDDP.AbstractCalc)
    atoms = atoms_from_cell(get_cell(calc))
    ase_calc = EDDPCalc(atoms, calc)
    atoms, ase_calc
end

"""
    view_ase(cell::Cell; kwargs...)
View a `Cell` using `ase.visualize.view`.
"""
function view_ase(cell::Cell; kwargs...)
    atoms = atoms_from_cell(cell)
    visualize.view(atoms; kwargs...)
end

"""
    view_ase(cell::Vector; kwargs...)
View a series of `Cell` using `ase.visualize.view`.
"""
function view_ase(cell::Vector; kwargs...)
    atoms_list = atoms_from_cell.(cell)
    visualize.view(atoms_list; kwargs...)
end

view_ase(rec::ShelxRecord) = view_ase(read_res(rec))
view_ase(rec::Vector{ShelxRecord}) = view_ase(read_res_many(rec))

end

using .ASEInterface

export get_ase_atoms_and_calculator, atoms_from_cell
export view_ase
