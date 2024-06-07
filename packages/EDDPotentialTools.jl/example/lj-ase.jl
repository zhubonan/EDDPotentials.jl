#=
Example for using EDDPotential.jl using ASE

It is probably easier to run all this from the Julia side. 
Alternatively, juliacall can be used to used from the python side....
=#

using Pkg
Pkg.activate("packages/EDDPotentialTools.jl")
using CellBase
using EDDPotential
using EDDPotentialTools
using PyCall

# Create a random structure
cell = Cell(Lattice(5.0, 5.0, 5.0), repeat([:H], 10), rand(3, 10) .* 5)
ecalc = EDDPotential.lj_like_calc(cell)
atoms, calc = EDDPotentialTools.get_ase_atoms_and_calculator(ecalc)

# Optimize the structure
opt_class = pyimport("ase.optimize.bfgs").BFGS
opt = opt_class(atoms)
opt.run(steps=300)

println(atoms.get_stress())
println(atoms.get_forces())

# VC optimisation using the UnitCellFilter object
vc_atoms = pyimport("ase.constraints").UnitCellFilter(atoms)
gpa = pyimport("ase.units").GPa
opt = opt_class(vc_atoms)
opt.run(steps=300)

println(atoms.get_stress() .* gpa)
println(atoms.get_forces())
