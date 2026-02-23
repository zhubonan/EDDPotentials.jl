"""
    CellRelaxationModifier{T}

Enables variable cell optimization by augmenting positions with lattice degrees of freedom.

This modifier transforms the calculator to allow simultaneous relaxation of atomic
positions and cell parameters. It implements the approach from:
> E. B. Tadmor, G. S. Smith, N. Bernstein, and E. Kaxiras,
> Phys. Rev. B 59, 235 (1999)

Based on ASE's UnitCellFilter.

# Mathematical Details

The modifier:
1. Stores the original lattice and computes deformation gradient F such that F*C₀ = C
2. Augments positions: [atomic_positions, lattice_positions] where lattice positions
   encode the deformation gradient
3. Transforms forces to include virial contributions
4. Applies external pressure to the stress tensor

# Fields
- `orig_lattice::Lattice{T}`: Original lattice for deformation gradient computation
- `lattice_factor::T`: Scaling factor for lattice positions (typically number of atoms)
- `external_pressure::Matrix{T}`: External pressure (3×3 symmetric matrix)
- `deform_grad::Matrix{T}`: Workspace for deformation gradient
- `augmented_positions::Matrix{T}`: Workspace for augmented positions
- `augmented_forces::Matrix{T}`: Workspace for augmented forces

# Example
```julia
# Create modifier with external pressure
P = diagm([3.0, 3.0, 3.0])  # 3 GPa hydrostatic pressure
modifier = CellRelaxationModifier(cell; external_pressure=P)

# Use in MetaCalculator
calc = MetaCalculator(source, [modifier])

# Change pressure during relaxation
modifier.external_pressure .= diagm([5.0, 5.0, 5.0])
```
"""
mutable struct CellRelaxationModifier{T} <: AbstractModifier
    # Original lattice for deformation gradient computation
    orig_lattice::Lattice{T}

    # Scaling factor for lattice positions
    lattice_factor::T

    # External pressure (3x3 symmetric matrix)
    external_pressure::Matrix{T}

    # Workspace for deformation gradient
    deform_grad::Matrix{T}

    # Workspace for augmented positions
    augmented_positions::Matrix{T}

    # Workspace for augmented forces
    augmented_forces::Matrix{T}
end

"""
    CellRelaxationModifier(cell::Cell; external_pressure=zeros(3,3))

Create a cell relaxation modifier.

# Arguments
- `cell::Cell`: The cell to be relaxed
- `external_pressure=zeros(3,3)`: External pressure tensor (3×3 matrix)

# Notes
- The lattice factor is set to the number of atoms in the cell
- Workspaces are allocated based on cell dimensions
"""
function CellRelaxationModifier(cell::Cell; external_pressure=zeros(3, 3))
    T = eltype(cellmat(cell))
    #D = CellBase.n_dimensions(cell)
    D = 3 # Only three dimensions
    nat = length(cell)

    CellRelaxationModifier(
        Lattice(copy(cellmat(cell))),
        convert(T, nat),
        external_pressure,
        zeros(T, D, 3),
        zeros(T, D, nat + 3),  # nat atoms + 3 lattice vectors
        zeros(T, D, nat + 3),
    )
end

# This modifier changes position semantics
modifies_positions(::CellRelaxationModifier) = true
requires_forces(::CellRelaxationModifier) = true

"""
    deformgradient(mod::CellRelaxationModifier, cell::Cell) -> Matrix

Compute deformation gradient F such that F*C₀ = C_current.
"""
function deformgradient(mod::CellRelaxationModifier, cell::Cell)
    copy(transpose(transpose(cellmat(mod.orig_lattice)) \ transpose(cellmat(cell))))
end

"""
    pre_calculate!(mod::CellRelaxationModifier, cell::Cell)

Pre-calculation: compute and store deformation gradient.
"""
function pre_calculate!(mod::CellRelaxationModifier, cell::Cell)
    mod.deform_grad .= deformgradient(mod, cell)
    nothing
end

"""
    post_calculate!(mod::CellRelaxationModifier, cell, energy, forces, stress)

Post-calculation: transform forces/stress to virial form and apply external pressure.

The transformation:
1. Effective stress = stress - external_pressure
2. Virial = F⁻¹ * (vol * effective_stress)
3. Atomic forces transformed: F * forces
4. Virial forces for lattice: virial / lattice_factor
5. Augment: [atomic_forces, virial_forces]
6. Energy includes enthalpy term: E + P*V
"""
function post_calculate!(
    mod::CellRelaxationModifier,
    cell::Cell,
    energy::T,
    forces::Matrix{T},
    stress::Matrix{T}
) where {T}
    dgrad = mod.deform_grad
    vol = volume(cell)

    # Effective stress (subtract external pressure)
    eff_stress = stress .- mod.external_pressure

    # Virial from stress
    virial = (dgrad \ (vol .* eff_stress))

    # Transform atomic forces through deformation
    atomic_forces = copy(forces)
    atomic_forces[1:3, :] .= dgrad * atomic_forces[1:3, :]

    # Virial forces for lattice DOF
    virial_forces = zeros(T, size(forces, 1), 3)
    virial_forces[1:3, :] = virial ./ mod.lattice_factor

    # Augment forces with lattice contributions
    aug_forces = hcat(atomic_forces, virial_forces)

    # Compute enthalpy contribution from pressure (H = E + PV)
    # Use trace(P)/3 as average pressure
    pressure_contrib = vol * tr(mod.external_pressure) / 3

    return (energy + pressure_contrib, aug_forces, stress)
end

"""
    get_augmented_positions(mod::CellRelaxationModifier, cell::Cell) -> Matrix

Get augmented positions [atomic_positions, lattice_positions].

The lattice positions encode the deformation gradient scaled by lattice_factor.
"""
function get_augmented_positions(mod::CellRelaxationModifier, cell::Cell)
    dgrad = deformgradient(mod, cell)

    # Transform atomic positions
    apos = copy(positions(cell))
    apos[1:3, :] .= dgrad \ apos[1:3, :]

    # Lattice positions (deformation gradient transposed and scaled)
    lpos = mod.lattice_factor .* transpose(dgrad)

    # Pad for high-dimensional cells
    npad = size(apos, 1) - size(lpos, 1)
    if npad > 0
        lpos = vcat(lpos, zeros(eltype(lpos), npad, 3))
    end

    hcat(apos, lpos)
end

"""
    set_augmented_positions!(mod::CellRelaxationModifier, cell::Cell, new::Matrix)

Set positions from augmented array [atomic_positions, lattice_positions].

Extracts the deformation gradient from lattice positions and updates both
cell matrix and atomic positions.
"""
function set_augmented_positions!(mod::CellRelaxationModifier, cell::Cell, new::Matrix)
    nat = length(cell)

    # Extract atomic and lattice positions
    pos = new[:, 1:nat]
    new_dgrad = transpose(new[:, nat+1:end]) ./ mod.lattice_factor

    # Update cell matrix
    set_cellmat!(cell, new_dgrad[:, 1:3] * cellmat(mod.orig_lattice); scale_positions=false)

    # Transform and set atomic positions
    pos_new = copy(pos)
    pos_new[1:3, :] .= new_dgrad[:, 1:3] * pos_new[1:3, :]
    set_positions!(cell, pos_new)
end

# Display
function Base.show(io::IO, mod::CellRelaxationModifier)
    # Check if external pressure is diagonal
    P = mod.external_pressure
    if isdiag(P) && P[1, 1] ≈ P[2, 2] ≈ P[3, 3]
        print(
            io,
            "CellRelaxationModifier(P=$(P[1,1]), lattice_factor=$(mod.lattice_factor))"
        )
    else
        print(
            io,
            "CellRelaxationModifier(external_pressure=3×3 Matrix, lattice_factor=$(mod.lattice_factor))"
        )
    end
end
