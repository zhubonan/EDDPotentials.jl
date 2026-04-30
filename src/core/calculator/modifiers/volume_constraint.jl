"""
    VolumeConstraintModifier{T}

Adds a harmonic penalty on cell volume to prevent collapse during hyperspace
relaxation. The constraint resists large volume changes while allowing gentle
relaxation.

# Mathematical Form
```math
E'   = E            + ½ k (V − V₀)² / V₀
σ'_ij = σ_ij        + k (V − V₀) / V₀ · δ_ij
```

The constraint stiffness `k` can be ramped down over time using the
scheduled penalty infrastructure (`set_modifier_parameter!`).

# Fields
- `vol0::T`: Initial volume V₀
- `k::T`: Constraint stiffness (typically 1–100 eV)
"""
mutable struct VolumeConstraintModifier{T} <: AbstractModifier
    vol0::T
    k::T
end

"""
    VolumeConstraintModifier(cell::Cell; k=10.0)

Create a volume constraint modifier with the given cell as reference.
"""
function VolumeConstraintModifier(cell::Cell; k=10.0)
    T = eltype(cellmat(cell))
    VolumeConstraintModifier{T}(CellBase.volume(cell), convert(T, k))
end

requires_forces(::VolumeConstraintModifier) = true

function post_calculate!(
    mod::VolumeConstraintModifier,
    cell::Cell,
    energy::T,
    forces::Matrix{T},
    stress::Matrix{T}
) where {T}
    V = CellBase.volume(cell)
    dV = V - mod.vol0

    if abs(dV) < eps(T)
        return (energy, forces, stress)
    end

    # Harmonic penalty: E_vol = ½ k (V−V₀)² / V₀
    penalty_energy = mod.k / 2 * dV^2 / mod.vol0

    # Hydrostatic stress: σ = k (V−V₀)/V₀ · I
    p = mod.k * dV / mod.vol0
    sig_correction = [p zero(T) zero(T); zero(T) p zero(T); zero(T) zero(T) p]

    new_stress = stress .+ sig_correction
    return (energy + penalty_energy, forces, new_stress)
end

function Base.show(io::IO, mod::VolumeConstraintModifier)
    print(io, "VolumeConstraintModifier(V₀=$(round(mod.vol0, digits=1)), k=$(mod.k))")
end
