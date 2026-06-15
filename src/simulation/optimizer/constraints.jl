using LinearAlgebra

"""
    AbstractStepConstraint

Constraint applied directly to an optimizer step vector. These constraints
project the proposed displacement, unlike calculator modifiers that change the
energy/force model.
"""
abstract type AbstractStepConstraint end

"""
Linear selective-dynamics constraints use a Lagrange multiplier projection.
For an ionic displacement vector `x` and linear constraints `A * x = 0`, the
closest admissible displacement is

```
x_proj = x - A' * λ,    (A * A') * λ = A * x
```

implemented with `pinv(A * A')` so redundant constraint rows are allowed. The
same projection is applied to forces before convergence checks, so constrained
degrees of freedom can carry raw forces without preventing convergence.

Cartesian constraints act directly on `vec(positions)`, where this codebase's
`D x N` storage gives component order `[x1, y1, z1, x2, y2, z2, ...]`. Scaled
constraints first map displacements by `ds = cellmat \\ dr`, project in `s`,
then map back with `dr = cellmat * ds`. Forces are covectors, so scaled force
projection uses the dual transform `q = cellmat' * F` and maps back with
`F = cellmat' \\ q`, matching ASE's `FixScaled` convention after translating
from ASE's row-major `N x 3` storage.

Cell constraints use the same projection idea on the appended cell block of the
optimizer vector. For variable-cell native optimizers this block is a
dimensionless symmetric strain increment, so `FixCellConstraint` masks Voigt
components in `(xx, yy, zz, yz, xz, xy)` order directly. Stress is projected
with the same Voigt mask for optimizer gradients, convergence, and logging
only; `eddp_stress(calc)` remains the raw calculator stress.
"""
function _constraint_math_note end

const _DIRECT_COORDINATES = (:direct, :scaled, :fractional)

_collect_indices(index::Integer) = [Int(index)]
_collect_indices(indices) = collect(Int, indices)

_normalize_mask(mask) = begin
    values = Tuple(Bool.(collect(mask)))
    length(values) == 3 ||
        throw(ArgumentError("Constraint masks must have three entries, got $(length(values))"))
    values
end

_normalize_voigt_mask(mask) = begin
    values = Tuple(Bool.(collect(mask)))
    length(values) == 6 ||
        throw(ArgumentError("Cell constraint masks must have six Voigt entries, got $(length(values))"))
    values
end

_normalize_coordinate(coordinate::Symbol) =
    coordinate in _DIRECT_COORDINATES ? :direct :
    coordinate === :cartesian ? :cartesian :
    throw(ArgumentError("coordinate must be :cartesian, :direct, :scaled, or :fractional"))

function _collect_pairs(pair::Tuple{<:Integer,<:Integer})
    [(Int(pair[1]), Int(pair[2]))]
end

function _collect_pairs(pairs)
    [(Int(pair[1]), Int(pair[2])) for pair in pairs]
end

function _check_atom(atom::Integer, nat::Integer)
    1 <= atom <= nat || throw(BoundsError(1:nat, atom))
end

function _check_dim(dim::Integer, D::Integer)
    1 <= dim <= D || throw(BoundsError(1:D, dim))
end

_col(D::Integer, atom::Integer, dim::Integer) = (atom - 1) * D + dim

function _mask_dims(mask, D::Integer; direct::Bool=false)
    maxdim = direct ? 3 : D
    direct && D < 3 && throw(ArgumentError("Direct-coordinate constraints require at least 3 dimensions"))
    dims = Int[]
    for dim in 1:min(3, maxdim)
        mask[dim] && push!(dims, dim)
    end
    dims
end

function _cell3(calc::AbstractCalc)
    get_cellmat(calc)[1:3, 1:3]
end

function _ionic_matrix(v::AbstractVector, D::Integer, nat::Integer)
    reshape(view(v, 1:(D * nat)), D, nat)
end

function _step_to_coordinate(calc::AbstractCalc, v::AbstractVector, coordinate::Symbol)
    pos = get_positions(calc)
    D, nat = size(pos)
    out = copy(_ionic_matrix(v, D, nat))
    if _normalize_coordinate(coordinate) === :direct
        D >= 3 || throw(ArgumentError("Direct-coordinate constraints require at least 3 dimensions"))
        out[1:3, :] .= _cell3(calc) \ out[1:3, :]
    end
    out
end

function _coordinate_to_step!(calc::AbstractCalc, v::AbstractVector, coord::AbstractMatrix, coordinate::Symbol)
    pos = get_positions(calc)
    D, nat = size(pos)
    out = _ionic_matrix(v, D, nat)
    if _normalize_coordinate(coordinate) === :direct
        D >= 3 || throw(ArgumentError("Direct-coordinate constraints require at least 3 dimensions"))
        out[1:3, :] .= _cell3(calc) * coord[1:3, :]
        if D > 3
            out[4:D, :] .= coord[4:D, :]
        end
    else
        out .= coord
    end
    v
end

function _force_to_coordinate(calc::AbstractCalc, F::AbstractMatrix, coordinate::Symbol)
    out = copy(F)
    if _normalize_coordinate(coordinate) === :direct
        size(F, 1) >= 3 || throw(ArgumentError("Direct-coordinate constraints require at least 3 dimensions"))
        out[1:3, :] .= transpose(_cell3(calc)) * F[1:3, :]
    end
    out
end

function _coordinate_to_force!(calc::AbstractCalc, F::AbstractMatrix, coord::AbstractMatrix, coordinate::Symbol)
    if _normalize_coordinate(coordinate) === :direct
        size(F, 1) >= 3 || throw(ArgumentError("Direct-coordinate constraints require at least 3 dimensions"))
        F[1:3, :] .= transpose(_cell3(calc)) \ coord[1:3, :]
        if size(F, 1) > 3
            F[4:end, :] .= coord[4:end, :]
        end
    else
        F .= coord
    end
    F
end

function _project_linear!(x::AbstractVector, A::AbstractMatrix)
    size(A, 1) == 0 && return x
    size(A, 2) == length(x) ||
        throw(DimensionMismatch("Constraint matrix has $(size(A, 2)) columns for vector length $(length(x))"))
    residual = A * x
    x .-= transpose(A) * (pinv(A * transpose(A)) * residual)
    x
end

_strain_to_voigt(ε::AbstractMatrix) = [
    ε[1, 1],
    ε[2, 2],
    ε[3, 3],
    ε[2, 3],
    ε[1, 3],
    ε[1, 2],
]

function _voigt_to_strain(v::AbstractVector)
    [
        v[1] v[6] v[5]
        v[6] v[2] v[4]
        v[5] v[4] v[3]
    ]
end

"""
    ZeroTranslationConstraint()

Remove the mean Cartesian displacement from all atoms in an optimizer step.
"""
struct ZeroTranslationConstraint <: AbstractStepConstraint end

"""
    FixAtoms(indices)

Freeze all Cartesian degrees of freedom for the atoms listed in `indices`.
"""
struct FixAtoms <: AbstractStepConstraint
    atoms::Vector{Int}
    FixAtoms(atoms) = new(_collect_indices(atoms))
end

const IonicConstraint = FixAtoms

"""
    FixCartesian(indices; mask=(true, true, true))

Freeze selected Cartesian components for the atoms listed in `indices`.
`mask=true` means the component is fixed, matching ASE convention.
"""
struct FixCartesian <: AbstractStepConstraint
    atoms::Vector{Int}
    mask::NTuple{3,Bool}
    FixCartesian(atoms; mask=(true, true, true)) = new(_collect_indices(atoms), _normalize_mask(mask))
end

"""
    FixScaled(indices; mask=(true, true, true))

Freeze selected fractional/direct components for the atoms listed in `indices`.
When the cell relaxes, constrained fractional components follow the lattice.
"""
struct FixScaled <: AbstractStepConstraint
    atoms::Vector{Int}
    mask::NTuple{3,Bool}
    FixScaled(atoms; mask=(true, true, true)) = new(_collect_indices(atoms), _normalize_mask(mask))
end

"""
    FixRelative(pairs; mask=(true, true, true), coordinate=:cartesian)

Preserve component-wise relative displacements for each atom pair. Use
`coordinate=:direct` (or `:scaled`) to preserve relative fractional components.
"""
struct FixRelative <: AbstractStepConstraint
    pairs::Vector{Tuple{Int,Int}}
    mask::NTuple{3,Bool}
    coordinate::Symbol
    function FixRelative(pairs; mask=(true, true, true), coordinate=:cartesian)
        new(_collect_pairs(pairs), _normalize_mask(mask), _normalize_coordinate(coordinate))
    end
end

"""
    FixRigidGroup(indices; mask=(true, true, true), coordinate=:cartesian)

Constrain selected atoms to share one translational displacement, preserving
their internal relative coordinates in the selected components.
"""
struct FixRigidGroup <: AbstractStepConstraint
    atoms::Vector{Int}
    mask::NTuple{3,Bool}
    coordinate::Symbol
    function FixRigidGroup(atoms; mask=(true, true, true), coordinate=:cartesian)
        new(_collect_indices(atoms), _normalize_mask(mask), _normalize_coordinate(coordinate))
    end
end

"""
    FixLinearIonic(A; coordinate=:cartesian)

Advanced linear ionic constraint. The rows of `A` define equality constraints
`A * dx = 0` in Cartesian or direct-coordinate ionic displacement space.
"""
struct FixLinearIonic <: AbstractStepConstraint
    A::Matrix{Float64}
    coordinate::Symbol
    FixLinearIonic(A; coordinate=:cartesian) = new(Matrix{Float64}(A), _normalize_coordinate(coordinate))
end

"""
    FixCellConstraint(; mask=(true, true, true, true, true, true))

Fix selected cell degrees of freedom in Voigt strain order
`(xx, yy, zz, yz, xz, xy)`. `mask=true` means fixed. The default
fixes the full cell, preserving the historical all-or-nothing behavior.
"""
struct FixCellConstraint <: AbstractStepConstraint
    mask::NTuple{6,Bool}
    FixCellConstraint(; mask=(true, true, true, true, true, true)) = new(_normalize_voigt_mask(mask))
end

const FixCell = FixCellConstraint

"""
    FixVolumeConstraint()

Project the cell part of a proposed optimizer step onto trace-free strain, so
the accepted infinitesimal cell update preserves volume to first order.
"""
struct FixVolumeConstraint <: AbstractStepConstraint end

"""
    SymmetryConstraint(cell; symprec=1e-5)

Project optimizer vectors onto the symmetry detected for `cell`. The constraint
is inactive when no nontrivial symmetry is found or atom mappings cannot be
constructed reliably.
"""
struct SymmetryConstraint <: AbstractStepConstraint
    rotations_frac::Array{Float64,3}
    translations_frac::Matrix{Float64}
    mappings::Matrix{Int}
    active::Bool
    symprec::Float64
end

function SymmetryConstraint(cell; symprec::Real=1e-5)
    constraint = _build_symmetry_constraint(cell, Float64(symprec))
    constraint === nothing && return SymmetryConstraint(
        zeros(Float64, 3, 3, 0),
        zeros(Float64, 3, 0),
        zeros(Int, 0, 0),
        false,
        Float64(symprec),
    )
    constraint
end

Base.isempty(constraint::SymmetryConstraint) = !constraint.active

function _wrap_fractional(v::AbstractVector)
    out = collect(Float64, v)
    out .-= floor.(out)
    out
end

function _fractional_distance(a::AbstractVector, b::AbstractVector, H::AbstractMatrix)
    d = collect(Float64, a .- b)
    d .-= round.(d)
    norm(H * d)
end

function _build_symmetry_constraint(cell, symprec::Float64)
    local rotations_raw, translations_raw
    try
        rotations_raw, translations_raw = CellBase.get_symmetry(cell, symprec)
    catch err
        @debug "Symmetry detection failed" exception=(err, catch_backtrace())
        return nothing
    end

    nsym = size(rotations_raw, 3)
    nsym <= 1 && return nothing

    rotations = Array{Float64,3}(undef, 3, 3, nsym)
    for isym in 1:nsym
        # Spglib/CellBase rotations are returned in the row-vector convention;
        # the optimizer stores positions as column vectors, s' = R * s + t.
        rotations[:, :, isym] .= transpose(Float64.(rotations_raw[:, :, isym]))
    end
    translations = Matrix{Float64}(translations_raw)

    mappings = _symmetry_atom_mappings(cell, rotations, translations, symprec)
    mappings === nothing && return nothing

    eye = Matrix{Float64}(I, 3, 3)
    nontrivial = any(isym -> rotations[:, :, isym] != eye ||
        any(abs.(translations[:, isym]) .> symprec), 1:nsym)
    nontrivial || return nothing

    SymmetryConstraint(rotations, translations, mappings, true, symprec)
end

function _symmetry_atom_mappings(cell, rotations::Array{Float64,3}, translations::Matrix{Float64}, symprec::Float64)
    H = CellBase.cellmat(cell)[1:3, 1:3]
    spos = CellBase.get_scaled_positions(cell)[1:3, :]
    symbols = species(cell)
    nat = size(spos, 2)
    nsym = size(rotations, 3)
    mappings = zeros(Int, nsym, nat)

    for isym in 1:nsym
        R = rotations[:, :, isym]
        t = translations[:, isym]
        used_targets = falses(nat)
        for source in 1:nat
            target_pos = _wrap_fractional(R * spos[:, source] .+ t)
            best = 0
            best_dist = Inf
            for target in 1:nat
                used_targets[target] && continue
                symbols[target] == symbols[source] || continue
                dist = _fractional_distance(target_pos, spos[:, target], H)
                if dist < best_dist
                    best = target
                    best_dist = dist
                end
            end
            if best == 0 || best_dist > max(symprec, 1e-8)
                @debug "Failed to map atom under symmetry operation" operation=isym atom=source distance=best_dist
                return nothing
            end
            mappings[isym, best] = source
            used_targets[best] = true
        end
    end

    any(==(0), mappings) && return nothing
    mappings
end

function _cartesian_rotations(constraint::SymmetryConstraint, calc::AbstractCalc)
    H = _cell3(calc)
    invH = inv(H)
    nsym = size(constraint.rotations_frac, 3)
    [H * constraint.rotations_frac[:, :, isym] * invH for isym in 1:nsym]
end

function _project_symmetry_vectors!(constraint::SymmetryConstraint, calc::AbstractCalc, V::AbstractMatrix)
    constraint.active || return V
    D, nat = size(V)
    D >= 3 || return V
    nat == size(constraint.mappings, 2) ||
        throw(DimensionMismatch("SymmetryConstraint has mappings for $(size(constraint.mappings, 2)) atoms but got $nat"))

    rotations_cart = _cartesian_rotations(constraint, calc)
    work = copy(V)
    fill!(V, 0.0)
    nsym = length(rotations_cart)
    for atom in 1:nat
        for isym in 1:nsym
            source = constraint.mappings[isym, atom]
            V[1:3, atom] .+= rotations_cart[isym] * work[1:3, source]
        end
        V[1:3, atom] ./= nsym
        if D > 3
            V[4:D, atom] .= work[4:D, atom]
        end
    end
    V
end

function _project_symmetry_matrix!(constraint::SymmetryConstraint, calc::AbstractCalc, S::AbstractMatrix)
    constraint.active || return S
    rotations_cart = _cartesian_rotations(constraint, calc)
    work = copy(S)
    fill!(S, 0.0)
    for R in rotations_cart
        S .+= R * work * transpose(R)
    end
    S ./= length(rotations_cart)
    S
end

function apply_step_constraints!(constraints, opt::AbstractOptimizer, calc::AbstractCalc, dx::AbstractVector)
    isempty(constraints) && return dx
    for constraint in constraints
        constrain_step!(constraint, opt, calc, dx)
    end
    return dx
end

function apply_force_constraints!(constraints, opt::AbstractOptimizer, calc::AbstractCalc, F::AbstractMatrix)
    isempty(constraints) && return F
    for constraint in constraints
        constrain_force!(constraint, opt, calc, F)
    end
    return F
end

function apply_stress_constraints!(constraints, opt::AbstractOptimizer, calc::AbstractCalc, S::AbstractMatrix)
    isempty(constraints) && return S
    for constraint in constraints
        constrain_stress!(constraint, opt, calc, S)
    end
    return S
end

function _constraint_matrix(constraint::FixAtoms, D::Integer, nat::Integer)
    rows = Matrix{Float64}(undef, length(constraint.atoms) * D, D * nat)
    fill!(rows, 0.0)
    row = 0
    for atom in constraint.atoms
        _check_atom(atom, nat)
        for dim in 1:D
            row += 1
            rows[row, _col(D, atom, dim)] = 1.0
        end
    end
    rows
end

function _constraint_matrix(constraint::FixCartesian, D::Integer, nat::Integer)
    dims = _mask_dims(constraint.mask, D)
    rows = zeros(Float64, length(constraint.atoms) * length(dims), D * nat)
    row = 0
    for atom in constraint.atoms
        _check_atom(atom, nat)
        for dim in dims
            row += 1
            rows[row, _col(D, atom, dim)] = 1.0
        end
    end
    rows
end

function _constraint_matrix(constraint::FixScaled, D::Integer, nat::Integer)
    dims = _mask_dims(constraint.mask, D; direct=true)
    rows = zeros(Float64, length(constraint.atoms) * length(dims), D * nat)
    row = 0
    for atom in constraint.atoms
        _check_atom(atom, nat)
        for dim in dims
            row += 1
            rows[row, _col(D, atom, dim)] = 1.0
        end
    end
    rows
end

function _constraint_matrix(constraint::FixRelative, D::Integer, nat::Integer)
    dims = _mask_dims(constraint.mask, D; direct=constraint.coordinate === :direct)
    rows = zeros(Float64, length(constraint.pairs) * length(dims), D * nat)
    row = 0
    for (a, b) in constraint.pairs
        _check_atom(a, nat)
        _check_atom(b, nat)
        for dim in dims
            row += 1
            rows[row, _col(D, a, dim)] = 1.0
            rows[row, _col(D, b, dim)] = -1.0
        end
    end
    rows
end

function _constraint_matrix(constraint::FixRigidGroup, D::Integer, nat::Integer)
    length(constraint.atoms) <= 1 && return zeros(Float64, 0, D * nat)
    dims = _mask_dims(constraint.mask, D; direct=constraint.coordinate === :direct)
    rows = zeros(Float64, (length(constraint.atoms) - 1) * length(dims), D * nat)
    anchor = constraint.atoms[1]
    _check_atom(anchor, nat)
    row = 0
    for atom in constraint.atoms[2:end]
        _check_atom(atom, nat)
        for dim in dims
            row += 1
            rows[row, _col(D, atom, dim)] = 1.0
            rows[row, _col(D, anchor, dim)] = -1.0
        end
    end
    rows
end

function _constraint_matrix(constraint::FixLinearIonic, D::Integer, nat::Integer)
    if constraint.coordinate === :direct && D != 3
        throw(ArgumentError("FixLinearIonic with coordinate=:direct requires a 3D cell"))
    end
    size(constraint.A, 2) == D * nat ||
        throw(DimensionMismatch("FixLinearIonic has $(size(constraint.A, 2)) columns for $(D * nat) ionic degrees of freedom"))
    constraint.A
end

function _project_coordinate_step!(constraint, opt::AbstractOptimizer, calc::AbstractCalc, dx::AbstractVector, coordinate::Symbol)
    pos = get_positions(calc)
    D, nat = size(pos)
    coord = _step_to_coordinate(calc, dx, coordinate)
    A = _constraint_matrix(constraint, D, nat)
    _project_linear!(vec(coord), A)
    _coordinate_to_step!(calc, dx, coord, coordinate)
end

function _project_coordinate_force!(constraint, opt::AbstractOptimizer, calc::AbstractCalc, F::AbstractMatrix, coordinate::Symbol)
    D, nat = size(F)
    coord = _force_to_coordinate(calc, F, coordinate)
    A = _constraint_matrix(constraint, D, nat)
    _project_linear!(vec(coord), A)
    _coordinate_to_force!(calc, F, coord, coordinate)
end

function constrain_step!(::ZeroTranslationConstraint, opt::AbstractOptimizer, calc::AbstractCalc, dx::AbstractVector)
    pos = get_positions(calc)
    D, nat = size(pos)
    nat == 0 && return dx
    ionic = reshape(view(dx, 1:(D * nat)), D, nat)
    shift = vec(sum(ionic; dims=2)) ./ nat
    for i in 1:nat
        @views ionic[:, i] .-= shift
    end
    return dx
end

constrain_step!(constraint::FixAtoms, opt::AbstractOptimizer, calc::AbstractCalc, dx::AbstractVector) =
    _project_coordinate_step!(constraint, opt, calc, dx, :cartesian)

constrain_step!(constraint::FixCartesian, opt::AbstractOptimizer, calc::AbstractCalc, dx::AbstractVector) =
    _project_coordinate_step!(constraint, opt, calc, dx, :cartesian)

constrain_step!(constraint::FixScaled, opt::AbstractOptimizer, calc::AbstractCalc, dx::AbstractVector) =
    _project_coordinate_step!(constraint, opt, calc, dx, :direct)

constrain_step!(constraint::FixRelative, opt::AbstractOptimizer, calc::AbstractCalc, dx::AbstractVector) =
    _project_coordinate_step!(constraint, opt, calc, dx, constraint.coordinate)

constrain_step!(constraint::FixRigidGroup, opt::AbstractOptimizer, calc::AbstractCalc, dx::AbstractVector) =
    _project_coordinate_step!(constraint, opt, calc, dx, constraint.coordinate)

constrain_step!(constraint::FixLinearIonic, opt::AbstractOptimizer, calc::AbstractCalc, dx::AbstractVector) =
    _project_coordinate_step!(constraint, opt, calc, dx, constraint.coordinate)

function constrain_step!(constraint::FixCellConstraint, opt::AbstractOptimizer, calc::AbstractCalc, dx::AbstractVector)
    opt.relax_cell || return dx
    pos = get_positions(calc)
    D, nat = size(pos)
    cell_start = D * nat + 1
    cell_step = reshape(view(dx, cell_start:length(dx)), 3, 3)

    strain = 0.5 .* (cell_step .+ transpose(cell_step))
    voigt = _strain_to_voigt(strain)
    for i in 1:6
        constraint.mask[i] && (voigt[i] = 0.0)
    end
    cell_step .= _voigt_to_strain(voigt)
    return dx
end

function constrain_step!(::FixVolumeConstraint, opt::AbstractOptimizer, calc::AbstractCalc, dx::AbstractVector)
    opt.relax_cell || return dx

    pos = get_positions(calc)
    D, nat = size(pos)
    cell_start = D * nat + 1
    cell_step = reshape(view(dx, cell_start:length(dx)), 3, 3)
    strain = 0.5 .* (cell_step .+ transpose(cell_step))
    strain .-= (tr(strain) / 3.0) .* Matrix{Float64}(I, 3, 3)
    cell_step .= strain
    return dx
end

function constrain_step!(constraint::SymmetryConstraint, opt::AbstractOptimizer, calc::AbstractCalc, dx::AbstractVector)
    constraint.active || return dx
    pos = get_positions(calc)
    D, nat = size(pos)
    ionic = reshape(view(dx, 1:(D * nat)), D, nat)
    _project_symmetry_vectors!(constraint, calc, ionic)
    if opt.relax_cell
        cell_start = D * nat + 1
        cell_step = reshape(view(dx, cell_start:length(dx)), 3, 3)
        strain = 0.5 .* (cell_step .+ transpose(cell_step))
        _project_symmetry_matrix!(constraint, calc, strain)
        cell_step .= 0.5 .* (strain .+ transpose(strain))
    end
    return dx
end

function constrain_step!(constraint::AbstractStepConstraint, opt::AbstractOptimizer, calc::AbstractCalc, dx::AbstractVector)
    throw(MethodError(constrain_step!, (constraint, opt, calc, dx)))
end

function constrain_force!(::ZeroTranslationConstraint, opt::AbstractOptimizer, calc::AbstractCalc, F::AbstractMatrix)
    nat = size(F, 2)
    nat == 0 && return F
    shift = vec(sum(F; dims=2)) ./ nat
    for i in 1:nat
        @views F[:, i] .-= shift
    end
    return F
end

constrain_force!(constraint::FixAtoms, opt::AbstractOptimizer, calc::AbstractCalc, F::AbstractMatrix) =
    _project_coordinate_force!(constraint, opt, calc, F, :cartesian)

constrain_force!(constraint::FixCartesian, opt::AbstractOptimizer, calc::AbstractCalc, F::AbstractMatrix) =
    _project_coordinate_force!(constraint, opt, calc, F, :cartesian)

constrain_force!(constraint::FixScaled, opt::AbstractOptimizer, calc::AbstractCalc, F::AbstractMatrix) =
    _project_coordinate_force!(constraint, opt, calc, F, :direct)

constrain_force!(constraint::FixRelative, opt::AbstractOptimizer, calc::AbstractCalc, F::AbstractMatrix) =
    _project_coordinate_force!(constraint, opt, calc, F, constraint.coordinate)

constrain_force!(constraint::FixRigidGroup, opt::AbstractOptimizer, calc::AbstractCalc, F::AbstractMatrix) =
    _project_coordinate_force!(constraint, opt, calc, F, constraint.coordinate)

constrain_force!(constraint::FixLinearIonic, opt::AbstractOptimizer, calc::AbstractCalc, F::AbstractMatrix) =
    _project_coordinate_force!(constraint, opt, calc, F, constraint.coordinate)

function constrain_force!(constraint::SymmetryConstraint, opt::AbstractOptimizer, calc::AbstractCalc, F::AbstractMatrix)
    _project_symmetry_vectors!(constraint, calc, F)
end

constrain_force!(constraint::AbstractStepConstraint, opt::AbstractOptimizer, calc::AbstractCalc, F::AbstractMatrix) = F

function constrain_stress!(constraint::FixCellConstraint, opt::AbstractOptimizer, calc::AbstractCalc, S::AbstractMatrix)
    opt.relax_cell || return S
    voigt = _strain_to_voigt(0.5 .* (S .+ transpose(S)))
    for i in 1:6
        constraint.mask[i] && (voigt[i] = 0.0)
    end
    S .= _voigt_to_strain(voigt)
    S
end

function constrain_stress!(constraint::SymmetryConstraint, opt::AbstractOptimizer, calc::AbstractCalc, S::AbstractMatrix)
    opt.relax_cell || return S
    _project_symmetry_matrix!(constraint, calc, S)
end

constrain_stress!(constraint::AbstractStepConstraint, opt::AbstractOptimizer, calc::AbstractCalc, S::AbstractMatrix) = S

function constrain_cartesian_position_step!(constraint::SymmetryConstraint, opt::AbstractOptimizer, calc::AbstractCalc, dx::AbstractVector)
    constraint.active || return dx
    opt.relax_cell && return dx
    pos = get_positions(calc)
    D, nat = size(pos)
    ionic = reshape(view(dx, 1:(D * nat)), D, nat)
    _project_symmetry_vectors!(constraint, calc, ionic)
    dx
end

constrain_cartesian_position_step!(constraint, opt::AbstractOptimizer, calc::AbstractCalc, dx::AbstractVector) =
    constrain_step!(constraint, opt, calc, dx)

function apply_cartesian_position_constraints!(constraints, opt::AbstractOptimizer, calc::AbstractCalc, pos_old::AbstractMatrix, pos_new::AbstractMatrix)
    isempty(constraints) && return pos_new
    D, nat = size(pos_old)
    dx = vec(copy(pos_new .- pos_old))
    for constraint in constraints
        if constraint isa FixAtoms || constraint isa FixCartesian ||
                (constraint isa FixRelative && constraint.coordinate === :cartesian) ||
                (constraint isa FixRigidGroup && constraint.coordinate === :cartesian) ||
                (constraint isa FixLinearIonic && constraint.coordinate === :cartesian) ||
                constraint isa SymmetryConstraint
            constrain_cartesian_position_step!(constraint, opt, calc, dx)
        end
    end
    pos_new .= pos_old .+ reshape(dx, D, nat)
    pos_new
end
