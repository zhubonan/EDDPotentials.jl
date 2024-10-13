#=
Extensions to the AtomsBase.jl

Contains commonly used functions for working with periodic systems.
=#
using LinearAlgebra
import AtomsBase: AbstractSystem, bounding_box


"""
Returns the cell matrix in the format of a 3x3 matrix made of COLUMN vectors.
"""
function cellmatrix(sys::AbstractSystem)
    hcat(bounding_box(sys)...)
end


"""
Returns the volume of the system
"""
function volume(sys::AbstractSystem)
    a, b, c = bounding_box(sys)
    abs(dot(a, cross(b, c)))
end


function set_cellmatrix(sys::AbstractSystem, vecs; scale_atoms=false)
    if scale_atoms
        spos = scaled_position(sys, :)
        cellmat = hcat(vecs...)
        new_pos = map(x -> cellmat * x, spos)
        new_sys = set_positions(sys, new_pos)
        AbstractSystem(new_sys; bounding_box=vecs)
    else
        AbstractSystem(sys; bounding_box=vecs)
    end
end


function set_cellmatrix(sys::AbstractSystem, mat::AbstractMatrix; scale_atoms=false)
    set_cellmatrix(sys, collect(eachcol(mat)), scale_atoms=scale_atoms)
end

function set_positions(sys::AbstractSystem, positions::AbstractVector)
    particles = [Atom(atom, position=pos) for (atom, pos) in zip(sys, positions)]
    AbstractSystem(sys; particles=particles)
end

set_positions(sys::AbstractSystem, positions::AbstractMatrix) =
    set_positions(sys, collect(eachcol(positions)))


function rec_bounding_box(vecs)
    eachcol(rec_cellmatrix(vecs))
end

function rec_cellmatrix(vecs)
    inv(ustrip.(hcat(vecs...))) .* (1 / unit(vecs[1][1]))
end

function scaled_position(system::AbstractSystem, idx)
    rec_matrix = rec_cellmatrix(bounding_box(system))
    map(x -> rec_matrix * x, AtomsBase.position(system, idx))
end

"""
    make_supercell(structure::AbstractSystem, a, b, c)

Make a supercell

Currently only work with diagonal transform matrices.
"""
function make_supercell(structure::AbstractSystem, tmat)

    old_cell = cellmatrix(structure)
    new_cell = old_cell * tmat

    # Compute the shift vectors
    svec = shift_vectors(bounding_box(structure), a - 1, b - 1, c - 1, 0, 0, 0)
    nshifts = length(svec)

    current_pos = AtomsBase.position(structure, :)
    ns = length(structure)
    # New positions
    new_pos = Vector{eltype(current_pos)}(undef, nshifts * ns)
    iorig = Vector{Int}(undef, nshifts * ns)
    for (i, shift) in enumerate(svec)
        for j = 1:ns
            idx = j + (i - 1) * ns  # New index
            new_pos[idx] = current_pos[j] + shift[n]
            iorig[idx] = j
        end
    end
    new_spec = repeat(AtomsBase.species(structure, :), nshifts)
    AtomsBase.periodic_system(
        [
            Atom(spec, pos; original_idx=i) for
            (spec, pos, i) in zip(new_spec, new_pos, iorig)
        ],
        collect(eachcol(new_cell)),
    )
end

function make_supercell(structure::AbstractSystem, a, b, c)
    tmat = [
        a 0 0
        0 b 0
        0 0 c
    ]
    make_supercell(structure, tmat)
end
