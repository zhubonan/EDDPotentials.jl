
"""
    shake_res(files::Vector, nshake::Int, amp::Real)

Shake the given structures and write new files with suffix `-shake-N.res`.

"""
function shake_res(files::Vector, nshake::Int, amp::Real, cellamp::Real=0.02)
    for f in files
        cell = read_res(f)
        pos_backup = get_positions(cell)
        cellmat_backup = get_cellmat(cell)
        label = cell.metadata[:label]
        for i = 1:nshake
            rattle!(cell, amp)
            rattle_cell!(cell, cellamp)
            cell.metadata[:label] = label * "-shake-$i"
            write_res(splitext(f)[1] * "-shake-$i.res", cell)
            # Reset the original cellmatrix and positions
            set_cellmat!(cell, cellmat_backup)
            set_positions!(cell, pos_backup)
        end
    end
end

"""
    rattle_cell(cell::Cell, amp)

Rattle the cell shape by applying a random symmetric strain matrix.
"""
function rattle_cell!(cell::Cell{T}, amp) where {T}
    eps = (2 .* rand(6) .- 1) .* amp
    cb = cellmat(cell)
    deform = Matrix{T}(undef, 3, 3)
    deform[1, 1] = one(T) + T(eps[1])
    deform[2, 2] = one(T) + T(eps[2])
    deform[3, 3] = one(T) + T(eps[3])
    deform[1, 2] = T(eps[4]); deform[2, 1] = T(eps[4])
    deform[1, 3] = T(eps[5]); deform[3, 1] = T(eps[5])
    deform[2, 3] = T(eps[6]); deform[3, 2] = T(eps[6])
    set_cellmat!(cell, deform * cb; scale_positions=true)
    cell
end
