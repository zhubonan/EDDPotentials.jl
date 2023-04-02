
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

Rattle the cell shape based on random fractional changes on the cell parameters.
"""
function rattle_cell!(cell::Cell, amp)
    local new_cellpar
    i = 0
    while true
        new_cellpar = [x * (1 + rand() * amp) for x in cellpar(cell)]
        CellBase.isvalidcellpar(new_cellpar...) && break
        # Cannot found a valid cell parameters?
        if i > 10
            return cell
        end
        i += 1
    end
    new_lattice = Lattice(new_cellpar)
    spos = get_scaled_positions(cell)
    CellBase.set_cellmat!(cell, cellmat(new_lattice))
    positions(cell) .= cellmat(cell) * spos
    cell
end
