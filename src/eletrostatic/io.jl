# Reading cell files


"""A castep cell file"""
struct CastepCell
    blocks::Dict
    keyvalues::Dict
end

"""A crystal structure"""
struct Structure
    lattice::Matrix{Float64}
    positions::Matrix{Float64}
    species::Vector{String}
    info::Dict
end

"""
    readcell(handle::IOStream)

Read the structure from a cell file
"""
function tokenise_cell(handle)
    # Storage space
    current_block = String[]
    keyvalues = Dict()
    blocks = Dict()
    block_name = ""
    for (number, line) in enumerate(handle) 
        line = String(strip(line))
        # Skip empty line
        isempty(line) && continue
        # Skip comment line
        (line[1] == '#') && continue
        # Whether in or not in the block
        if occursin(r"^%BLOCK", uppercase(line))
            block_name = lowercase(split(line)[2])
            empty!(current_block)
            continue
        end
        if occursin(r"^%ENDBLOCK", uppercase(line))
            @assert block_name == lowercase(split(line)[2]) 
            blocks[block_name] = copy(current_block)
            block_name = ""
            continue
        end

        # If in the block - add the line
        if !isempty(block_name)
            push!(current_block, line)
        else
            res = split(line, r"[ :=]+")
            key = res[1]
            if length(res) == 2
                value = res[2]
            else
                value = ""
            end
            blocks[lowercase(key)] = lowercase(value)
        end
    end
    return CastepCell(blocks, keyvalues)
end

"""
    structure(cell::CastepCell)

Parse the cell and obtain the structure
"""
function structure(cell::CastepCell)
    blocks = cell.blocks

    # Read the lattice
    lattice = zeros(Float64, (3,3))
    if "lattice_cart" in keys(blocks)
        for (i, line) in enumerate(blocks["lattice_cart"])
            lattice[:, i] = map(x->parse(Float64, x), split(line))
        end
    elseif "lattice_abc" in keys(blocks)
        data = blocks["lattice_abc"]
        abc = zeros(Float64, 6)
        abc[1:3] .= map(x->parse(Float64, x), split(data[1]))
        abc[4:6] .= map(x->parse(Float64, x), split(data[2]))
        lattice = cellpar2vec(abc)
    end

    # Read the positions
    if "positions_abs" in keys(blocks)
        data = blocks["positions_abs"]
        is_abs = true
    elseif  "positions_frac" in keys(blocks)
        data = blocks["positions_frac"]
        is_abs = false
    else
        throw(ErrorException("Unable to locate structures in $(keys(blocks))"))
    end

    nions = length(data)
    species = String[]
    positions = zeros(Float64, (3, nions))
    for (i, line) in enumerate(data)
        tokens = split(line)
        push!(species, String(tokens[1]))
        for idx in 1:3
            positions[idx, i] = parse(Float64, tokens[idx+1])
        end
    end

    # Convert to absolute coordinates
    if !is_abs
        positions = lattice * positions
    end
    return Structure(lattice, positions, species, cell.keyvalues)
end

"""Reading an cell structure"""
function read_cell(fname::AbstractString)
    open(fname) do file 
        cell = tokenise_cell(readlines(file))
        structure(cell)
    end
end

"""Compute energy for a single file"""
function energy(fname::AbstractString, z_dict; rd::Real=2.0, length_scale::Real)
    cell = read_cell(fname)
    energy(cell, z_dict; rd=rd, length_scale=length_scale)
end


"""
  energy(structure::Structure, z_dict::Dict, rc::Real, rd::Real)
"""
function energy(structure::Structure, z_dict::Dict, rc::Real, rd::Real)
    species = structure.species
    nions = length(species)
    chg = zeros(Float64, nions)
    for (i, value) in enumerate(species)
        chg[i] = z_dict[value]
    end
    energy(structure.lattice, structure.positions, chg, rc, rd)
end
"""
  energy(structure::Structure, z_dict::Dict, rc::Real; length_scale::Real)

Compute electrostatic energy of a structure with given cut-off and length_scale
"""
function energy(structure::Structure, z_dict::Dict;rd::Real=2.0, length_scale)
    rd_hat = rd / length_scale
    rc_hat = 3.0 * rd_hat * rd_hat
    rc = length_scale / rc_hat
    energy(structure, z_dict, rc, rd)
end