#= 
Building structures
=#
import Base: rand

"""
Return a given structure with positions randomised
"""
function random_pos(base_structure::Cell)
    new_structure = deepcopy(base_structure)
    random_pos!(new_structure)
end

function random_pos!(base_structure::Cell)
    for site in base_structure.sites
        disp = random_vec(base_structure.lattice)
        displace!(site, disp)
        wrap!(site, base_structure.lattice)
    end
    base_structure
end


"""Generate a random lattice for a given volume"""
function random_cellpar(targvol)
    lowerbound = 30.
    ok = false
    a, b, c, α, β, γ = 0., 0., 0., 0., 0., 0.
    # Try until success
    while ok != true
        a, b, c = randf(), randf(), randf()
        α, β, γ = rand_range(lowerbound, 120.), rand_range(lowerbound, 120.), rand_range(lowerbound, 120.)
        ok = isvalidcellpar(a, b, c, α, β, γ)
    end
    # Scale for volume
    vol = volume(a, b, c, α, β, γ)
    scale = (targvol / vol) ^ (1 /3)
    return a * scale, b * scale, c * scale, α, β, γ
end

"Construct a random Lattice"
function random_lattice(targvol)
    cellpar = random_cellpar(targvol)
    Lattice(cellpar...)
end

"""
random_build_simple(volume, species, minimum_separations)

Simple routine for building random structures for a given volume and species and minimum separations
"""
function random_build_simple(targvol, species, minsep)
    itermax = 1000
    i = 1
    while i < itermax
        # Build and random lattice
        cellpar = random_cellpar(targvol)
        lattice = Lattice(cellpar...)

        # Assemble the structure
        ni = length(species)
        sites = [Site(random_vec(lattice), i, sn) for (i, sn) in enumerate(species)]
        structure = Cell(lattice, sites)

        # Ensure that the minimum separations are satified
        check_minsep(structure, minsep ) && return structure
        i += 1
    end
end


function random_abc!(cellpar, abc_max=2.0, cons_len=[1, 2, 3])
    abc = [1.0, randf() * abc_max, randf() * abc_max]
    abc = abc[randperm(3)]
    constrain!(abc, cons_len)
    cellpar[1:3] .= abc
end


"""
    function random_αβγ!(cellpar, ang_min, ang_max, cang=[1., 2., 3.])

Generate the random α β γ part, taking account of maximum/minimum angle values and 
angle constraints as a result of selecting particular space groups.

"""
function random_αβγ!(cellpar, ang_min, ang_max, cang=[0., 0., 0.], skew=0.2)
    @label begining
    for i = 1:3
        if cang[i] <= 0.0
            cellpar[i + 3] = rand_range(ang_min, ang_max)
        elseif cang[i] == 999.0 # Specal case for Fmm2 and Fdd2
            diagonal 
            temp = cellpar[i] * (sum(cellpar[1:3] .^ 2) - 2.0 * cellpar[i]^2) / 2.0 / (prod(cellpar[1:3]))
            abs(temp) > 1.0 && @goto(begining)
            angle = acos(temp) / dgrd
            cellpar[3 + i] = angle
        else
            # Angle given by the space group
            cellpar[3 + 1] = cang[i]
        end
    end

    # Handling for case 9999.0
    if any(cang == 9999.0)
        _handle_cang_9999!(cellpar, ang_min, ang_max, 9999)
    end

    # Handling for case 99999.0
    if any(cang == 99999.0)
        _handle_cang_9999!(cellpar, ang_min, ang_max, 99999)
    end
    
    # Apply constraint condition in lattice angles
    constrain!(cellpar, cang)
    isvalidcellpar(cellpar...;degree=true) || @goto(begining)

end 

"""
    _handle_cang_9999!(lattice_abc, ang_min, ang_max, case)

Handle constrained cases of 9999. or 99999. to makes sure the lattice is consistent with the 
"""
function _handle_cang_9999!(lattice_abc::Vector, ang_min::Number, ang_max::Number, case::Int)
    idxa, idxb, idxc = 4, 6, 5
    if case === 99999
        idxa, idxb, idxc = 4, 5, 6
    end
    angle = 1.0e10
    while (angle < ang_min) | (angle > ang_max)
        temp = 9999.0
        while abs(temp) > 1.0
            rn = rand3f(ang_min, ang_max)
            lattice_abc[idxa] = rn[1]
            if case == 99999
                lattice_abc[idxb] = rn[1]
            else
                lattice_abc[idxb] = rn[2]
            end
            temp = -(cos(lattice_abc[idxa] * dgrd) + cos(lattice_abc[idxb] * dgrd) + 1) 
        end 
        angle = acos(temp) / dgrd
    end
    lattice_abc[idxc] = angle
    return lattice_abc
end


"Construct a random lattice within certain constraints"
function random_lattice(targvol, cons_len, cons_ang;angmax=120., angmin=30., abc_max=2.0, skew=0.2)
    cellpar = zeros(6)

    @label begining

    # Obtain the abc part
    random_abc!(cellpar, abc_max, cons_len)
    # Generate the cell angles
    random_αβγ!(cellpar, angmin, angmax, cons_ang)


    # Check if the cell is too flat
    α, β, γ = cellpar[4:6] .* dgrd
    # The volume factor -> 1 if all angles are 90 degree
    vfactor = 1 + 2 * cos(α) * cos(β) * cos(γ) - cos(α)^2 - cos(β)^2 - cos(γ)^2
    sqrt(vfactor) < skew && @goto(begining)

    scale = (targvol / volume(cellpar...)) ^ (1/3) 
    cellpar[1:3] .*= scale

    Lattice(cellpar)
end

"""
    constrain!(vec::Vector{T}, spec::Vector, offset::Int=0)

Constraint a vector using a specification vector where 
minus values mean the same (or offseted) position in the vector should be constrained to be the same.
"""
function constrain!(vec::Vector{T}, spec::Vector, offset::Int=0) where T
    temp::T= 0.0
    for i = 1:3
        if spec[i] < 0.0 
            temp = vec[i + offset]
            break
        end
    end
    for i = 1:3
        if spec[i] < 0.0; vec[i + offset] = temp; end
    end
    return vec
end