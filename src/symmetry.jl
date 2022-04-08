#= 
For generating structures with symmetry
=#
using SymmetryData

"""
Choose multiplicity for a given symmetry. By default try to maximize the general positions.
    Example: 4 ion 8 operations -> 4 0 0 0
"""
function choose_multi(nions::Int, nsymm::Int, adjgen=0)
    # Here we explicitly write out the possible multiplicities for given 
    # number of symmetry operations
    # Multiplicity factor for given number of symmetry
    cases = Dict([[1, [1]], 
             [2, [1, 2]],
             [3, [1, 3]],
             [4, [1, 2, 4]],
             [5, [1, 5]],
             [6, [1, 2, 3, 6]],
             [7, [1, 7]],
             [8, [1, 2, 4, 8]],
             [9, [1, 3, 9]],
             [10, [1, 2, 5, 10]],
             [11, [1, 11]],
             [12, [1, 2, 3, 4, 6, 12]],
             [16, [1, 2, 4, 8, 16]],
             [20, [1, 2, 4, 5, 10, 20]],
             [24, [1, 2, 3, 4, 6, 8, 12, 24]],
             [48, [1, 2, 3, 4, 6, 8, 12, 16, 24, 48]]
             ])
    factors = cases[nsymm]

    # Assuming maximum use of most generate positions
    numf = similar(factors)   # Array of assigned factors
    nleft = nions
    # Here we try to assign the highest possible factors 
    for nf = length(factors):-1:1
        numf[nf] = div(nleft, factors[nf])  # Number of ions to assign such factor
        nleft = nleft % factors[nf]
    end

    adjgen > 0 && adjust_general_positions!(numf, factors, nions, adjgen)
    return multi_unpack(factors, numf, nions), sum(numf)
end

"""
    adjust_general_positions!(numf::Vector{Int}, factors::Vector{Int}, nions::Int, adjgen::Int)

Adjust the occupation array `numf` for the multiplictiy factors to increase the number of inequivalent ions.
"""
function adjust_general_positions!(numf::Vector{Int}, factors::Vector{Int}, nions::Int, adjgen::Int)

    minequiv = sum(numf)  # Minimum inequivalent ions
    # Initialize 
    adj_inequiv = min(minequiv + adjgen, nions)
    ninequiv = rand(_RNG, minequiv:adj_inequiv) 
    ncount = -1
    # Heuristically find possibel combintaion
    while true
        ncount += 1
        if ncount > 1000 || ncount < 0  # Re initialize
            adj_inequiv = min(adj_inequiv + 1, nions)
            ninequiv = rand(_RNG, minequiv:adj_inequiv)  # Re randomize
            ncount = 0
        end

        # Randomly assign multiplicities to inequivalent ions
        fill!(numf, 0)
        for n = 1:ninequiv
            # Random factors
            nf = rand(_RNG, 1:length(factors))
            numf[nf] = numf[nf] + 1
        end

        # Break if a valid solution is found
        if sum(numf .* factors) == nions
            break
        end
    end
end

"""
Unpack multiplicty factors

# Arguments
- `factors::Vector{Int}`: An vector of multiplicity factors
- `numf::Vector{Int}`: The number of ions for each multiplicity factor

Returns a vector of multiplicity for each ion.
"""
function multi_unpack(factors::Vector{Int}, numf::Vector{Int}, nions::Int)
    nmult = 1
    mult = zeros(Int, nions)
    for nf = length(factors):-1:1
        # Set the factors, if num[nf] is zero, nothing changes
        mult[nmult:nmult + numf[nf]-1] .= factors[nf]
        nmult += numf[nf]
    end
    return mult
end


"Create symmetry images of a site"
function symimages(s::Site{T}, l::Lattice, symm::SpaceGroup)::Vector{Site{T}} where T
    nsymm= symm.num_symm
    ops = symm.symm_ops
    new_sites = Site{T}[deepcopy(s) for x in 1:nsymm]
    for ns in 1:nsymm
        vtemp = ops[:, 1:3, ns] * s.position + sum(transpose(ops[:, 4, ns]) .* cellmat(l), dims=2)
        new_sites[ns].position[:] = vtemp
    end
    new_sites
end


#= 
Steps for building symmetrical structures
    1. Identify the multiplicity
    2. Create symmetry expanded structures, each site now has fractional occupation
    3. Search and merge the sites that are close to each other, across multiple periodic boundaries
    4. Verify the merged structure contains the symmetry originally applied. If not, restart from from step one.
=#

"""
    apply_symmetry(structure::Cell, sym::SymmetryData)

Apply symmetry operations to the structure and create symmetrised 
"""
function apply_symmetry(structure::Cell, sym::SpaceGroup)
    uspec = unique(species(structure))
    nsym = sym.num_symm
    # Do this for each unique specie 
    new_sites = []
    for (isepc, spec) in enumerate(uspec)
        # Select the site with the specie
        smask = [x == spec for x in species(structure)]
        selected_sites = sites(structure)[smask]
        nselect = length(selected_sites)
        # Multiplicity for each site
        vmulti, numf = choose_multi(length(selected_sites), nsym)
        # Expand and merge for each expanded site(symmetry star)
        for (multi, site) in zip(vmulti, selected_sites)
            multi == 0. && continue
            # There are nsym number of images
            images = symimages(site, lattice(structure), sym)
            # Occupation of each image
            occ = 1 / sym.num_symm * multi
            # Merge for this symmetry star
            merged_sites = merge_nearest_sites(images, cellmat(structure), occ)
            # Append the merged sites
            append!(new_sites, merged_sites)
        end
    end
    structure.sites[:] = new_sites
    return structure
end


"""
    merge_nearest_sites(all_images, occs)

Merge sites that are close to each other in order so each sites have unity occupation.
The algorithm looks for the nearest sites with fractional occupation and merge them.
In the end, the occupation array should be either one or zero. Positions merged takes
account of the occupation. General sites with fraction occupations are merged into special
positions.
"""
function merge_nearest_sites(images, latt, occ::Float64)
    tol = 1e-5   # Floating point tolorance is need to check if the sum is close to one....
    nimage = length(images)
    positions = zeros(3, nimage)
    for (i, image) in enumerate(images)
        positions[:, i] = image.position
    end
    occs = fill(occ, nimage)
    svecs = shift_vectors(latt, 1, 1, 1)
    # Merge the positions
    im = 1
    posa = zeros(3)
    posb = zeros(3)
    while true
        im > nimage && break
        # Skip this image if it has been taken or fully merged
        if (occs[im] == 0) | ( abs(occs[im] - 1.) < tol)
            im += 1
            continue
        end

        posa[:] .= positions[:, im]
        posb[:] .= positions[:, im+1]
        mind = 99999.
        minjm = -1
        # Search for other site
        for jm in im+1:nimage
            # skip  already merged sites
            if (occs[jm] == 0) | ( abs(occs[im] - 1.) < tol)
                continue
            end
            # Search through periodic boundaries
            for svec in eachcol(svecs)
                posb[:] .= positions[:, jm] + svec 
                dist = distance_squared_between(posa, posb)
                if dist < mind
                    mind = dist
                    minjm = jm
                end
            end
        end
        # Merge the positions using weighted average
        posb[:] = positions[:, minjm]
        occb = occs[minjm]
        occa = occs[im]
        positions[:, im] .= (posa .* occa + posb .* occb) ./ (occa + occb)
        # Transfer the occupation from jm to im
        occs[im] += occs[minjm]
        occs[minjm] = 0.

    end
    @assert any(x-> (x == 0) | (x == 1), occs) "One of the occupation is not zero or one"
    # Construct the merged sites
    merged = []
    for i in 1:nimage
        occs[i] == 0 && continue
        push!(merged, Site(positions[:, i], images[1].index, images[1].symbol))
    end
    merged
end