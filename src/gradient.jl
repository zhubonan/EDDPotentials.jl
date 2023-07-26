#=
Code related to taking gradients
=#
using StaticArrays
using Base.Threads
include("repulsive_core.jl")

function similar_zero(x)
    y = similar(x)
    fill!(y, 0)
    y
end

function _is_same_pqrcutf(features3)
    (
        all(bd -> bd.p == features3[1].p, features3) &&
        all(bd -> bd.q == features3[1].q, features3) &&
        all(bd -> bd.rcut == features3[1].rcut, features3) &&
        all(bd -> bd.f == features3[1].f, features3)
    )
end

"""
Populate a Matrix with the content of the first column
"""
function _populate_with_first_column(arr)
    m, n = size(arr)
    for i = 2:n
        for j = 1:m
            @inbounds arr[j, i] = arr[j, 1]
        end
    end
end



"""
    ForceBuffer{T}

Buffer for storing forces and stress and support their calculations
"""
struct ForceBuffer{T,N}
    fvec::Array{T,2}
    "dF/dri"
    gvec::Array{T,4}
    "dF/dσ"
    stotv::Array{T,4}
    "Calculated forces"
    forces::Array{T,2}
    fcore::Array{T,2}
    ecore::Array{T,1}
    "Calculated stress"
    stress::Array{T,2}
    score::Array{T,2}
    gbuffer::Matrix{T}
    core::N
end


"""
Initialise a buffer for computing forces
"""
function ForceBuffer(fvec::Matrix{T}; ndims=3, core=nothing, mode="one-pass") where {T}
    nf, nat = size(fvec)
    if mode == "one-pass"
        gvec = zeros(T, ndims, nf, nat, nat)
        stotv = zeros(T, ndims, ndims, nf, nat)
    elseif mode == "two-pass"
        gvec = zeros(T, 0, 0, 0, 0)
        stotv = zeros(T, 0, 0, 0, 0)
    else
        throw(ErrorException("Unknown mode: $mode"))
    end
    forces = zeros(T, ndims, nat)
    fcore = zeros(T, ndims, nat)
    stress = zeros(T, ndims, ndims)
    ecore = zeros(T, 1)
    score = zeros(T, ndims, ndims)
    ForceBuffer(
        fvec,
        gvec,
        stotv,
        forces,
        fcore,
        ecore,
        stress,
        score,
        zeros(T, ndims, nf),
        core,
    )
end


function _hard_core_update!(fcore, score, iat, jat, rij, vij, modvij, core)

    # forces
    # Note that we add the negative size here since F=-dE/dx
    gcore = -core.g(rij, core.rcut) * core.a
    if iat != jat
        @inbounds for elm = 1:length(modvij)
            # Newton's second law - only need to update this atom
            # as we also go through the same ij pair with ji 
            fcore[elm, iat] -= 2 * gcore * modvij[elm]
            #fcore[elm, jat] +=  gcore * modvij[elm]
        end
    end
    for elm1 = 1:3
        for elm2 = 1:3
            @inbounds score[elm1, elm2] += vij[elm1] * modvij[elm2] * gcore
        end
    end
    core.f(rij, core.rcut) * core.a
end

"""
Update the ``P_{ij}`` or ``P_{ik}`` term.

# Args
- `same`: if set to `true` assume all features have the same powers/reduced distances,
so only the first one needs to be calculated.
"""
function _update_pij!(pij, inv_fij, r12, features3, same=false)
    same ? l = 1 : l = length(features3)
    for i = 1:l
        feat = features3[i]
        ftmp = feat.f(r12, feat.rcut)
        # 1/f(rij)
        @inbounds inv_fij[i] = 1.0 / ftmp
        for j = 1:length(feat.p)
            @inbounds pij[j, i] = ftmp^feat.p[j]
        end
    end
end

"""
Update the Q_{jk} term.

# Args
- `same`: if set to `true` assume all features have the same powers/reduced distances, 
so only the first one needs to be calculated.
"""
function _update_qjk!(qjk, inv_qji, r12, features3, same=false)
    same ? l = 1 : l = length(features3)
    for i = 1:l
        feat = features3[i]
        ftmp = feat.f(r12, feat.rcut)
        # 1/f(rij)
        @inbounds inv_qji[i] = 1.0 / ftmp
        for j = 1:length(feat.q)
            @inbounds qjk[j, i] = ftmp^feat.q[j]
        end
    end
end


function _update_two_body!(
    fvec,
    gtot,
    stot,
    sym,
    iat,
    jat,
    rij,
    vij,
    modvij,
    features2,
    offset,
)
    i = 1 + offset
    for (_, f) in enumerate(features2)
        # Skip if this rij is not for this feature
        if !permequal(f.sij_idx, sym[iat], sym[jat])
            i += f.np
            continue
        end

        fij2 = f.f(rij, f.rcut)
        # df(rij)/drij
        gij = f.g(rij, f.rcut)
        for m = 1:length(f.p)
            val = fast_pow(fij2, f.p[m])
            fvec[i, iat] += val
            # Force updates df(rij)^p/drij
            if val != 0.0
                gfij = f.p[m] * val / fij2 * gij
            else
                gfij = zero(val)
            end

            # Force update 
            @inbounds for elm = 1:length(vij)
                if iat != jat
                    gtot[elm, i, iat, iat] -= modvij[elm] * gfij
                    gtot[elm, i, iat, jat] += modvij[elm] * gfij
                end
            end
            # Stress update
            @inbounds for elm2 = 1:3
                for elm1 = 1:3
                    stot[elm1, elm2, i, iat] += vij[elm1] * modvij[elm2] * gfij
                end
            end
            i += 1
        end
    end
end

function _update_two_body!(
    fvec,
    forces,
    stress,
    sym,
    iat,
    jat,
    rij,
    vij,
    modvij,
    features2,
    offset,
    gv,
)
    i = 1 + offset
    for (_, f) in enumerate(features2)
        # Skip if this rij is not for this feature
        if !permequal(f.sij_idx, sym[iat], sym[jat])
            i += f.np
            continue
        end

        fij2 = f.f(rij, f.rcut)
        # df(rij)/drij
        gij = f.g(rij, f.rcut)
        for m = 1:length(f.p)
            val = fast_pow(fij2, f.p[m])
            #fvec[i, iat] += val
            # Force updates df(rij)^p/drij
            if val != 0.0
                gfij = f.p[m] * val / fij2 * gij
            else
                gfij = zero(val)
            end
            g = gv[i, iat]
            # Force update 
            @inbounds for elm = 1:length(vij)
                if iat != jat
                    tmp = modvij[elm] * gfij * g
                    forces[elm, iat] += tmp
                    forces[elm, jat] -= tmp
                end
            end
            # Stress update
            @inbounds for elm2 = 1:3
                for elm1 = 1:3
                    stress[elm1, elm2] -= vij[elm1] * modvij[elm2] * gfij * g
                end
            end
            i += 1
        end
    end
end

function _update_two_body!(fvec, sym, iat, jat, rij, features2, offset)
    i = 1 + offset
    for (_, f) in enumerate(features2)
        # Skip if this rij is not for this feature
        if !permequal(f.sij_idx, sym[iat], sym[jat])
            i += f.np
            continue
        end

        fij2 = f.f(rij, f.rcut)
        for m = 1:length(f.p)
            val = fast_pow(fij2, f.p[m])
            fvec[i, iat] += val
            # Force updates df(rij)^p/drij
            i += 1
        end
    end
end


"""
Update the three body features

# Args

- `fvec`: The feature vector to store the features calculated
- `iat`: Index of the centre atom
- `jat`: Index of the first neighbour
- `kat`: Index of the second neighbour
- `sym`: A vector storing the symbols of each atom in the structure
- `pij`: Powers of the reduced distance distance between atom *i* and *j*
- `pjk`: Powers of the reduced distance distance between atom *i* and *k*
- `qjk`: Powers of the reduced distance distance between atom *j* and *k*
- `feature3`: A tuple of the features (specifications)
- `offset`: A offset value for updating the feature vector
"""
function _update_three_body!(
    fvec,
    iat,
    jat,
    kat,
    sym,
    pij,
    pik,
    qjk,
    features3,
    offset;
    same=false,
)
    i = 1 + offset
    for (ife, f) in enumerate(features3)
        # All features share the same p,q,rcut and f so powers are only calculated for once
        if same
            ife = 1
        end
        # Not for this triplets of atom symbols....
        if !permequal(f.sijk_idx, sym[iat], sym[jat], sym[kat])
            i += f.np * f.nq
            continue
        end
        # Proceed with with update
        @inbounds for m = 1:f.np
            # Cache computed value
            ijkp = pij[m, ife] * pik[m, ife]
            @inbounds for o = 1:f.nq  # Note that q is summed in the inner loop
                # Feature term
                val = ijkp * qjk[o, ife]
                fvec[i, iat] += val
                # Increment the feature index
                i += 1
            end
        end
    end  # 3body-feature update loop 
end

"""
Update the three body term and the gradients
"""
function _update_three_body_with_gradient!(
    fvec,
    gtot,
    stot,
    iat,
    jat,
    kat,
    sym,
    pij,
    pik,
    qjk,
    rij,
    rik,
    rjk,
    inv_fij,
    inv_fik,
    inv_fjk,
    vij,
    vik,
    vjk,
    modvij,
    modvik,
    modvjk,
    features3,
    offset,;
    same=false
)
    i = 1 + offset
    for (ife, f) in enumerate(features3)
        if same
            ife = 1
        end
        # Not for this triplets of atoms....
        if !permequal(f.sijk_idx, sym[iat], sym[jat], sym[kat])
            i += f.np * f.nq
            continue
        end

        # populate the buffer storing the gradients against rij, rik, rjk
        rcut = f.rcut
        # df(r)/dr
        gij = f.g(rij, rcut) * inv_fij[ife]
        gik = f.g(rik, rcut) * inv_fik[ife]
        gjk = f.g(rjk, rcut) * inv_fjk[ife]
        gtmp1 = f.p .* gij
        gtmp2 = f.p .* gik
        gtmp3 = gjk .* f.q
        @inbounds for m = 1:f.np
            # Cache computed value
            ijkp = pij[m, ife] * pik[m, ife]
            @inbounds for o = 1:f.nq  # Note that q is summed in the inner loop
                # Feature term
                val = ijkp * qjk[o, ife]
                fvec[i, iat] += val

                # dv/drij, dv/drik, dv/drjk
                val == 0 && continue
                gfij = gtmp1[m] * val
                gfik = gtmp2[m] * val
                gfjk = gtmp3[o] * val

                # Apply chain rule to the the forces
                t1 = modvij * gfij
                t2 = modvik * gfik
                t3 = modvjk * gfjk

                @inbounds @fastmath @simd for elm = 1:length(vij)
                    gtot[elm, i, iat, iat] -= t1[elm] + t2[elm]
                    gtot[elm, i, iat, jat] += t1[elm] - t3[elm]
                    gtot[elm, i, iat, kat] += t2[elm] + t3[elm]
                end

                # Stress
                for elm2 = 1:3
                    @simd for elm1 = 1:3
                        @fastmath @inbounds stot[elm1, elm2, i, iat] += (
                            vij[elm1] * t1[elm2] +
                            vik[elm1] * t2[elm2] +
                            vjk[elm1] * t3[elm2]
                        )
                    end
                end
                # Increment the feature index
                i += 1
            end
        end
    end  # 3body-feature update loop 
end

"""
A two-pass version for computing the forces
NOTE: inefficient and not used!
"""
function _update_three_body_two_pass!(
    fvec,
    forces,
    stress,
    iat,
    jat,
    kat,
    sym,
    pij,
    pik,
    qjk,
    rij,
    rik,
    rjk,
    inv_fij,
    inv_fik,
    inv_fjk,
    vij,
    vik,
    vjk,
    modvij,
    modvik,
    modvjk,
    features3,
    offset,
    gv,
)
    i = 1 + offset
    for (ife, f) in enumerate(features3)

        # Not for this triplets of atoms....
        if !permequal(f.sijk_idx, sym[iat], sym[jat], sym[kat])
            i += f.nq * f.np
            continue
        end

        # populate the buffer storing the gradients against rij, rik, rjk
        rcut = f.rcut
        # df(r)/dr
        gij = f.g(rij, rcut) * inv_fij[ife]
        gik = f.g(rik, rcut) * inv_fik[ife]
        gjk = f.g(rjk, rcut) * inv_fjk[ife]
        gtmp1 = f.p .* gij
        gtmp2 = f.p .* gik
        gtmp3 = gjk .* f.q
        @inbounds for m = 1:f.np
            # Cache computed value
            ijkp = pij[m, ife] * pik[m, ife]
            @inbounds for o = 1:f.nq  # Note that q is summed in the inner loop
                # Feature term
                val = ijkp * qjk[o, ife]
                #fvec[i, iat] += val

                # dv/drij, dv/drik, dv/drjk
                val == 0 && continue
                gfij = gtmp1[m] * val
                gfik = gtmp2[m] * val
                gfjk = gtmp3[o] * val

                # Apply chain rule to the the forces
                g = gv[i, iat]
                @inbounds @fastmath @simd for elm = 1:length(vij)
                    t1 = modvij[elm] * gfij * g
                    t2 = modvik[elm] * gfik * g
                    t3 = modvjk[elm] * gfjk * g
                    forces[elm, iat] += t1 + t2
                    forces[elm, jat] -= t1 - t3
                    forces[elm, kat] -= t2 + t3
                    # gtot[elm, i, iat, iat] -= modvij[elm] * gfij + modvik[elm] * gfik
                    # gtot[elm, i, iat, jat] += modvij[elm] * gfij - modvjk[elm] * gfjk
                    # gtot[elm, i, iat, kat] += modvik[elm] * gfik + modvjk[elm] * gfjk
                end

                # Stress
                @inbounds @fastmath for elm2 = 1:3
                    @simd for elm1 = 1:3
                        stress[elm1, elm2] -=
                            (
                                vij[elm1] * modvij[elm2] * gfij +
                                vik[elm1] * modvik[elm2] * gfik +
                                vjk[elm1] * modvjk[elm2] * gfjk
                            ) * gv[i, iat]
                    end
                end
                # Increment the feature index
                i += 1
            end
        end
    end  # 3body-feature update loop 
end


"""
    compute_fv!(fvecs, features2, features3, cell;nl, 

Compute feature vectors only. 
"""
function compute_fv!(
    fvec,
    features2,
    features3,
    cell::Cell;
    nl=NeighbourList(
        cell,
        maximum(x.rcut for x in (features2..., features3...));
        savevec=false,
    ),
    offset=0,
    core=nothing,
)

    # Main quantities

    nfe2 = map(nfeatures, features2)
    totalfe2 = sum(nfe2)

    nat = natoms(cell)
    sym = species(cell)

    # Caches for three-body computation
    # The goal is to avoid ^ operator as much as possible by caching results
    npmax3 = maximum(length(x.p) for x in features3)
    # All values of q
    nqmax3 = maximum(length(x.q) for x in features3)


    maxrcut = maximum(x -> x.rcut, (features3..., features2...))
    ecore_buffer = [0.0 for _ = 1:nthreads()]

    # Check if three body feature all have the same powers
    # if so, there is no need to recompute the powers for each feature individually
    same_3b = _is_same_pqrcutf(features3)

    # All features are the same - so powers does not need to be recalculated for each feature.
    lfe3::Int = 1
    if !same_3b
        lfe3 = length(features3)
    end

    Threads.@threads for iat = 1:nat
        #for iat = 1:nat
        ecore = 0.0
        pij = zeros(npmax3, lfe3)
        inv_fij = zeros(lfe3)

        pik = zeros(npmax3, lfe3)
        inv_fik = zeros(lfe3)

        qjk = zeros(nqmax3, lfe3)
        inv_fjk = zeros(lfe3)

        for (jat, jextend, rij) in CellBase.eachneighbour(nl, iat)
            rij > maxrcut && continue
            # Compute pij
            length(features3) !== 0 && _update_pij!(pij, inv_fij, rij, features3, same_3b)

            if !isnothing(core)
                ecore += core.f(rij, core.rcut) * core.a
            end

            # Update two body features and forces
            _update_two_body!(fvec, sym, iat, jat, rij, features2, offset)

            # Skip three body update if no three-body feature is passed
            if length(features3) == 0
                continue
            end

            #### Update three body features and forces
            for (kat, kextend, rik) in CellBase.eachneighbour(nl, iat)
                rik > maxrcut && continue

                # Avoid double counting i j k is the same as i k j
                if kextend <= jextend
                    continue
                end

                # Compute the distance between extended j and k
                vjk = nl.ea.positions[kextend] .- nl.ea.positions[jextend]
                rjk = norm(vjk)

                rjk > maxrcut && continue

                # This is a valid pair - compute the distances

                # Compute pik
                _update_pij!(pik, inv_fik, rik, features3, same_3b)

                # Compute qjk
                _update_qjk!(qjk, inv_fjk, rjk, features3, same_3b)

                # Starting index for three-body feature udpate
                i = totalfe2 + offset
                _update_three_body!(
                    fvec,
                    iat,
                    jat,
                    kat,
                    sym,
                    pij,
                    pik,
                    qjk,
                    features3,
                    i;
                    same=same_3b,
                )
            end # i,j,k pair
        end
        ecore_buffer[threadid()] += ecore
    end
    sum(ecore_buffer)
end


"""
    compute_fv_gv!(fvecs, gvecs, features2, features3, cell::Cell;nl=NeighbourList(cell, features[1].rcut))

Compute the feature vector for a given set of two and three body interactions, compute gradients as well.
Optimised version with reduced computational cost....
"""
function compute_fv_gv!(
    fb::ForceBuffer,
    features2,
    features3,
    cell::Cell;
    nl=NeighbourList(
        cell,
        maximum(x.rcut for x in (features2..., features3...));
        savevec=true,
    ),
    offset=0,
)

    # Main quantities
    fvec = fb.fvec  # Size (nfe, nat)
    gtot = fb.gvec  # Size (3, nfe, nat, nat)
    stot = fb.stotv # Size (3, 3, totalfe, nat)
    core = fb.core

    @assert length(gtot) > 0 "The ForceBuffer passed is not suitable for single-pass calculation!"

    # Check if all features are the same - so powers does not need to be recalculated for each feature.
    same_3b = _is_same_pqrcutf(features3)
    lfe3::Int = 1
    if !same_3b
        lfe3 = length(features3)
    end

    nfe2 = map(nfeatures, features2)
    totalfe2 = sum(nfe2)

    nat = natoms(cell)
    sym = species(cell)

    # Caches for three-body computation
    # The goal is to avoid ^ operator as much as possible by caching results
    npmax3 = maximum(length(x.p) for x in features3)
    # All values of q
    nqmax3 = maximum(length(x.q) for x in features3)

    # Reset all gradients to zero
    fill!(gtot, 0)
    fill!(stot, 0)
    fill!(fb.fcore, 0)
    fill!(fb.score, 0)


    maxrcut = maximum(x -> x.rcut, (features3..., features2...))

    ecore_buffer = zeros(nthreads())
    score_buffer = [similar(fb.stress) for _ = 1:nthreads()]
    fcore_buffer = [similar(fb.fcore) for _ = 1:nthreads()]
    fill!.(score_buffer, 0)
    fill!.(fcore_buffer, 0)

    for iat = 1:nat
        #for iat = 1:nat
        ecore = 0.0
        pij = zeros(npmax3, lfe3)
        # pij_1 = zeros(npmax3, lfe3)
        inv_fij = zeros(lfe3)

        pik = zeros(npmax3, lfe3)
        #pik_1 = zeros(npmax3, lfe3)
        inv_fik = zeros(lfe3)

        qjk = zeros(nqmax3, lfe3)
        #qjk_1 = zeros(nqmax3, lfe3)
        inv_fjk = zeros(lfe3)
        score = similar_zero(fb.score)
        fcore = similar_zero(fb.fcore)


        for (jat, jextend, rij, vij) in CellBase.eachneighbourvector(nl, iat)
            rij > maxrcut && continue
            # Compute pij
            modvij = vij / rij
            _update_pij!(pij, inv_fij, rij, features3, same_3b)

            # Add hard core repulsion
            if !isnothing(core)
                ecore +=
                    _hard_core_update!(fcore, score, iat, jat, rij, vij, modvij, core)[1]
            end

            # Update two body features and forces
            _update_two_body!(
                fvec,
                gtot,
                stot,
                sym,
                iat,
                jat,
                rij,
                vij,
                modvij,
                features2,
                offset,
            )

            # Skip three body update if no three-body feature is passed
            if length(features3) == 0
                continue
            end

            #### Update three body features and forces
            for (kat, kextend, rik, vik) in CellBase.eachneighbourvector(nl, iat)
                rik > maxrcut && continue

                # Avoid double counting i j k is the same as i k j
                if kextend <= jextend
                    continue
                end

                # Compute the distance between extended j and k
                vjk = nl.ea.positions[kextend] .- nl.ea.positions[jextend]
                rjk = norm(vjk)
                rjk > maxrcut && continue
                modvik = vik / rik
                modvjk = vjk / rjk

                # This is a valid pair - compute the distances

                # Compute pik
                _update_pij!(pik, inv_fik, rik, features3, same_3b)

                # Compute qjk
                _update_qjk!(qjk, inv_fjk, rjk, features3, same_3b)

                # Starting index for three-body feature udpate
                i = totalfe2 + offset

                _update_three_body_with_gradient!(
                    fvec,
                    gtot,
                    stot,
                    iat,
                    jat,
                    kat,
                    sym,
                    pij,
                    pik,
                    qjk,
                    rij,
                    rik,
                    rjk,
                    inv_fij,
                    inv_fik,
                    inv_fjk,
                    vij,
                    vik,
                    vjk,
                    modvij,
                    modvik,
                    modvjk,
                    features3,
                    i,;
                    same=same_3b,
                )
            end # i,j,k pair
        end
        ecore_buffer[threadid()] += ecore
        score_buffer[threadid()] .+= score
        fcore_buffer[threadid()] .+= fcore
    end
    fb.ecore[1] = sum(ecore_buffer)
    for i = 1:nthreads()
        fb.score .+= score_buffer[i]
        fb.fcore .+= fcore_buffer[i]
    end
    fb
end

function compute_fv_gv_two_pass!(
    fb::ForceBuffer,
    features2,
    features3,
    cell::Cell,
    gv;
    nl=NeighbourList(
        cell,
        maximum(x.rcut for x in (features2..., features3...));
        savevec=true,
    ),
    offset=0,
)

    # Main quantities
    core = fb.core


    lfe3 = length(features3)

    nfe2 = map(nfeatures, features2)
    totalfe2 = sum(nfe2)

    nat = natoms(cell)
    sym = species(cell)

    # Caches for three-body computation
    # The goal is to avoid ^ operator as much as possible by caching results
    npmax3 = maximum(length(x.p) for x in features3)
    # All values of q
    nqmax3 = maximum(length(x.q) for x in features3)

    # Reset all gradients to zero
    fill!(fb.fcore, 0)
    fill!(fb.score, 0)
    fill!(fb.forces, 0)
    fill!(fb.stress, 0)


    maxrcut = maximum(x -> x.rcut, (features3..., features2...))

    # Thread private forces/stress
    forces_buff = [similar_zero(fb.forces) for _ = 1:nthreads()]
    stress_buff = [similar_zero(fb.stress) for _ = 1:nthreads()]

    ecore_buffer = zeros(nthreads())
    score_buffer = [similar_zero(fb.stress) for _ = 1:nthreads()]
    fcore_buffer = [similar_zero(fb.fcore) for _ = 1:nthreads()]


    Threads.@threads for iat = 1:nat

        # Allocate work space - this is necessary to allow dynamic scheduling?

        forces = similar_zero(fb.forces)
        stress = similar_zero(fb.stress)
        score = similar_zero(fb.stress)
        fcore = similar_zero(fb.fcore)

        ecore = 0.0
        pij = zeros(npmax3, lfe3)
        # pij_1 = zeros(npmax3, lfe3)
        inv_fij = zeros(lfe3)

        pik = zeros(npmax3, lfe3)
        #pik_1 = zeros(npmax3, lfe3)
        inv_fik = zeros(lfe3)

        qjk = zeros(nqmax3, lfe3)
        #qjk_1 = zeros(nqmax3, lfe3)
        inv_fjk = zeros(lfe3)

        for (jat, jextend, rij, vij) in CellBase.eachneighbourvector(nl, iat)
            rij > maxrcut && continue
            # Compute pij
            modvij = vij / rij
            _update_pij!(pij, inv_fij, rij, features3)

            # Add hard core repulsion
            if !isnothing(core)
                ecore +=
                    _hard_core_update!(fcore, score, iat, jat, rij, vij, modvij, core)[1]
            end

            # Update two body features and forces
            _update_two_body!(
                nothing,
                forces,
                stress,
                sym,
                iat,
                jat,
                rij,
                vij,
                modvij,
                features2,
                offset,
                gv,
            )

            # Skip three body update if no three-body feature is passed
            if length(features3) == 0
                continue
            end

            #### Update three body features and forces
            for (kat, kextend, rik, vik) in CellBase.eachneighbourvector(nl, iat)
                rik > maxrcut && continue

                # Avoid double counting i j k is the same as i k j
                if kextend <= jextend
                    continue
                end

                # Compute the distance between extended j and k
                vjk = nl.ea.positions[kextend] .- nl.ea.positions[jextend]
                rjk = norm(vjk)
                rjk > maxrcut && continue
                modvik = vik / rik
                modvjk = vjk / rjk

                # This is a valid pair - compute the distances

                # Compute pik
                _update_pij!(pik, inv_fik, rik, features3)

                # Compute qjk
                _update_qjk!(qjk, inv_fjk, rjk, features3)

                # Starting index for three-body feature udpate
                i = totalfe2 + offset

                _update_three_body_two_pass!(
                    nothing,
                    forces,
                    stress,
                    iat,
                    jat,
                    kat,
                    sym,
                    pij,
                    pik,
                    qjk,
                    rij,
                    rik,
                    rjk,
                    inv_fij,
                    inv_fik,
                    inv_fjk,
                    vij,
                    vik,
                    vjk,
                    modvij,
                    modvik,
                    modvjk,
                    features3,
                    i,
                    gv,
                )
            end # i,j,k pair
        end
        # Update buffers
        ecore_buffer[threadid()] += ecore
        forces_buff[threadid()] .+= forces
        stress_buff[threadid()] .+= stress
        fcore_buffer[threadid()] .+= fcore
        score_buffer[threadid()] .+= score
    end

    fb.ecore[1] = sum(ecore_buffer)
    for i = 1:nthreads()
        fb.forces .+= forces_buff[i]
        fb.stress .+= stress_buff[i]
        fb.fcore .+= fcore_buffer[i]
        fb.score .+= score_buffer[i]
    end
    fb
end


"""
    _force_update!(buffer::Array{T, 2}, gv, g) 

Propagate chain rule to obtain the forces
"""
function _force_update!(fb::ForceBuffer, nl, gv; offset=0)
    # Zero the buffer
    gf_at = fb.gvec
    fill!(fb.forces, 0)
    Threads.@threads for iat in axes(gf_at, 4)  # Atom index
        # Only neighbouring atoms will affect each other via feature vectors
        self_updated = false
        for (j, _, _) in eachneighbour(nl, iat, unique=true)
            j == iat && (self_updated = true)
            for i = 1+offset:size(gf_at, 2)
                for _i in axes(fb.forces, 1)  # xyz
                    @inbounds fb.forces[_i, iat] += gf_at[_i, i, j, iat] * gv[i, j] * -1  # F(xi) = -∇E(xi)
                end
            end
        end
        if !self_updated
            # Affect from iat itself
            j = iat
            for i = 1+offset:size(gf_at, 2)
                for _i in axes(fb.forces, 1)  # xyz
                    @inbounds fb.forces[_i, iat] += gf_at[_i, i, j, iat] * gv[i, j] * -1  # F(xi) = -∇E(xi)
                end
            end
        end
    end

    if !isnothing(fb.core)
        fb.forces .+= fb.fcore
    end
    _substract_force_drift(fb.forces)
end

"""
    _stress_update!(buffer::Array{T, 2}, sv, s) where {T}

Propagate chain rule to obtain the stress
"""
function _stress_update!(fb::ForceBuffer, gv; offset=0)
    # Zero the buffer
    gf_at = fb.stotv
    fill!(fb.stress, 0)
    for j in axes(gf_at, 4)
        for i = 1+offset:size(gf_at, 3)
            for _i = 1:3
                for _j = 1:3
                    @inbounds fb.stress[_i, _j] += gf_at[_i, _j, i, j] * gv[i, j] * -1 # F(xi) = -∇E(xi)
                end
            end
        end
    end
    #fb.stress .= sum(stress_copy, dims=3)[:, :, 1]
    if !isnothing(fb.stress)
        fb.stress .+= fb.score
    end
end

function _substract_force_drift(forces)
    for i in axes(forces, 1)
        forces[i, :] .-= sum(@view(forces[i, :]))
    end
end
