#=
Code related to taking gradients
=#
using StaticArrays
using Base.Threads
include("repulsive_core.jl")

"""
    ForceBuffer{T}

Buffer for storing forces and stress and support their calculations
"""
struct ForceBuffer{T, N}
    fvec::Array{T, 2}
    "dF/dri"
    gvec::Array{T, 4}
    "dF/dσ"
    stotv::Array{T, 4}
    "Calculated forces"
    forces::Array{T, 2}
    fcore::Array{T, 2}
    ecore::Array{T, 1}
    "Calculated stress"
    stress::Array{T, 2}
    score::Array{T,2}
    gbuffer::Matrix{T}
    core::N
end


"""
Initialise a buffer for computing forces
"""
function ForceBuffer(fvec::Matrix{T};ndims=3, core=nothing) where {T}
    nf, nat = size(fvec)
    gvec = zeros(T, ndims, nf, nat, nat)
    stotv = zeros(T, ndims, ndims, nf, nat)
    forces = zeros(T, ndims, nat)
    fcore = zeros(T, ndims, nat)
    stress = zeros(T, ndims, ndims)
    ecore = zeros(T, 1)
    score = zeros(T, ndims, ndims)
    ForceBuffer(fvec, gvec, stotv, forces, fcore, ecore, stress, score, zeros(T, ndims, nf), core)
end


function _hard_core_update!(fcore, score_global, tid, iat, jat, rij, vij, modvij, core)

    # forces
    # Note that we add the negative size here since F=-dE/dx
    gcore = -core.g(rij, core.rcut) * core.a
    if iat != jat
        @inbounds for elm in 1:length(modvij)
            # Newton's second law - only need to update this atom
            # as we also go through the same ij pair with ji 
            fcore[elm, iat] -= 2 * gcore * modvij[elm]
            #fcore[elm, jat] +=  gcore * modvij[elm]
        end
    end
    for elm1 in 1:3
        for elm2 in 1:3
            @inbounds score_global[elm1, elm2, tid] += vij[elm1] * modvij[elm2] * gcore
        end
    end
    core.f(rij, core.rcut) * core.a
end

function _update_pij!(pij, inv_fij, r12, features3)
    for (i, feat) in enumerate(features3)
        ftmp = feat.f(r12, feat.rcut)
        # 1/f(rij)
        inv_fij[i] = 1. / ftmp
        pij[:, i] = ftmp .^ feat.p
    end
end

function _update_qjk!(qjk, inv_qji, r12, features3)
    for (i, feat) in enumerate(features3)
        ftmp = feat.f(r12, feat.rcut)
        # 1/f(rij)
        inv_qji[i] = 1. / ftmp
        qjk[:, i] = ftmp .^ feat.q
    end
end


function _update_two_body!(fvec, gtot, stot, sym, iat, jat, rij, vij, modvij, features2, offset)
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
        for m in 1:length(f.p)
            val = fast_pow(fij2, f.p[m])
            fvec[i, iat] += val
            # Force updates df(rij)^p/drij
            if val != 0.
                gfij = f.p[m] * val / fij2 * gij
            else
                gfij = zero(val)
            end

            # Force update 
            @inbounds for elm in 1:length(vij) 
                if iat != jat 
                    gtot[elm, i, iat, iat] -= modvij[elm] * gfij
                    gtot[elm, i, iat, jat] += modvij[elm] * gfij
                end
            end
            # Stress update
            @inbounds for elm2 in 1:3
                for elm1 in 1:3
                    stot[elm1, elm2, i, iat] += vij[elm1] * modvij[elm2] * gfij
                end
            end
            i += 1
        end
    end
end

function _update_three_body!(fvec, gtot, stot, iat, jat, kat, sym, pij, pik, qjk, 
                             rij, rik, rjk,
                             inv_fij, inv_fik, inv_fjk,
                             vij, vik, vjk, modvij, modvik, modvjk, features3, i)
    for (ife, f) in enumerate(features3)

        # Not for this triplets of atoms....
        if !permequal(f.sijk_idx, sym[iat], sym[jat], sym[kat])
            i += nfe3[ife]
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
        @inbounds for m in 1:f.np
            # Cache computed value
            ijkp = pij[m, ife] * pik[m, ife] 
            @inbounds for o in 1:f.nq  # Note that q is summed in the inner loop
                # Feature term
                val = ijkp * qjk[o, ife]   
                fvec[i, iat] += val

                # dv/drij, dv/drik, dv/drjk
                val == 0 && continue
                gfij = gtmp1[m] * val
                gfik = gtmp2[m] * val 
                gfjk = gtmp3[o] * val

                # Apply chain rule to the the forces
                @inbounds @fastmath @simd for elm in 1:length(vij)
                    gtot[elm, i, iat, iat] -= modvij[elm] * gfij + modvik[elm] * gfik
                    gtot[elm, i, iat, jat] += modvij[elm] * gfij - modvjk[elm] * gfjk
                    gtot[elm, i, iat, kat] += modvik[elm] * gfik + modvjk[elm] * gfjk
                end

                # Stress
                @inbounds @fastmath for elm2 in 1:3
                    @simd for elm1 in 1:3
                        stot[elm1, elm2, i, iat] += (
                            vij[elm1] * modvij[elm2] * gfij + 
                            vik[elm1] * modvik[elm2] * gfik + 
                            vjk[elm1] * modvjk[elm2] * gfjk 

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
    compute_fv_gv!(fvecs, gvecs, features2, features3, cell::Cell;nl=NeighbourList(cell, features[1].rcut))

Compute the feature vector for a given set of two and three body interactions, compute gradients as well.
Optimised version with reduced computational cost....
"""
function compute_fv_gv!(fb::ForceBuffer, features2, features3, cell::Cell;
                        nl=NeighbourList(cell, maximum(x.rcut for x in (features2..., features3...));savevec=true), 
                        offset=0,
                        ) 
   
    # Main quantities
    fvec = fb.fvec  # Size (nfe, nat)
    gtot = fb.gvec  # Size (3, nfe, nat, nat)
    stot = fb.stotv # Size (3, 3, totalfe, nat)
    fcore = fb.fcore
    score = fb.score
    core = fb.core


    nfe3 = map(nfeatures, features3) 
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
    fill!(fvec, 0)
    fill!(gtot, 0)
    fill!(stot, 0)
    fill!(fcore, 0)
    fill!(score, 0)


    maxrcut = maximum(x -> x.rcut, (features3..., features2...))

    ecore_global = zeros(nthreads())
    score_global = zeros(3,3, nthreads())

    Threads.@threads for iat = 1:nat
    #for iat = 1:nat
        tid = threadid()
        ecore = 0.
        pij = zeros(npmax3, lfe3)
        # pij_1 = zeros(npmax3, lfe3)
        inv_fij = zeros(lfe3)

        pik = zeros(npmax3, lfe3)
        #pik_1 = zeros(npmax3, lfe3)
        inv_fik = zeros(lfe3)

        qjk = zeros(nqmax3, lfe3)
        #qjk_1 = zeros(nqmax3, lfe3)
        inv_fjk = zeros(lfe3)
        gtmp3 = zeros(maximum(x.nq for x in features3))

        for (jat, jextend, rij, vij) in CellBase.eachneighbourvector(nl, iat)
            rij > maxrcut && continue
            # Compute pij
            modvij = vij / rij
            _update_pij!(pij, inv_fij, rij, features3)

            # Add hard core repulsion
            if !isnothing(core)
                ecore += _hard_core_update!(fcore, score_global, tid, iat, jat, rij, vij, modvij, core)[1]
            end

            # Update two body features and forces
            _update_two_body!(fvec, gtot, stot, sym, iat, jat, rij, vij, modvij, features2, offset)

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
                i = totalfe2 + 1 + offset

                _update_three_body!(fvec, gtot, stot, iat, jat, kat, sym, 
                                pij, pik, qjk, rij, rik, rjk, 
                                inv_fij, inv_fik, inv_fjk, 
                                vij, vik, vjk, modvij, modvik, modvjk, features3, i)
            end # i,j,k pair
        end
        ecore_global[tid] += ecore
    end
    fb.ecore[1] = sum(ecore_global)
    for i in 1:nthreads()
        score .+= score_global[:, :, i]
    end
    fvec, gtot, stot 
end

"""
    compute_fv_gv!(fvecs, gvecs, features2, features3, cell::Cell, gv::Matrix;nl=NeighbourList(cell, features[1].rcut))

Compute the feature vector for a given set of two and three body interactions, compute gradients as well.
Optimised version with reduced computational cost....

Apply the gradient on the fly.... But the a first pass is needed...
"""
function compute_fv_gv!(fb::ForceBuffer, features2, features3, cell::Cell, gv::Matrix;nl=NeighbourList(cell, maximum(x.rcut for x in (features2..., features3...));
                            savevec=true), offset=0, gv_offset=0) 
   
    # Main quantities
    fvec = fb.fvec  # Size (nfe, nat)
    gtot = fb.gvec  # Size (3, nfe, nat, nat)
    stot = fb.stotv # Size (3, 3, totalfe, nat)
    forces = fb.forces
    stress = fb.stress


    nfe3 = map(nfeatures, features3) 
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

    pij = zeros(npmax3, lfe3)
    pij_1 = zeros(npmax3, lfe3)
    inv_fij = zeros(lfe3)

    pik = zeros(npmax3, lfe3)
    pik_1 = zeros(npmax3, lfe3)
    inv_fik = zeros(lfe3)

    qjk = zeros(nqmax3, lfe3)
    qjk_1 = zeros(nqmax3, lfe3)
    inv_fjk = zeros(lfe3)

    # Reset all gradients to zero
    fill!(fvec, 0)
    fill!(gtot, 0)
    fill!(stot, 0)
    fill!(forces, 0)
    fill!(stress, 0)

    maxrcut = maximum(x -> x.rcut, (features3..., features2...))

    for iat = 1:nat
        for (jat, jextend, rij, vij) in CellBase.eachneighbourvector(nl, iat)
            rij > maxrcut && continue
            # Compute pij
            for (i, feat) in enumerate(features3)
                ftmp = feat.f(rij, feat.rcut)
                inv_fij[i] = 1. / ftmp
                @inbounds for j in 1:length(feat.p)
                    pij_1[j, i] = fast_pow(ftmp, feat.p[j]-1)
                    pij[j, i] = pij_1[j, i] * ftmp
                end
            end
            modvij = vij / rij

            # Update two body features and forces
            i = 1 + offset
            for (ife, f) in enumerate(features2)
                # Skip if this rij is not for this feature
                if !permequal(f.sij_idx, sym[iat], sym[jat]) 
                    i += f.np
                    continue
                end

                fij2 = f.f(rij, f.rcut)
                # df(rij)/drij
                gij = f.g(rij, f.rcut)
                @inbounds for m in 1:length(f.p)
                    val = fast_pow(fij2, f.p[m])
                    fvec[i, iat] += val
                    # Force updates df(rij)^p/drij
                    if val != 0.
                        gfij = f.p[m] * val / fij2 * gij
                    else
                        gfij = zero(val)
                    end

                    # For update 
                    @inbounds for elm in 1:length(vij) 
                        # gtot[elm, i, iat, iat] -= modvij[elm] * gfij
                        # gtot[elm, i, iat, jat] += modvij[elm] * gfij
                        v1 = -modvij[elm] * gfij * gv[i+gv_offset, iat]
                        forces[elm, iat] -= v1
                        forces[elm, jat] += v1
                    end
                    # Stress update
                    @inbounds for elm2 in 1:3
                        for elm1 in 1:3
                            stress[elm1, elm2] += vij[elm1] * modvij[elm2] * gfij * gv[i+gv_offset, iat]
                        end
                    end
                    i += 1
                end
            end

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
                for (i, feat) in enumerate(features3)
                    ftmp = feat.f(rik, feat.rcut)
                    inv_fik[i] = 1. / ftmp
                    @inbounds for j in 1:length(feat.p)
                        pik_1[j, i] = fast_pow(ftmp, feat.p[j]-1)
                        pik[j, i] = pik_1[j, i] * ftmp
                    end
                end

                # Compute pjk
                for (i, feat) in enumerate(features3)
                    ftmp = feat.f(rjk, feat.rcut)
                    inv_fjk[i] = 1. / ftmp
                    @inbounds for j in 1:length(feat.q)
                        qjk_1[j, i] = fast_pow(ftmp, feat.q[j]-1)
                        qjk[j, i] = qjk_1[j, i] * ftmp
                    end
                end
                
                # Starting index for three-body feature udpate
                i = totalfe2 + 1 + offset
                for (ife, f) in enumerate(features3)

                    # Not for this triplets of atoms....
                    if !permequal(f.sijk_idx, sym[iat], sym[jat], sym[kat])
                        i += nfe3[ife]
                        continue
                    end

                    # populate the buffer storing the gradients against rij, rik, rjk
                    rcut = f.rcut
                    # df(r)/dr
                    gij = f.g(rij, rcut)
                    gik = f.g(rik, rcut)
                    gjk = f.g(rjk, rcut)
                    @inbounds for m in 1:f.np
                        # Cache computed value
                        ijkp = pij[m, ife] * pik[m, ife] 
                        @inbounds for o in 1:f.nq  # Note that q is summed in the inner loop
                            # Feature term
                            val = ijkp * qjk[o, ife]   
                            fvec[i, iat] += val

                            # dv/drij, dv/drik, dv/drjk
                            if val != 0.
                                gfij = f.p[m] * val * inv_fij[ife] * gij
                                gfik = f.p[m] * val * inv_fik[ife] * gik
                                gfjk = f.q[o] * val * inv_fjk[ife] * gjk
                            else
                                gfij = zero(val)
                                gfik = zero(val)
                                gfjk = zero(val)
                            end

                            # Apply chain rule to the the forces
                            gx = gv[i+gv_offset, iat]
                            @inbounds  @fastmath @simd for elm in 1:length(vij)
                                v1 = modvij[elm] * gfij * gx
                                v2 = modvik[elm] * gfik * gx
                                v3 = modvjk[elm] * gfjk * gx 
                                # gtot[elm, i, iat, iat] -= modvij[elm] * gfij
                                # gtot[elm, i, iat, jat] += modvij[elm] * gfij
                                # gtot[elm, i, iat, iat] -= modvik[elm] * gfik
                                # gtot[elm, i, iat, kat] += modvik[elm] * gfik
                                # gtot[elm, i, iat, jat] -= modvjk[elm] * gfjk
                                # gtot[elm, i, iat, kat] += modvjk[elm] * gfjk
                                forces[elm, iat] += v2 + v1 
                                forces[elm, jat] -= v1 - v3
                                forces[elm, kat] -= v2 + v3
                            end

                            # Stress
                            @inbounds @fastmath for elm2 in 1:3
                                for elm1 in 1:3
                                    stress[elm1, elm2] -= vij[elm1] * modvij[elm2] * gfij * gx
                                    stress[elm1, elm2] -= vik[elm1] * modvik[elm2] * gfik * gx
                                    stress[elm1, elm2] -= vjk[elm1] * modvjk[elm2] * gfjk * gx
                                end
                            end
                            # Increment the feature index
                            i += 1
                        end
                    end
                end  # 3body-feature update loop 
            end # i,j,k pair
        end
    end
    fvec, forces, stress
end

"""
    _force_update!(buffer::Array{T, 2}, gv, g) where {T}

Propagate chain rule to obtain the forces
"""
function _force_update!(fb::ForceBuffer, gv) where {T}
    # Zero the buffer
    gf_at = fb.gvec
    fill!(fb.forces, 0)
    Threads.@threads for iat in axes(gf_at, 4)  # Atom index
        for j in axes(gf_at, 3)  # Atom index for the feature vector
            for i in axes(gf_at, 2)  # Feature index
                for _i in axes(fb.forces, 1)  # xyz
                    @inbounds fb.forces[_i, iat] += gf_at[_i, i, j, iat] * gv[i, j] * -1  # F(xi) = -∇E(xi)
                end
            end
        end
    end
    if !isnothing(fb.core)
        fb.forces .+= fb.fcore
    end
end

"""
    _stress_update!(buffer::Array{T, 2}, sv, s) where {T}

Propagate chain rule to obtain the stress
"""
function _stress_update!(fb::ForceBuffer, gv) where {T}
    # Zero the buffer
    gf_at = fb.stotv
    fill!(fb.stress, 0)
    #stress_copy = zeros(3, 3, nthreads())
    #Threads.@threads for j in axes(gf_at, 4)
    for j in axes(gf_at, 4)
        sid = threadid()
        for i in axes(gf_at, 3)
            for _i = 1:3
                for _j = 1:3
                    @inbounds fb.stress[_i, _j, sid] += gf_at[_i, _j, i, j] .* gv[i, j] * -1 # F(xi) = -∇E(xi)
                end
            end
        end
    end
    #fb.stress .= sum(stress_copy, dims=3)[:, :, 1]
    if !isnothing(fb.stress)
        fb.stress .+= fb.score
    end
end


function apply_chainrule!(fb::ForceBuffer, gv)
    _force_update!(fb, gv)
    _stress_update!(fb, gv)
    fb
end