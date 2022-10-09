#=
Code related to taking gradients
=#

"""
    ForceBuffer{T}

Buffer for storing forces and stress and support their calculations
"""
struct ForceBuffer{T}
    fvec::Array{T, 2}
    "dF/dri"
    gvec::Array{T, 4}
    "dF/dσi"
    svec::Array{T, 5}
    "dF/dσ"
    stotv::Array{T, 4}
    "Calculated forces"
    forces::Array{T, 2}
    "Calculated stress"
    stress::Array{T, 2}
    gbuffer::Matrix{T}
end


"""
Initialise a buffer for computing forces
"""
function ForceBuffer(fvec::Matrix{T};ndims=3) where {T}
    nf, nat = size(fvec)
    gvec = zeros(T, ndims, nf, nat, nat)
    svec = zeros(T, ndims, ndims, nf, nat, nat)
    stotv = zeros(T, ndims, ndims, nf, nat)
    forces = zeros(T, ndims, nat)
    stress = zeros(T, ndims, ndims)
    ForceBuffer(fvec, gvec, svec, stotv, forces, stress, zeros(T, 3, nf))
end

_collect_stress!(fb::ForceBuffer) = fb.stotv .= sum(fb.svec, dims=5)

"""
    compute_two_body_fv_gv!(fb::ForceBuffer, features::Vector{TwoBodyFeature}, cell::Cell;...) 

Compute the feature vector for a given set of two body interactions, compute gradients as well.

Args:
    - offset: an integer offset when updating the feature vectors
"""
function compute_two_body_fv_gv!(fb::ForceBuffer, features::Vector{T}, cell::Cell;nl=NeighbourList(cell, features[1].rcut;savevec=true), offset=0) where {T<:TwoBodyFeature}
    # vecs -> size (nfeature, nions)
    # gvec -> size (ndims, nfeature, nions)
    # Feature vectors
    fvec = fb.fvec
    gvecs = fb.gvec
    svecs = fb.svec
    gbuffer = fb.gbuffer # Buffer for holding d(f(r)^p)/dr
    fill!(gbuffer, 0)

    nfe = map(nfeatures, features) 
    totalfe = sum(nfe)
    nat = natoms(cell)
    sym = species(cell)

    fvec[offset+1:offset+totalfe, :] .= 0 # Size of (nfe, nat) - feature vectors for each atom
    gvecs[offset+1:offset+totalfe, :, :, :] .= 0 # Size of (nfe, nat, 3, nat) - gradients of the feature vectors to atoms
    svecs[offset+1:offset+totalfe, :, :, :, :] .= 0 # Size of (nfe, nat, 3, 3, nat) - gradient of the feature vectors to the cell deformation

    maxrcut = maximum(x -> x.rcut, features)
    for iat = 1:nat  # Each central atom
        for (jat, jextend, rij, vij) in CellBase.eachneighbourvector(nl, iat)
            rij > maxrcut && continue
            # Accumulate feature vectors
            ist = 1 + offset
            # Clear the buffer for storing gradient, since the called function *accumulates* it
            fill!(gbuffer, 0)
            for (ife, f) in enumerate(features)
                withgradient!(fvec, gbuffer, f, rij, sym[iat], sym[jat], iat, ist)
                ist += nfe[ife]
            end
            # We now have the gbuffer filled
            for i = 1:totalfe
                j = i + offset
                vtmp = gbuffer[j] .* vij / rij
                # Gradient 
                for idx in 1:size(vtmp, 1)
                    @inbounds gvecs[j, iat, idx, iat] -= vtmp[idx]
                    @inbounds gvecs[j, iat, idx, jat] += vtmp[idx]
                end

                # Derivative of the cell deformation (stress)
                # Factor of two for double counting 
                # NB. can be optimised with only one update if only total is needed
                stmp = vij * vtmp' ./ 2
                for jdx in 1:size(stmp, 2)
                    for idx in 1:size(stmp, 1)
                        @inbounds svecs[j, iat, idx, jdx, iat] += stmp[idx, jdx]
                        @inbounds svecs[j, iat, idx, jdx, jat] += stmp[idx, jdx]
                    end
                end
            end
        end
    end
    fvec, gvecs, svecs
end

"""
    compute_three_body_fv_gv!(fvecs, gvecs, features::Vector{ThreeBodyFeature}, cell::Cell;nl=NeighbourList(cell, features[1].rcut))

Compute the feature vector for a given set of three body interactions, compute gradients as well.
"""
function compute_three_body_fv_gv!(fb::ForceBuffer, features::Vector{T}, cell::Cell;nl=NeighbourList(cell, features[1].rcut;savevec=true), offset=0) where {T<:ThreeBodyFeature}
    # vecs -> size (nfeature, nions)
    # gvec -> size (ndims, nfeature, nions)
    # Feature vectors
    fvec = fb.fvec
    gvecs = fb.gvec
    svecs = fb.svec
    gbuffer = fb.gbuffer # Buffer for holding d(f(r)^p)/dr
    fill!(gbuffer, zero(eltype(gbuffer)))

    nfe = map(nfeatures, features) 
    totalfe = sum(nfe)
    nat = natoms(cell)
    sym = species(cell)
    fvec[offset+1:offset+totalfe, :] .= 0 # Size of (nfe, nat) - feature vectors for each atom
    gvecs[:, offset+1:offset+totalfe, :, :] .= 0 # Size of (nfe, nat, 3, nat) - gradients of the feature vectors to atoms
    svecs[:, offset+1:offset+totalfe, :, :, :] .= 0 # Size of (nfe, nat, 3, 3, nat) - gradient of the feature vectors to the cell deformation


    maxrcut = maximum(x -> x.rcut, features)
    for iat = 1:nat
        for (jat, jextend, rij, vij) in CellBase.eachneighbourvector(nl, iat)
            rij > maxrcut && continue
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
                # accumulate the feature vector
                # Clear the buffer for storing gradient, since the called function *accumulates* it
                fill!(gbuffer, 0)
                ist = 1 + offset
                for (ife, f) in enumerate(features)
                    # populate the buffer storing the gradients against rij, rik, rjk
                    withgradient!(fvec, gbuffer, f, rij, rik, rjk, sym[iat], sym[jat], sym[kat], iat, ist)
                    ist += nfe[ife]
                end
                # Update forces and the stres
                for i = 1:totalfe
                    j = i + offset
                    tij = gbuffer[1, j] * vij / rij
                    tik = gbuffer[2, j] * vik / rik
                    tjk = gbuffer[3, j] * vjk / rjk
                    # Gradient with positions
                    for idx in 1:length(tij)
                        @inbounds gvecs[idx, j, iat, iat] -= tij[idx]
                        @inbounds gvecs[idx, j, iat, jat] += tij[idx]
                        @inbounds gvecs[idx, j, iat, iat] -= tik[idx]
                        @inbounds gvecs[idx, j, iat, kat] += tik[idx]
                        @inbounds gvecs[idx, j, iat, jat] -= tjk[idx]
                        @inbounds gvecs[idx, j, iat, kat] += tjk[idx]
                    end
                    iat==2 && jat==1 && i==1 && kextend == 95 &&@show(gvecs[2,2,1,3], jat, kextend)
                    # Stress (gradient on cell deformation)
                    sij = vij .* tij' ./ 2
                    sik = vik .* tik' ./ 2
                    sjk = vjk .* tjk' ./ 2
                    for jdx in 1:size(sij, 2)
                        for idx in 1:size(sij, 1)
                            @inbounds svecs[idx, jdx, j, iat, iat] += sij[idx, jdx]
                            @inbounds svecs[idx, jdx, j, iat, jat] += sij[idx, jdx]
                            @inbounds svecs[idx, jdx, j, iat, iat] += sik[idx, jdx]
                            @inbounds svecs[idx, jdx, j, iat, kat] += sik[idx, jdx]
                            @inbounds svecs[idx, jdx, j, iat, jat] += sjk[idx, jdx]
                            @inbounds svecs[idx, jdx, j, iat, kat] += sjk[idx, jdx]
                        end
                    end
                end
            end
        end
    end
    fvec, gvecs, svecs
end


"""
    _force_update!(buffer::Array{T, 2}, gv, g) where {T}

Propagate chain rule to obtain the forces
"""
function _force_update!(fb::ForceBuffer, gv) where {T}
    # Zero the buffer
    gf_at = fb.gvec
    fill!(fb.forces, 0)
    for iat in axes(gf_at, 4)
        for j in axes(gf_at, 2)
            for i in axes(gf_at, 1)
                for _i in axes(fb.forces, 1)
                    fb.forces[_i, iat] += gf_at[_i, i, j, iat] * gv[i, j] * -1  # F(xi) = -∇E(xi)
                end
            end
        end
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
    for j in axes(gf_at, 2)
        for i in axes(gf_at, 1)
            for _i = 1:3
                for _j = 1:3
                    fb.stress[_i, _j] += gf_at[_i, _j, i, j] .* gv[i, j] * -1 # F(xi) = -∇E(xi)
                end
            end
        end
    end
end


function apply_chainrule!(fb::ForceBuffer, gv)
    _force_update!(fb, gv)
    _stress_update!(fb, gv)
    fb
end



"""
    compute_fv_gv_new!(fvecs, gvecs, features::Vector{ThreeBodyFeature}, cell::Cell;nl=NeighbourList(cell, features[1].rcut))

Compute the feature vector for a given set of two and three body interactions, compute gradients as well.
Optimised version with reduced computational cost....
"""
function compute_fv_gv_new!(fb::ForceBuffer, features2::Vector{N}, features3::Vector{T}, cell::Cell;nl=NeighbourList(cell, maximum(x.rcut for x in (features2..., features3...));savevec=true)) where {T<:ThreeBodyFeature, N<:TwoBodyFeature}
   
    # Main quantities
    fvec = fb.fvec  # Size (nfe, nat)
    gtot = fb.gvec  # Size (3, nfe, nat, nat)
    stot = fb.stotv # Size (3, 3, totalfe, nat)


    nfe3 = map(nfeatures, features3) 
    lfe3 = length(features3)

    nfe2 = map(nfeatures, features2) 
    totalfe2 = sum(nfe2)

    nat = natoms(cell)
    sym = species(cell)

    # Caches for three-body computation
    # The goal is to avoid ^ operator as much as possible
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
            i = 1
            for (ife, f) in enumerate(features2)
                fij2 = f.f(rij, f.rcut)
                # df(rij)/drij
                gij = f.g(rij, f.rcut)
                for m in 1:length(f.p)
                    val = fast_pow(fij2, f.p[m])
                    fvec[i, iat] += val
                    # Force updates df(rij)^p/drij
                    gfij = f.p[m] * val / fij2 * gij
                    # For update 
                    for elm in 1:length(vij) 
                        gtot[elm, i, iat, iat] -= modvij[elm] * gfij
                        gtot[elm, i, iat, jat] += modvij[elm] * gfij
                    end
                    # Stress update
                    for elm2 in 1:3
                        for elm1 in 1:3
                            stot[elm1, elm2, i, iat] += vij[elm1] * modvij[elm2] * gfij
                        end
                    end
                    i += 1
                end
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

                ist = totalfe2 + 1
                for (ife, f) in enumerate(features3)
                    if permequal(f.sijk_idx, sym[iat], sym[jat], sym[kat])
                        # populate the buffer storing the gradients against rij, rik, rjk
                        rcut = f.rcut
                        gij = f.g(rij, rcut)
                        gik = f.g(rik, rcut)
                        gjk = f.g(rjk, rcut)
                        i = ist  # Index of the element
                        for m in 1:f.np
                            # Cache computed value
                            ijkp = pij[m, ife] * pik[m, ife] 
                            for o in 1:f.nq  # Note that q is summed in the inner loop
                                # Feature term
                                val = ijkp * qjk[o, ife]   
                                fvec[i, iat] += val
                                # dv/drij, dv/drik, dv/drjk
                                gfij = f.p[m] * val * inv_fij[ife] * gij
                                gfik = f.p[m] * val * inv_fik[ife] * gik
                                gfjk = f.q[o] * val * inv_fjk[ife] * gjk

                                # Apply chain rule to the the forces
                                @inbounds @fastmath @simd for elm in 1:length(vij)
                                    gtot[elm, i, iat, iat] -= modvij[elm] * gfij
                                    gtot[elm, i, iat, jat] += modvij[elm] * gfij
                                    gtot[elm, i, iat, iat] -= modvik[elm] * gfik
                                    gtot[elm, i, iat,  kat] += modvik[elm] * gfik
                                    gtot[elm, i, iat, jat] -= modvjk[elm] * gfjk
                                    gtot[elm, i, iat, kat] += modvjk[elm] * gfjk
                                end

                                # Stress
                                @inbounds @fastmath for elm2 in 1:3
                                    @simd for elm1 in 1:3
                                        stot[elm1, elm2, i, iat] += vij[elm1] * modvij[elm2] * gfij
                                        stot[elm1, elm2, i, iat] += vik[elm1] * modvik[elm2] * gfik
                                        stot[elm1, elm2, i, iat] += vjk[elm1] * modvjk[elm2] * gfjk
                                    end
                                end

                                i += 1
                            end
                        end
                        # Incremeta the feature 
                    end
                    ist += nfe3[ife]
                end  # 3body-feature update loop 
            end # i,j,k pair
        end
    end
    fvec, gtot, stot
end
