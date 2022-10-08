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
    gvec = zeros(T, nf, nat, ndims, nat)
    svec = zeros(T, nf, nat, ndims, ndims, nat)
    stotv = zeros(T, nf, nat, ndims, ndims)
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
    gvecs[offset+1:offset+totalfe, :, :, :] .= 0 # Size of (nfe, nat, 3, nat) - gradients of the feature vectors to atoms
    svecs[offset+1:offset+totalfe, :, :, :, :] .= 0 # Size of (nfe, nat, 3, 3, nat) - gradient of the feature vectors to the cell deformation


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
                        @inbounds gvecs[j, iat, idx, iat] -= tij[idx]
                        @inbounds gvecs[j, iat, idx, jat] += tij[idx]
                        @inbounds gvecs[j, iat, idx, iat] -= tik[idx]
                        @inbounds gvecs[j, iat, idx, kat] += tik[idx]
                        @inbounds gvecs[j, iat, idx, jat] -= tjk[idx]
                        @inbounds gvecs[j, iat, idx, kat] += tjk[idx]
                    end
                    # Stress (gradient on cell deformation)
                    sij = vij .* tij' ./ 2
                    sik = vik .* tik' ./ 2
                    sjk = vjk .* tjk' ./ 2
                    for jdx in 1:size(sij, 2)
                        for idx in 1:size(sij, 1)
                            @inbounds svecs[j, iat, idx, jdx, iat] += sij[idx, jdx]
                            @inbounds svecs[j, iat, idx, jdx, jat] += sij[idx, jdx]
                            @inbounds svecs[j, iat, idx, jdx, iat] += sik[idx, jdx]
                            @inbounds svecs[j, iat, idx, jdx, kat] += sik[idx, jdx]
                            @inbounds svecs[j, iat, idx, jdx, jat] += sjk[idx, jdx]
                            @inbounds svecs[j, iat, idx, jdx, kat] += sjk[idx, jdx]
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
                    fb.forces[_i, iat] += gf_at[i, j, _i, iat] * gv[i, j] * -1  # F(xi) = -∇E(xi)
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
                    fb.stress[_i, _j] += gf_at[i, j, _i, _j] .* gv[i, j] * -1 # F(xi) = -∇E(xi)
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
    compute_three_body_fv_gv_new!(fvecs, gvecs, features::Vector{ThreeBodyFeature}, cell::Cell;nl=NeighbourList(cell, features[1].rcut))

Compute the feature vector for a given set of three body interactions, compute gradients as well.
Optimised version with reduced computational cost....
"""
function compute_three_body_fv_gv_new!(fb::ForceBuffer, features::Vector{T}, cell::Cell;nl=NeighbourList(cell, features[1].rcut;savevec=true), offset=0) where {T<:ThreeBodyFeature}
    # vecs -> size (nfeature, nions)
    # gvec -> size (ndims, nfeature, nions)
    # Feature vectors
    fvec = fb.fvec
    gvecs = fb.gvec
    svecs = fb.svec
    gbuffer = fb.gbuffer # Buffer for holding d(f(r)^p)/dr
    fill!(gbuffer, zero(eltype(gbuffer)))

    nfe = map(nfeatures, features) 
    lfeat = length(features)
    totalfe = sum(nfe)
    nat = natoms(cell)
    sym = species(cell)
    fvec[offset+1:offset+totalfe, :] .= 0 # Size of (nfe, nat) - feature vectors for each atom
    gvecs[offset+1:offset+totalfe, :, :, :] .= 0 # Size of (nfe, nat, 3, nat) - gradients of the feature vectors to atoms
    svecs[offset+1:offset+totalfe, :, :, :, :] .= 0 # Size of (nfe, nat, 3, 3, nat) - gradient of the feature vectors to the cell deformation

    # All values of P
    npmax = maximum(length(x.p) for x in features)
    # All values of q
    nqmax = maximum(length(x.q) for x in features)

    pij = zeros(npmax, lfeat)
    pij_1 = zeros(npmax, lfeat)
    fij = zeros(lfeat)

    pik = zeros(npmax, lfeat)
    pik_1 = zeros(npmax, lfeat)
    fik = zeros(lfeat)

    qjk = zeros(nqmax, lfeat)
    qjk_1 = zeros(nqmax, lfeat)
    fjk = zeros(lfeat)


    maxrcut = maximum(x -> x.rcut, features)
    for iat = 1:nat
        for (jat, jextend, rij, vij) in CellBase.eachneighbourvector(nl, iat)
            rij > maxrcut && continue
            # Compute pij
            for (i, feat) in enumerate(features)
                ftmp = feat.f(rij, feat.rcut)
                fij[i] = ftmp
                @inbounds for j in 1:length(feat.p)
                    pij_1[j, i] = fast_pow(ftmp, feat.p[j]-1)
                    pij[j, i] = pij_1[j, i] * ftmp
                end
            end

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

                # This is a valid pair - compute the distances
 
                # Compute pik
                for (i, feat) in enumerate(features)
                    ftmp = feat.f(rik, feat.rcut)
                    fik[i] = ftmp
                    @inbounds for j in 1:length(feat.p)
                        pik_1[j, i] = fast_pow(ftmp, feat.p[j]-1)
                        pik[j, i] = pik_1[j, i] * ftmp
                    end
                end

                # Compute pjk
                for (i, feat) in enumerate(features)
                    ftmp = feat.f(rjk, feat.rcut)
                    fjk[i] = ftmp
                    @inbounds for j in 1:length(feat.q)
                        qjk_1[j, i] = fast_pow(ftmp, feat.q[j]-1)
                        qjk[j, i] = qjk_1[j, i] * ftmp
                    end
                end

                # accumulate the feature vector
                # Clear the buffer for storing gradient, since the called function *accumulates* it
                fill!(gbuffer, 0)
                ist = 1 + offset
                for (ife, f) in enumerate(features)
                    if permequal(f.sijk_idx, sym[iat], sym[jat], sym[kat])
                        # populate the buffer storing the gradients against rij, rik, rjk
                        #withgradient!(fvec, gbuffer, f, rij, rik, rjk, sym[iat], sym[jat], sym[kat], iat, ist)
                        rcut = f.rcut
                        gij = f.g(rij, rcut)
                        gik = f.g(rik, rcut)
                        gjk = f.g(rjk, rcut)
                        i = ist  # Index of the element

                        for m in 1:f.np
                            # Cache computed value
                            ijkp = pij[m, ife] * pik[m, ife] 
                            tmp = pij[m, ife] * pik_1[m, ife] 

                            for o in 1:f.nq  # Note that q is summed in the inner loop
                                # Feature term
                                fvec[i, iat] += ijkp * qjk[o, ife] 
                                # Gradient - NOTE this can be optimised further...
                                #g[1, i] += f.p[m] * fast_pow(fij, (f.p[m] - 1)) * fast_pow(fik, f.p[m]) * fast_pow(fjk, f.q[o]) * gij
                                gbuffer[1, i] += f.p[m] * pij_1[m,ife] * pik[m, ife] * qjk[o, ife] * gij
                                # g[2, i] += tmp  * f.p[m] * fast_pow(fjk, f.q[o]) * gik
                                gbuffer[2, i] += tmp  * f.p[m] * qjk[o, ife] * gik
                                # g[3, i] += ijkp * f.q[o] * fast_pow(fjk, (f.q[o] - 1)) * gjk
                                gbuffer[3, i] += ijkp * f.q[o] * qjk_1[o, ife]  * gjk
                                i += 1
                            end
                        end
                    end
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
                        @inbounds gvecs[j, iat, idx, iat] -= tij[idx]
                        @inbounds gvecs[j, iat, idx, jat] += tij[idx]
                        @inbounds gvecs[j, iat, idx, iat] -= tik[idx]
                        @inbounds gvecs[j, iat, idx, kat] += tik[idx]
                        @inbounds gvecs[j, iat, idx, jat] -= tjk[idx]
                        @inbounds gvecs[j, iat, idx, kat] += tjk[idx]
                    end
                    # Stress (gradient on cell deformation)
                    sij = vij .* tij' ./ 2
                    sik = vik .* tik' ./ 2
                    sjk = vjk .* tjk' ./ 2
                    for jdx in 1:size(sij, 2)
                        for idx in 1:size(sij, 1)
                            @inbounds svecs[j, iat, idx, jdx, iat] += sij[idx, jdx]
                            @inbounds svecs[j, iat, idx, jdx, jat] += sij[idx, jdx]
                            @inbounds svecs[j, iat, idx, jdx, iat] += sik[idx, jdx]
                            @inbounds svecs[j, iat, idx, jdx, kat] += sik[idx, jdx]
                            @inbounds svecs[j, iat, idx, jdx, jat] += sjk[idx, jdx]
                            @inbounds svecs[j, iat, idx, jdx, kat] += sjk[idx, jdx]
                        end
                    end
                end
            end
        end
    end
    fvec, gvecs, svecs
end
