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
function ForceBuffer{T}(fvec::Array{T};ndims=3) where {T}
    nf, nat = size(fvec)
    gvec = zeros(T, nf, nat, ndims, nat)
    svec = zeros(T, nf, nat, ndims, ndims, nat)
    stotv = zeros(T, nf, nat, ndims, ndims)
    forces = zeros(T, ndims, nat)
    stress = zeros(T, ndims, ndims)
    ForceBuffer(fvec, gvec, svec, stotv, forces, stress, zeros(T, 3, nf))
end


"""
    compute_two_body_fv_gv!(fvecs, gvecs, features::Vector{TwoBodyFeature{T}}, cell::Cell;nl=NeighbourList(cell, features[1].rcut)) where T

Compute the feature vector for a given set of two body interactions, compute gradients as well.

Args:
    - offset: an integer offset when updating the feature vectors
"""
function compute_two_body_fv_gv!(fb::ForceBuffer, features::Vector{TwoBodyFeature{T, N}}, cell::Cell;nl=NeighbourList(cell, features[1].rcut), offset=0) where {T, N}
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
                vtmp = gbuffer[i] .* vij / rij
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
    compute_three_body_fv_gv!(fvecs, gvecs, features::Vector{ThreeBodyFeature{T}}, cell::Cell;nl=NeighbourList(cell, features[1].rcut)) where T

Compute the feature vector for a given set of three body interactions, compute gradients as well.
"""
function compute_three_body_fv_gv!(fb::ForceBuffer, features::Vector{ThreeBodyFeature{T, N}}, cell::Cell;nl=NeighbourList(cell, features[1].rcut), offset=0) where {T, N}
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
                    @inbounds tij = gbuffer[1, i] * vij / rij
                    @inbounds tik = gbuffer[2, i] * vik / rik
                    @inbounds tjk = gbuffer[3, i] * vjk / rjk
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