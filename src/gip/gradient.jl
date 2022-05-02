#=
Code related to taking gradients
=#


"""
    feature_vector_and_gradients!(fvecs, gvecs, features::Vector{TwoBodyFeature{T}}, cell::Cell;nl=NeighbourList(cell, features[1].rcut)) where T

Compute the feature vector for a given set of two body interactions, compute gradients as well.
"""
function feature_vector_and_gradients!(evecs, gvecs, svecs, features::Vector{TwoBodyFeature{T, N}}, cell::Cell;nl=NeighbourList(cell, features[1].rcut)) where {T, N}
    # vecs -> size (nfeature, nions)
    # gvec -> size (ndims, nfeature, nions)
    # Feature vectors
    nfe = map(nfeatures, features) 
    totalfe = sum(nfe)
    nat = natoms(cell)
    sym = species(cell)
    z = zero(eltype(evecs))
    fill!(evecs, z)
    fill!(gvecs, z)  # Size of (nfe, nat, 3, nat) - gradients of the feature vectors to atoms
    fill!(svecs, z)  # Size of (nfe, nat, 3, 3, nat) - gradietn of the feature vectors to the cell deformation
    gbuffer = zeros(eltype(evecs), sum(nfe))   # Buffer for holding d(f(r)^p)/dr
    maxrcut = maximum(x -> x.rcut, features)
    
    for iat = 1:nat  # Each central atom
        for (jat, jextend, rij, vij) in CellBase.eachneighbourvector(nl, iat)
            rij > maxrcut && continue
            # Accumulate feature vectors
            ist = 1
            fill!(gbuffer, z)
            for (ife, f) in enumerate(features)
                withgradient!(evecs, gbuffer, f, rij, sym[iat], sym[jat], iat, ist)
                ist += nfe[ife]
            end
            # We now have the gbuffer filled
            for i = 1:totalfe
                vtmp = gbuffer[i] .* vij / rij
                # Gradient 
                gvecs[i, iat, :, iat] .-= vtmp
                gvecs[i, iat, :, jat] .+= vtmp

                # Derivative of the cell deformation (stress)
                # Factor of two for double counting 
                # NB. can be optimised with only one update if only total is needed
                stmp = vij * vtmp' ./ 2
                svecs[i, iat, :, :, iat] .+= stmp
                svecs[i, iat, :, :, jat] .+= stmp
            end
        end
    end
    evecs, gvecs, svecs
end


"""
    feature_vector_and_gradients!(fvecs, gvecs, features::Vector{ThreeBodyFeature{T}}, cell::Cell;nl=NeighbourList(cell, features[1].rcut)) where T

Compute the feature vector for a given set of two body interactions, compute gradients as well.
"""
function feature_vector_and_gradients!(evecs, gvecs, svecs, features::Vector{ThreeBodyFeature{T, N}}, cell::Cell;nl=NeighbourList(cell, features[1].rcut)) where {T, N}
    # vecs -> size (nfeature, nions)
    # gvec -> size (ndims, nfeature, nions)
    # Feature vectors
    nfe = map(nfeatures, features) 
    totalfe = sum(nfe)
    nat = natoms(cell)
    sym = species(cell)
    z = zero(eltype(evecs))
    fill!(evecs, z)
    fill!(gvecs, z)  # Size of (nfe, nat, 3, nat) - gradients of the feature vectors to atoms
    fill!(svecs, z)  # Size of (nfe, nat, 3, 3, nat) - gradietn of the feature vectors to the cell deformation
    gbuffer = zeros(eltype(evecs), 3, totalfe)   # Buffer for holding df/dr for rij, rik, rjk
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
                ist = 1
                for (ife, f) in enumerate(features)
                    # populate the buffer storing the gradients against rij, rik, rjk
                    withgradient!(evecs, gbuffer, f, rij, rik, rjk, sym[iat], sym[jat], sym[kat], iat, ist)
                    ist += nfe[ife]
                end
                # Update forces and the stres
                for i = 1:totalfe
                    tij = gbuffer[1, i] .* vij / rij
                    tik = gbuffer[2, i] .* vik / rik
                    tjk = gbuffer[3, i] .* vjk / rjk
                    # Gradient with positions
                    gvecs[i, iat, :, iat] .-= tij
                    gvecs[i, iat, :, jat] .+= tij
                    gvecs[i, iat, :, iat] .-= tik
                    gvecs[i, iat, :, kat] .+= tik
                    gvecs[i, iat, :, jat] .-= tjk
                    gvecs[i, iat, :, kat] .+= tjk

                    # Stress (gradient on cell deformation)
                    sij = vij * tij' ./ 2
                    sik = vik * tik' ./ 2
                    sjk = vjk * tjk' ./ 2
                    svecs[i, iat, :, :, iat] .+= sij
                    svecs[i, iat, :, :, jat] .+= sij
                    svecs[i, iat, :, :, iat] .+= sik
                    svecs[i, iat, :, :, kat] .+= sik
                    svecs[i, iat, :, :, jat] .+= sjk
                    svecs[i, iat, :, :, kat] .+= sjk
                end
            end
        end
    end
    evecs, gvecs, svecs
end