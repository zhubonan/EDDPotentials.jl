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
    nat = natoms(cell)
    sym = species(cell)
    z = zero(eltype(evecs))
    fill!(evecs, z)
    fill!(gvecs, z)  # Size of (nfe, nat, 3, nat) - gradients of the feature vectors to atoms
    fill!(svecs, z)  # Size of (nfe, nat, 3, 3, nat) - gradietn of the feature vectors to the cell deformation
    gbuffer = zeros(eltype(evecs), sum(nfe))   # Buffer for holding d(f(r)^p)/dr
    rcut = maximum(x -> x.rcut, features)
    
    for iat = 1:nat  # Each central atom
        for (jat, jextend, rij, vij) in CellBase.eachneighbourvector(nl, iat)
            rij > rcut && continue
            # Accumulate feature vectors
            ist = 1
            fill!(gbuffer, z)
            for (ife, f) in enumerate(features)
                withgradient!(evecs, gbuffer, f, rij, sym[iat], sym[jat], iat, ist)
                ist += nfe[ife]
            end
            # We now have the gbuffer filled
            for i = 1:sum(nfe)
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
    evecs
end