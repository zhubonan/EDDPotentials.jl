

"""
    feature_vector!(fvecs, gvecs, features2, features3, cell;kwargs...)

Args:

- `nl`: If passed, using it as an existing `NeighbourList`.

Compute the feature vector for a given set of two and three body interactions, 
This is an optimised version for feature generation, but does not compute the gradients.
OLD REFERENCE IMPLEMENTATION - NOT USED!

Returns the feature vector and the core repulsion energy if any.
"""
function feature_vector_ref!(
    fvec,
    features2::Tuple,
    features3::Tuple,
    cell::Cell;
    nl=NeighbourList(
        cell,
        maximum(x.rcut for x in (features2..., features3...));
        savevec=true,
    ),
    offset=0,
    core=nothing,
)

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

    maxrcut = maximum(x -> x.rcut, (features3..., features2...))
    ecore = 0.0
    for iat = 1:nat
        for (jat, jextend, rij) in CellBase.eachneighbour(nl, iat)
            rij > maxrcut && continue

            if !isnothing(core)
                ecore += core.f(rij, core.rcut) * core.a
            end

            # Compute pij
            for (i, feat) in enumerate(features3)
                ftmp = feat.f(rij, feat.rcut)
                inv_fij[i] = 1.0 / ftmp
                @inbounds for j = 1:length(feat.p)
                    pij_1[j, i] = fast_pow(ftmp, feat.p[j] - 1)
                    pij[j, i] = pij_1[j, i] * ftmp
                end
            end

            # Update two body features
            i = 1 + offset
            for (ife, f) in enumerate(features2)
                if permequal(f.sij_idx, sym[iat], sym[jat])
                    fij = f.f(rij, f.rcut)
                    for m = 1:f.np
                        fvec[i, iat] += fast_pow(fij, f.p[m])
                        i += 1
                    end
                else
                    i += f.np
                end
            end

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
                for (i, feat) in enumerate(features3)
                    ftmp = feat.f(rik, feat.rcut)
                    inv_fik[i] = 1.0 / ftmp
                    @inbounds for j = 1:length(feat.p)
                        pik_1[j, i] = fast_pow(ftmp, feat.p[j] - 1)
                        pik[j, i] = pik_1[j, i] * ftmp
                    end
                end

                # Compute pjk
                for (i, feat) in enumerate(features3)
                    ftmp = feat.f(rjk, feat.rcut)
                    inv_fjk[i] = 1.0 / ftmp
                    @inbounds for j = 1:length(feat.q)
                        qjk_1[j, i] = fast_pow(ftmp, feat.q[j] - 1)
                        qjk[j, i] = qjk_1[j, i] * ftmp
                    end
                end

                i = totalfe2 + 1 + offset
                for (ife, f) in enumerate(features3)
                    if !permequal(f.sijk_idx, sym[iat], sym[jat], sym[kat])
                        i += nfe3[ife]
                        continue
                    end
                    # populate the buffer storing the gradients against rij, rik, rjk
                    for m = 1:f.np
                        # Cache computed value
                        ijkp = pij[m, ife] * pik[m, ife]
                        for o = 1:f.nq  # Note that q is summed in the inner loop
                            # Feature term
                            val = ijkp * qjk[o, ife]
                            fvec[i, iat] += val
                            i += 1
                        end
                    end
                end  # 3body-feature update loop 
            end # i,j,k pair
        end
    end
    fvec, ecore
end
