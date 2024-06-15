#=
Code related to taking gradients
=#
using StaticArrays
using Base.Threads

include("repulsive_core.jl")
include("workspace.jl")

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
            #@inbounds 
            arr[j, i] = arr[j, 1]
        end
    end
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
        #@inbounds 
        inv_fij[i] = 1.0 / ftmp
        for j = 1:length(feat.p)
            #@inbounds 
            pij[j, i] = ftmp^feat.p[j]
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
        #@inbounds 
        inv_qji[i] = 1.0 / ftmp
        for j = 1:length(feat.q)
            #@inbounds 
            qjk[j, i] = ftmp^feat.q[j]
        end
    end
end


"""
    compute_fv_gv_one!(workspace::GradientWorkspace, features2, features3, iat, cell, nl; offset=workspace.one_body_offset)

Compute the feature vector for a given set of two and three body interactions, compute gradients as well for a single 
atom.
"""
function compute_fv_gv_one!(
    workspace::GradientWorkspace,
    features2,
    features3,
    iat::Int,
    cell::Cell,
    nl::NeighbourList,
    ;
    offset=workspace.one_body_offset,
    do_grad=workspace.do_grad
)

    # NeighbourList must be built with vectors for gradient computation
    do_grad && (@assert nl.has_vectors)
    # Main quantities
    (; fvec, gvec, gvec_index, gvec_nn, stotv, hardcore) = workspace
    (; core, fcore, score, ecore) = hardcore

    if do_grad
        gvec[:, :, :, iat] .= 0.
        stotv[:, :, :, iat] .= 0.
    end

    # Check if all features are the same - so powers does not need to be recalculated for each feature.
    same_3b = _is_same_pqrcutf(features3)
    lfe3::Int = 1
    if !same_3b
        lfe3 = length(features3)
    end

    nfe2 = map(nfeatures, features2)
    totalfe2 = sum(nfe2)
    sym = species(cell)

    # Caches for three-body computation
    # The goal is to avoid ^ operator as much as possible by caching results
    npmax3 = maximum(length(x.p) for x in features3)
    # All values of q
    nqmax3 = maximum(length(x.q) for x in features3)

    # Maximum cut off 
    maxrcut = maximum(x -> x.rcut, (features3..., features2...))

    # Caches for three body terms
    pij = zeros(npmax3, lfe3)
    inv_fij = zeros(lfe3)

    pik = zeros(npmax3, lfe3)
    inv_fik = zeros(lfe3)

    qjk = zeros(nqmax3, lfe3)
    inv_fjk = zeros(lfe3)
    tiny_dist = 1e-14


    neighbour_indices = unique(nl.orig_indices[1:nl.nneigh[iat], iat])
    if do_grad
        # TODO information below may be cached to accelerate the computation?
        # Compute the indexing array for the gradients - tells each element is connected to which atom
        # The first element is always the centring atom
        gvec_index[1, iat] = iat
        # The rest are teh neighbouring atoms (can include the central atom itself)
        # We always use the original index not the extended index
        gvec_index[2:length(neighbour_indices)+1, iat] .= neighbour_indices
        gvec_nn[iat] = length(neighbour_indices) + 1
    end

    # Start main loop - iterate between the neighbours

    for (jat, jextend, rij, vij) in CellBase.eachneighbourvector(nl, iat)
        rij > maxrcut && continue
        # Check atoms are not overlapping...
        @assert rij > tiny_dist

        ineigth_j = findfirst(x -> x == jat, neighbour_indices) + 1

        # Compute pij
        modvij = vij / rij
        _update_pij!(pij, inv_fij, rij, features3, same_3b)

        # Add hard core repulsion
        if !isnothing(core)
            _hard_core_update!(ecore, fcore, score, iat, jat, rij, vij, modvij, core)[1]
        end

        # Update two body features and forces
        i = 1 + offset

        # Find the index of the atom j to be used
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

                # Compute gradients if needed
                if do_grad
                    # Force updates df(rij)^p/drij
                    if val != 0.0
                        @inbounds gfij = f.p[m] * val / fij2 * gij
                    else
                        gfij = zero(val)
                    end
                    # Force update 
                    #@inbounds 
                    for elm = axes(gvec, 1)
                        gvec[elm, i, 1, iat] -= modvij[elm] * gfij
                        gvec[elm, i, ineigth_j, iat] += modvij[elm] * gfij
                    end
                    # Stress update
                    #@inbounds 
                    for elm2 = axes(stotv, 2)
                        for elm1 = axes(stotv, 1)
                            stotv[elm1, elm2, i, iat] += vij[elm1] * modvij[elm2] * gfij
                        end
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
            @assert rik > tiny_dist


            # Avoid double counting i j k is the same as i k j
            if kextend <= jextend
                continue
            end

            # Compute the distance between extended j and k
            vjk = nl.ea.positions[kextend] .- nl.ea.positions[jextend]
            rjk = norm(vjk)
            rjk > maxrcut && continue
            @assert rjk > tiny_dist

            ineigth_k = findfirst(x -> x == kat, neighbour_indices) + 1

            # Start update of the three body term
            modvik = vik / rik
            modvjk = vjk / rjk

            # This is a valid pair - compute the distances

            # Compute pik
            _update_pij!(pik, inv_fik, rik, features3, same_3b)

            # Compute qjk
            _update_qjk!(qjk, inv_fjk, rjk, features3, same_3b)

            # Starting index for three-body feature udpate
            i = totalfe2 + offset + 1


            for (ife, f) in enumerate(features3)
                if same_3b
                    ife = 1
                end
                # Not for this triplets of atoms....
                if !permequal(f.sijk_idx, sym[iat], sym[jat], sym[kat])
                    i += f.np * f.nq
                    continue
                end

                # populate the buffer storing the gradients against rij, rik, rjk
                rcut = f.rcut
                if do_grad
                    # df(r)/dr
                    gij = f.g(rij, rcut) * inv_fij[ife]
                    gik = f.g(rik, rcut) * inv_fik[ife]
                    gjk = f.g(rjk, rcut) * inv_fjk[ife]
                    gtmp1 = f.p .* gij
                    gtmp2 = f.p .* gik
                    gtmp3 = gjk .* f.q
                end

                #@inbounds 
                for m = 1:f.np
                    # Cache computed value
                    ijkp = pij[m, ife] * pik[m, ife]

                    #@inbounds 
                    for o = 1:f.nq  # Note that q is summed in the inner loop
                        # Feature term
                        val = ijkp * qjk[o, ife]
                        fvec[i, iat] += val

                        # If computing gradients
                        if do_grad
                            # dv/drij, dv/drik, dv/drjk
                            val == 0 && continue

                            # Apply chain rule to the the forces
                            t1 = modvij * gtmp1[m] * val
                            t2 = modvik * gtmp2[m] * val
                            t3 = modvjk * gtmp3[o] * val

                            for elm = axes(gvec, 1)
                                gvec[elm, i, 1, iat] -= t1[elm] + t2[elm]
                                gvec[elm, i, ineigth_j, iat] += t1[elm] - t3[elm]
                                gvec[elm, i, ineigth_k, iat] += t2[elm] + t3[elm]
                            end

                            # Stress
                            for elm2 = axes(stotv, 2)
                                for elm1 = axes(stotv, 1)
                                    stotv[elm1, elm2, i, iat] += (
                                        vij[elm1] * t1[elm2] +
                                        vik[elm1] * t2[elm2] +
                                        vjk[elm1] * t3[elm2]
                                    )
                                end
                            end
                        end
                        # Increment the feature index
                        i += 1
                    end
                end
            end  # 3body-feature update loop 
        end # i,j,k pair
    end
    workspace
end

"Compute the fv and gv for a given set of two and three body interactions"
function compute_fv_gv(
    features2,
    features3,
    cell::Cell;
    nl=NeighbourList(
        cell,
        maximum(x.rcut for x in (features2..., features3...));
        savevec=true,
    ),
    core=nothing,
    offset=0,
)

    # Generate the gradient workspace
    workspace = get_workspace(features2, features3, nl; core, offset)
    # Update for each atom
    for iat = 1:natoms(cell)
        compute_fv_gv_one!(workspace, features2, features3, iat, cell, nl; offset=offset)
    end
    workspace
end

"""
    compute_fv!(workspace::GradientWorkspace, f2, f3, cell::Cell; offset=0, kwargs...)

Compute the feature vectors for a cell.
"""
function compute_fv!(
    workspace::GradientWorkspace,
    f2,
    f3,
    cell::Cell;
    nl,
    offset=workspace.one_body_offset,
    kwargs...,
)
    reset!(workspace)
    for iat = 1:natoms(cell)
        compute_fv_gv_one!(workspace, f2, f3, iat, cell, nl; do_grad=false, offset, kwargs...)
    end
    workspace.fvec
end

"""
    compute_fv(f2, f3, cell::Cell; nl, kwargs...)

Compute the feature vectors for a cell.
"""
function compute_fv(f2, f3, cell::Cell; nl, kwargs...)
    workspace = get_workspace(f2, f3, nl, false; kwargs...)
    compute_fv!(workspace, f2, f3, cell; nl, kwargs...)
end

function get_workspace(
    features2,
    features3,
    nl,
    do_grad=true;
    fvec=nothing,
    core=nothing,
    offset=0,
)
    # Compute the maximum number of unique neighbours for each atom
    nn_max = maximum(x -> length(unique(x)), eachcol(nl.orig_indices)) + 1
    # Generate the feature vector
    if fvec === nothing
        _fvec = zeros(
            offset + sum(nfeatures.(features2)) + sum(nfeatures.(features3)),
            length(nl.nneigh),
        )
    else
        _fvec = fvec
    end
    GradientWorkspace(_fvec, nn_max; core, do_grad)
end

get_workspace(cf::CellFeature, nl, do_grad=true; fvec=nothing) =
    get_workspace(cf.two_body, cf.three_body, nl, do_grad; fvec)

function compute_fv_gv!(
    workspace::GradientWorkspace,
    features2,
    features3,
    cell::Cell;
    nl=NeighbourList(
        cell,
        maximum(x.rcut for x in (features2..., features3...));
        savevec=true,
    ),
    offset=workspace.one_body_offset,
)
    reset!(workspace)
    for iat = 1:natoms(cell)
        compute_fv_gv_one!(workspace, features2, features3, iat, cell, nl; offset=offset)
    end
    workspace
end

compute_fv_gv(cf::CellFeature, cell::Cell; kwargs...) =
    compute_fv_gv(cf.two_body, cf.three_body, cell; offset=length(cf.elements), kwargs...)

function _force_update!(workspace, gv; offset=workspace.one_body_offset)
    gvec = workspace.gvec
    fill!(workspace.forces, 0)
    for iv in axes(gvec, 4)
        for ia = 1:workspace.gvec_nn[iv]  # Number of unique neighbours and self whose graidents are stored
            iat = workspace.gvec_index[ia, iv]  # Translate to the actual atom index
            for i = 1+offset:size(gvec, 2)
                for elm in axes(workspace.forces, 1)
                    workspace.forces[elm, iat] -= gvec[elm, i, ia, iv] * gv[i, iv]
                end
            end
        end
    end
    workspace.tot_forces .= workspace.forces
    if !isnothing(workspace.hardcore.core)
        workspace.tot_forces .+= workspace.hardcore.fcore
    end
    _substract_force_drift(workspace.tot_forces)
end

"""
    _stress_update!(buffer::Array{T, 2}, sv, s) where {T}

Propagate chain rule to obtain the stress
"""
function _stress_update!(workspace::GradientWorkspace, gv; offset=workspace.one_body_offset)
    # Zero the buffer
    stotv = workspace.stotv
    fill!(workspace.stress, 0)
    for iat in axes(stotv, 4)
        for i = 1+offset:size(stotv, 3)
            for a in axes(workspace.stress, 1), b in axes(workspace.stress, 2)
                workspace.stress[a, b, iat] -= stotv[a, b, i, iat] * gv[i, iat]
            end
        end
    end
    workspace.tot_stress .= sum(workspace.stress, dims=3)
    if !isnothing(workspace.hardcore.core)
        workspace.tot_stress .+= sum(workspace.hardcore.score, dims=3)
    end
end

"""
    _substract_force_drift(forces)

    Substrate drift forces due to numerical erros.
"""
function _substract_force_drift(forces)
    drift = sum(forces, dims=2)
    maximum(drift) > 1e-4 && @warn "Drift forces are large: $drift"
    forces .-= drift ./ size(forces, 2)
end
