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
    isempty(features3) && return true
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
    @inbounds for i = 1:l
        feat = features3[i]
        ftmp = feat.f(r12, feat.rcut)
        # 1/f(rij)
        inv_fij[i] = 1.0 / ftmp
        @inbounds @simd for j = 1:length(feat.p)
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
    @inbounds for i = 1:l
        feat = features3[i]
        ftmp = feat.f(r12, feat.rcut)
        # 1/f(rij)
        inv_qji[i] = 1.0 / ftmp
        @inbounds @simd for j = 1:length(feat.q)
            qjk[j, i] = ftmp^feat.q[j]
        end
    end
end



"""
    _ensure_feat_buf!(workspace::GradientWorkspace, npmax3, nqmax3, lfe3, nl)

Lazily allocate per-atom feature computation buffers in the workspace.
Uses `nl.nmax` (max NL capacity) for the neighbour_indices buffer so no
per-atom resize is ever needed.  Thread-safe: index 0 = uninitialized,
push! a new FeatureBuffers on first use.
"""
function _ensure_feat_buf!(workspace::GradientWorkspace, npmax3, nqmax3, lfe3, nl::NeighbourList)
    isempty(workspace.feat_bufs) || return
    T = eltype(workspace.fvec)
    push!(workspace.feat_bufs, FeatureBuffers{T}(
        zeros(T, npmax3, lfe3),
        zeros(T, npmax3, lfe3),
        zeros(T, nqmax3, lfe3),
        zeros(T, lfe3),
        zeros(T, lfe3),
        zeros(T, lfe3),
        zeros(Int, nl.nmax),
    ))
end

"""
    compute_fv_gv_one!(workspace, features2, features3, iat, cell, nl; offset, do_grad)

Compute feature vector and gradients for a single atom.

This is the core computational kernel that computes local atomic environment
features for one atom, considering all its neighbors. This function handles
both two-body (pairwise) and three-body (triplet) interactions.

# Arguments
- `workspace::GradientWorkspace`: Pre-allocated workspace for computation
- `features2`: Vector or tuple of `TwoBodyFeature` specifications
- `features3`: Vector or tuple of `ThreeBodyFeature` specifications
- `iat::Int`: Index of the central atom to compute features for
- `cell::Cell`: Crystal structure
- `nl::NeighbourList`: Neighbour list (must have `has_vectors=true` if `do_grad=true`)

# Keyword Arguments
- `offset=workspace.one_body_offset`: Offset in feature vector for one-body features
- `do_grad=workspace.do_grad`: Whether to compute gradients

# Returns
- Nothing (modifies workspace in-place)

# Notes
- This function is called internally by `compute_fv!` and `compute_fv_gv!`
- Computes features by summing contributions from all neighbor pairs and triplets
- For gradients, neighbor list must be built with `savevec=true`
- Results are accumulated in `workspace.fvec` and `workspace.gvec`

# See also
[`compute_fv!`](@ref), [`compute_fv_gv!`](@ref), [`GradientWorkspace`](@ref)
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
    do_grad=workspace.do_grad,
)

    # NeighbourList must be built with vectors for gradient computation
    do_grad && (@assert nl.has_vectors)
    # Main quantities
    (; fvec, gvec, gvec_index, gvec_nn, stotv, hardcore) = workspace
    (; core_repulsion, fcore, score, ecore) = hardcore

    if do_grad
        gvec[:, :, :, iat] .= 0.0
        stotv[:, :, :, iat] .= 0.0
    end

    # Check if all features are the same - so powers does not need to be recalculated for each feature.
    same_3b = _is_same_pqrcutf(features3)
    lfe3::Int = isempty(features3) ? 0 : 1
    if !same_3b
        lfe3 = length(features3)
    end

    nfe2 = map(nfeatures, features2)
    totalfe2 = sum(nfe2)
    sym = species(cell)

    # Caches for three-body computation
    # The goal is to avoid ^ operator as much as possible by caching results
    npmax3 = isempty(features3) ? 0 : maximum(length(x.p) for x in features3)
    # All values of q
    nqmax3 = isempty(features3) ? 0 : maximum(length(x.q) for x in features3)

    # Maximum cut off 
    maxrcut = isempty(features3) ? maximum(x -> x.rcut, features2) :
              maximum(x -> x.rcut, (features3..., features2...))

    # Ensure pre-allocated per-atom feature computation buffers
    _ensure_feat_buf!(workspace, npmax3, nqmax3, lfe3, nl)
    bufs = workspace.feat_bufs[1]
    pij = bufs.pij
    inv_fij = bufs.inv_fij
    pik = bufs.pik
    inv_fik = bufs.inv_fik
    qjk = bufs.qjk
    inv_fjk = bufs.inv_fjk
    tiny_dist = 1e-14

    # In-place unique neighbour indices using pre-allocated buffer
    nidx = bufs.neighbour_indices
    n = nl.nneigh[iat]
    if n == 0
        neighbour_indices = view(nidx, 1:0)
    else
        for j = 1:n
            nidx[j] = nl.orig_indices[j, iat]
        end
        sort!(view(nidx, 1:n))
        k = 1
        for j = 2:n
            if nidx[j] != nidx[k]
                k += 1
                nidx[k] = nidx[j]
            end
        end
        neighbour_indices = view(nidx, 1:k)
    end
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

        modvij = vij / rij
        if !isempty(features3)
            _update_pij!(pij, inv_fij, rij, features3, same_3b)
        end

        # Add hard core repulsion
        if !isnothing(core_repulsion)
            _hard_core_update!(ecore, fcore, score, iat, jat, rij, vij, modvij, core_repulsion, sym[iat], sym[jat])
        end

        # Update two body features and forces
        i = 1 + offset

        for (_, f) in enumerate(features2)
            # Skip if this rij is not for this feature
            if !permequal(f.sij_idx, sym[iat], sym[jat])
                i += f.np
                continue
            end

            fij2 = f.f(rij, f.rcut)
            if do_grad
                gij = f.g(rij, f.rcut)
                @fastmath @inbounds for m = 1:length(f.p)
                    val = fast_pow(fij2, f.p[m])
                    fvec[i, iat] += val
                    if val != 0.0
                        gfij = f.p[m] * val / fij2 * gij
                    else
                        gfij = zero(val)
                    end
                    gvec[1, i, 1, iat] -= modvij[1] * gfij
                    gvec[2, i, 1, iat] -= modvij[2] * gfij
                    gvec[3, i, 1, iat] -= modvij[3] * gfij
                    gvec[1, i, ineigth_j, iat] += modvij[1] * gfij
                    gvec[2, i, ineigth_j, iat] += modvij[2] * gfij
                    gvec[3, i, ineigth_j, iat] += modvij[3] * gfij
                    stotv[1, 1, i, iat] += vij[1] * modvij[1] * gfij
                    stotv[1, 2, i, iat] += vij[1] * modvij[2] * gfij
                    stotv[1, 3, i, iat] += vij[1] * modvij[3] * gfij
                    stotv[2, 1, i, iat] += vij[2] * modvij[1] * gfij
                    stotv[2, 2, i, iat] += vij[2] * modvij[2] * gfij
                    stotv[2, 3, i, iat] += vij[2] * modvij[3] * gfij
                    stotv[3, 1, i, iat] += vij[3] * modvij[1] * gfij
                    stotv[3, 2, i, iat] += vij[3] * modvij[2] * gfij
                    stotv[3, 3, i, iat] += vij[3] * modvij[3] * gfij
                    i += 1
                end
            else
                @fastmath @inbounds for m = 1:length(f.p)
                    fvec[i, iat] += fast_pow(fij2, f.p[m])
                    i += 1
                end
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

            # Starting index for three-body feature update
            i = totalfe2 + offset + 1

            if do_grad
                # -- Full three-body with force + stress gradients --
                for (ife, f) in enumerate(features3)
                    if same_3b
                        ife = 1
                    end
                    if !permequal(f.sijk_idx, sym[iat], sym[jat], sym[kat])
                        i += f.np * f.nq
                        continue
                    end

                    rcut = f.rcut
                    gij = f.g(rij, rcut) * inv_fij[ife]
                    gik = f.g(rik, rcut) * inv_fik[ife]
                    gjk = f.g(rjk, rcut) * inv_fjk[ife]
                    gtmp1 = f.p .* gij
                    gtmp2 = f.p .* gik
                    gtmp3 = gjk .* f.q

                    for m = 1:f.np
                        ijkp = pij[m, ife] * pik[m, ife]
                        gscale1_m = gtmp1[m] * ijkp
                        gscale2_m = gtmp2[m] * ijkp
                        @fastmath @inbounds for o = 1:f.nq
                            val = ijkp * qjk[o, ife]
                            fvec[i, iat] += val
                            val == 0 && (i += 1; continue)

                            gs1 = gscale1_m * qjk[o, ife]
                            gs2 = gscale2_m * qjk[o, ife]
                            gs3 = gtmp3[o] * val

                            gvec[1, i, 1, iat] -= modvij[1]*gs1 + modvik[1]*gs2
                            gvec[2, i, 1, iat] -= modvij[2]*gs1 + modvik[2]*gs2
                            gvec[3, i, 1, iat] -= modvij[3]*gs1 + modvik[3]*gs2
                            gvec[1, i, ineigth_j, iat] += modvij[1]*gs1 - modvjk[1]*gs3
                            gvec[2, i, ineigth_j, iat] += modvij[2]*gs1 - modvjk[2]*gs3
                            gvec[3, i, ineigth_j, iat] += modvij[3]*gs1 - modvjk[3]*gs3
                            gvec[1, i, ineigth_k, iat] += modvik[1]*gs2 + modvjk[1]*gs3
                            gvec[2, i, ineigth_k, iat] += modvik[2]*gs2 + modvjk[2]*gs3
                            gvec[3, i, ineigth_k, iat] += modvik[3]*gs2 + modvjk[3]*gs3

                            stotv[1, 1, i, iat] += vij[1]*modvij[1]*gs1 + vik[1]*modvik[1]*gs2 + vjk[1]*modvjk[1]*gs3
                            stotv[1, 2, i, iat] += vij[1]*modvij[2]*gs1 + vik[1]*modvik[2]*gs2 + vjk[1]*modvjk[2]*gs3
                            stotv[1, 3, i, iat] += vij[1]*modvij[3]*gs1 + vik[1]*modvik[3]*gs2 + vjk[1]*modvjk[3]*gs3
                            stotv[2, 1, i, iat] += vij[2]*modvij[1]*gs1 + vik[2]*modvik[1]*gs2 + vjk[2]*modvjk[1]*gs3
                            stotv[2, 2, i, iat] += vij[2]*modvij[2]*gs1 + vik[2]*modvik[2]*gs2 + vjk[2]*modvjk[2]*gs3
                            stotv[2, 3, i, iat] += vij[2]*modvij[3]*gs1 + vik[2]*modvik[3]*gs2 + vjk[2]*modvjk[3]*gs3
                            stotv[3, 1, i, iat] += vij[3]*modvij[1]*gs1 + vik[3]*modvik[1]*gs2 + vjk[3]*modvjk[1]*gs3
                            stotv[3, 2, i, iat] += vij[3]*modvij[2]*gs1 + vik[3]*modvik[2]*gs2 + vjk[3]*modvjk[2]*gs3
                            stotv[3, 3, i, iat] += vij[3]*modvij[3]*gs1 + vik[3]*modvik[3]*gs2 + vjk[3]*modvjk[3]*gs3
                            i += 1
                        end
                    end
                end  # 3body-feature update loop
            else
                # -- Feature-only path (no gradients) --
                for (ife, f) in enumerate(features3)
                    if same_3b
                        ife = 1
                    end
                    if !permequal(f.sijk_idx, sym[iat], sym[jat], sym[kat])
                        i += f.np * f.nq
                        continue
                    end
                    @fastmath @inbounds for m = 1:f.np
                        ijkp = pij[m, ife] * pik[m, ife]
                        for o = 1:f.nq
                            fvec[i, iat] += ijkp * qjk[o, ife]
                            i += 1
                        end
                    end
                end
            end
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
    compute_fv!(workspace::GradientWorkspace, f2, f3, cell::Cell; nl, offset=0, kwargs...)

Compute feature vectors for all atoms in a cell (in-place, no gradients).

This function computes the local atomic environment descriptors (features) for
each atom in a structure. Features include two-body (pairwise) and three-body
(triplet) interactions. This version does NOT compute gradients and modifies
the workspace in-place for efficiency.

# Arguments
- `workspace::GradientWorkspace`: Pre-allocated workspace for feature computation
- `f2`: Vector or tuple of `TwoBodyFeature` specifications
- `f3`: Vector or tuple of `ThreeBodyFeature` specifications
- `cell::Cell`: Crystal structure to compute features for

# Keyword Arguments
- `nl`: NeighbourList for the cell (must be pre-computed)
- `offset=workspace.one_body_offset`: Offset for one-body features in feature vector
- Additional kwargs passed to `compute_fv_gv_one!`

# Returns
- Feature matrix (nfeatures × natoms) stored in `workspace.fvec`

# Examples
```julia
# Compute features for a carbon structure
cf = CellFeature([:C])
nl = NeighbourList(cell, 5.0)  # 5 Å cutoff
workspace = get_workspace(cf, nl, false)  # false = no gradients
fvec = compute_fv!(workspace, cf.two_body, cf.three_body, cell; nl=nl)

# fvec dimensions
println(size(fvec))  # (nfeatures, natoms)
```

# Notes
- This version is faster than `compute_fv_gv!` as it skips gradient computation
- Workspace is reused across calls for efficiency
- For one-time computation, use `compute_fv` convenience function instead

# See also
[`compute_fv`](@ref), [`compute_fv_gv!`](@ref), [`get_workspace`](@ref), [`GradientWorkspace`](@ref)
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
        compute_fv_gv_one!(
            workspace,
            f2,
            f3,
            iat,
            cell,
            nl;
            do_grad=false,
            offset,
            kwargs...,
        )
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

"""
    get_workspace(features2, features3, nl, do_grad=true; fvec=nothing, core=nothing, offset=0)

Allocate a workspace for efficient feature and gradient computation.

Creates a `GradientWorkspace` object that pre-allocates all arrays needed for
feature vector and gradient calculations. Reusing workspaces across multiple
calculations avoids memory allocation overhead.

# Arguments
- `features2`: Vector or tuple of `TwoBodyFeature` specifications
- `features3`: Vector or tuple of `ThreeBodyFeature` specifications
- `nl`: NeighbourList for the structure
- `do_grad=true`: Whether to allocate arrays for gradient computation

# Keyword Arguments
- `fvec=nothing`: Pre-allocated feature vector array (if `nothing`, creates new array)
- `core=nothing`: Hard-core repulsion potential (optional)
- `offset=0`: Offset for one-body features in the feature vector

# Returns
- `GradientWorkspace`: Workspace object containing pre-allocated arrays

# Examples
```julia
# Create workspace for gradient computation
cf = CellFeature([:C])
nl = NeighbourList(cell, 5.0; savevec=true)
workspace = get_workspace(cf.two_body, cf.three_body, nl, true)

# Reuse workspace for multiple structures
for cell in structures
    compute_fv_gv!(workspace, cf.two_body, cf.three_body, cell; nl=nl)
    # Process features and gradients...
end

# Create workspace without gradients (faster for prediction)
workspace_no_grad = get_workspace(cf.two_body, cf.three_body, nl, false)
```

# Notes
- Set `do_grad=false` when only computing features (prediction), not forces
- Workspace size depends on maximum number of neighbors and feature dimensions
- Reusing workspaces significantly improves performance for batch computations

# See also
[`GradientWorkspace`](@ref), [`compute_fv!`](@ref), [`compute_fv_gv!`](@ref)
"""
function get_workspace(
    features2,
    features3,
    nl,
    do_grad=true;
    fvec=nothing,
    core=nothing,
    offset=0,
)
    # Generate the feature vector
    if fvec === nothing
        nfe2 = isempty(features2) ? 0 : sum(nfeatures, features2)
        nfe3 = isempty(features3) ? 0 : sum(nfeatures, features3)
        _fvec = zeros(
            offset + nfe2 + nfe3,
            length(nl.nneigh),
        )
    else
        _fvec = fvec
    end
    ndims = length(nl.ea.positions[1])
    # nn_max is only needed to size gradient arrays — skip the expensive
    # unique-per-column scan when not computing gradients
    nn_max = do_grad ? maximum(x -> length(unique(x)), eachcol(nl.orig_indices)) + 1 : 1
    GradientWorkspace(_fvec, nn_max; core, do_grad, ndims)
end

get_workspace(cf::CellFeature, nl, do_grad=true; fvec=nothing) =
    get_workspace(cf.two_body, cf.three_body, nl, do_grad; fvec)

function compute_fv!(
    workspace::GradientWorkspace,
    cf::CellFeature,
    cell::Cell;
    nl,
    offset=workspace.one_body_offset,
    kwargs...,
)
    compute_fv!(workspace, cf.two_body, cf.three_body, cell; nl, offset, kwargs...)
end

"""
    compute_fv_gv!(workspace::GradientWorkspace, features2, features3, cell::Cell; nl, offset=0)

Compute feature vectors AND gradients for all atoms in a cell (in-place).

This function computes both the local atomic environment features and their
gradients with respect to atomic positions. Gradients are essential for
computing forces during molecular dynamics or structure optimization.

# Arguments
- `workspace::GradientWorkspace`: Pre-allocated workspace (must have `do_grad=true`)
- `features2`: Vector or tuple of `TwoBodyFeature` specifications
- `features3`: Vector or tuple of `ThreeBodyFeature` specifications
- `cell::Cell`: Crystal structure to compute features and gradients for

# Keyword Arguments
- `nl=NeighbourList(...)`: NeighbourList for the cell (default: auto-created with `savevec=true`)
- `offset=workspace.one_body_offset`: Offset for one-body features

# Returns
- Modified `workspace` containing:
  - `workspace.fvec`: Feature vectors (nfeatures × natoms)
  - `workspace.gvec`: Feature gradients (3 × nfeatures × nneigh × natoms)

# Examples
```julia
# Compute features and gradients for force calculation
cf = CellFeature([:C])
nl = NeighbourList(cell, 5.0; savevec=true)  # savevec=true required for gradients
workspace = get_workspace(cf, nl, true)  # true = compute gradients

# Compute features and gradients
compute_fv_gv!(workspace, cf.two_body, cf.three_body, cell; nl=nl)

# Access results
fvec = workspace.fvec  # Features
gvec = workspace.gvec  # Gradients (for force calculations)
```

# Notes
- Neighbour list MUST have `savevec=true` for gradient computation
- Significantly more expensive than `compute_fv!` (which skips gradients)
- Gradients are used to compute forces via chain rule with neural network gradients
- Workspace must be created with `do_grad=true`

# See also
[`compute_fv!`](@ref), [`get_workspace`](@ref), [`GradientWorkspace`](@ref)
"""
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

function compute_fv_gv!(
    workspace::GradientWorkspace,
    cf::CellFeature,
    cell::Cell;
    nl=NeighbourList(cell, suggest_rcut(cf; shell=0.0); savevec=true),
    offset=workspace.one_body_offset,
)
    compute_fv_gv!(workspace, cf.two_body, cf.three_body, cell; nl, offset)
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
    if !isnothing(workspace.hardcore.core_repulsion)
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
    if !isnothing(workspace.hardcore.core_repulsion)
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
