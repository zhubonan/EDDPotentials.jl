#=
Code related to taking gradients
=#
using StaticArrays
using Base.Threads
using LinearAlgebra
using LoopVectorization

include("repulsive_core.jl")
include("workspace.jl")

function similar_zero(x)
    y = similar(x)
    fill!(y, 0)
    y
end

function _atom_chunks(natoms; ngps=10)
    nt = Threads.nthreads()
    nchunks = nt * ngps
    nper = div(natoms, nchunks)
    if nper == 0
        return [i:i for i = 1:natoms]
    end
    out = UnitRange{Int}[]
    i = 1
    for _ = 1:nchunks
        push!(out, i:i+nper-1)
        i += nper
    end
    m = mod(natoms, nchunks)
    if m != 0
        push!(out, natoms-m+1:natoms)
    end
    out
end

"""
Populate a Matrix with the content of the first column
"""
function _populate_with_first_column(arr)
    m, n = size(arr)
    for i = 2:n
        for j = 1:m
            arr[j, i] = arr[j, 1]
        end
    end
end

"""
    _populate_feat_bufs!(workspace::GradientWorkspace, cf::CellFeature, nl)

Eagerly populate per-thread `FeatureBuffers` in the workspace. Called once from
`get_workspace` before any threaded access, so no locking is needed.
"""
function _populate_feat_bufs!(workspace::GradientWorkspace, cf::CellFeature{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, NP3, NQ3}, nl::NeighbourList) where {NP3, NQ3}
    isempty(workspace.feat_bufs) || return
    lfe3 = NP3 == 0 ? 0 : 1
    npmax3 = Int(NP3)
    nqmax3 = Int(NQ3)
    T = eltype(workspace.fvec)
    ngps = 10
    nslots = Threads.nthreads() * ngps + 1
    max_atom_id = size(nl.orig_indices, 2)
    ndims = size(nl.ea.positions, 1)
    for _ = 1:nslots
        push!(workspace.feat_bufs, FeatureBuffers{T}(
            zeros(T, npmax3, lfe3),
            zeros(T, npmax3, lfe3),
            zeros(T, nqmax3, lfe3),
            zeros(T, lfe3),
            zeros(T, lfe3),
            zeros(T, lfe3),
            zeros(Int, nl.nmax),
            zeros(Int, max_atom_id + 1),
            zeros(T, npmax3),
            zeros(T, npmax3 * nqmax3, ndims),
            zeros(T, npmax3 * nqmax3, ndims),
            zeros(T, npmax3 * nqmax3, ndims),
            zeros(T, npmax3 * nqmax3),
            zeros(T, npmax3 * nqmax3),
            zeros(T, npmax3 * nqmax3),
        ))
    end
end

function _ensure_feat_buf!(workspace::GradientWorkspace, npmax3, nqmax3, lfe3, nl::NeighbourList)
    isempty(workspace.feat_bufs) || return
    lock(workspace.feat_bufs_lock) do
        isempty(workspace.feat_bufs) || return
        T = eltype(workspace.fvec)
        ngps = 10
        nslots = Threads.nthreads() * ngps + 1
        max_atom_id = size(nl.orig_indices, 2)
        ndims = size(nl.ea.positions, 1)
        for _ = 1:nslots
            push!(workspace.feat_bufs, FeatureBuffers{T}(
                zeros(T, npmax3, lfe3),
                zeros(T, npmax3, lfe3),
                zeros(T, nqmax3, lfe3),
                zeros(T, lfe3),
                zeros(T, lfe3),
                zeros(T, lfe3),
                zeros(Int, nl.nmax),
                zeros(Int, max_atom_id + 1),
                zeros(T, npmax3),
                zeros(T, npmax3 * nqmax3, ndims),
                zeros(T, npmax3 * nqmax3, ndims),
                zeros(T, npmax3 * nqmax3, ndims),
                zeros(T, npmax3 * nqmax3),
                zeros(T, npmax3 * nqmax3),
                zeros(T, npmax3 * nqmax3),
            ))
        end
    end
end

function _precompute_3b_factors!(
    ffact_center, ffact_j, ffact_k,
    modvij, modvik, modvjk,
    gtmp1, gtmp2, gtmp3,
    ::Val{np}, ::Val{nq},
    ::Cell{F,D},
) where {F, D, np, nq}
    for axis in 1:D
        for m = 1:np
            im = (m - 1) * nq
            ftmp = -(modvij[axis]*gtmp1[m] + modvik[axis]*gtmp2[m])
            @turbo for o in 1:nq
                ffact_center[im + o, axis] = ftmp
            end
        end
    end
    # for axis = 1:D
    #     @turbo for nn = 0:(np * nq - 1)
    #         m = div(nn, nq) + 1
    #         o = (nn % nq) + 1
    #         tmp1 = modvjk[axis] * gtmp3[o]
    #         ffact_j[nn + 1, axis] = modvij[axis] * gtmp1[m] - tmp1
    #         ffact_k[nn + 1, axis] = modvik[axis] * gtmp2[m] + tmp1
    #     end
    # end
    for axis in 1:D
        @inbounds @fastmath for m = 1:np
            g1 = gtmp1[m] * modvij[axis]
            g2 = gtmp2[m] * modvik[axis]
            im = (m - 1) * nq
            @turbo for o = 1:nq
                ffact_k[im + o, axis] = g2 + gtmp3[o] * modvjk[axis]
                ffact_j[im + o, axis] = g1 - gtmp3[o] * modvjk[axis]
            end
        end
    end
end

"""
    compute_fv_gv_one!(workspace, cf::CellFeature, iat, cell, nl; offset, do_grad, atom_species_ints)

Compute feature vector and gradients for a single atom using CellFeature directly.
"""
function compute_fv_gv_one!(
    workspace::GradientWorkspace,
    cf::CellFeature{<:Any, <:Any, <:Any, <:Any, <:Any, NELEM, NP2, NP3, NQ3},
    iat::Int,
    cell::Cell{F,D},
    nl::NeighbourList,
    ;
    offset=workspace.one_body_offset,
    do_grad=workspace.do_grad,
    atom_species_ints,
    _bufs=nothing,
    use_spline=true,
) where {NELEM, NP2, NP3, NQ3, F, D}

    do_grad && (@assert nl.has_vectors)
    (; fvec, gvec, gvec_index, gvec_nn, stotv, hardcore) = workspace
    (; core_repulsion, fcore, score, ecore) = hardcore

    if do_grad
        gvec[:, :, :, iat] .= 0.0
        stotv[:, :, :, iat] .= 0.0
    end

    has_3b = !(NP3 == 0 || NQ3 == 0)

    sym = species(cell)

    npmax3 = Int(NP3)
    nqmax3 = Int(NQ3)
    lfe3 = has_3b ? 1 : 0

    maxrcut = has_3b ? max(cf.rcut2, cf.rcut3) : cf.rcut2
    maxrcut_sq = maxrcut * maxrcut

    # Ensure pre-allocated per-atom feature computation buffers
    if _bufs !== nothing
        bufs = _bufs
    else
        _ensure_feat_buf!(workspace, npmax3, nqmax3, lfe3, nl)
        bufs = workspace.feat_bufs[Threads.threadid()]
    end
    tiny_dist = 1e-14

    # In-place unique neighbour indices using pre-allocated buffer
    nidx = bufs.neighbour_indices
    revmap = bufs.reverse_neighbour_map
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
        for pos = 1:k
            revmap[neighbour_indices[pos]] = pos
        end
    end
    if do_grad
        gvec_index[1, iat] = iat
        gvec_index[2:length(neighbour_indices)+1, iat] .= neighbour_indices
        gvec_nn[iat] = length(neighbour_indices) + 1
    end

    ci = atom_species_ints[iat]
    total_fe2 = NELEM^2 * NP2
    tbfe3 = NP3 * NQ3

    # Start main loop
    for (jat, jextend, rij, vij) in CellBase.eachneighbourvector(nl, iat)
        rij > maxrcut && continue
        @assert rij > tiny_dist

        ineigth_j = revmap[jat] + 1
        modvij = vij / rij

        # Add hard core repulsion
        if !isnothing(core_repulsion)
            _hard_core_update!(ecore, fcore, score, iat, jat, rij, vij, modvij, core_repulsion, sym[iat], sym[jat])
        end

        # ─── Two-body features ───
        if NP2 > 0
            cj = atom_species_ints[jat]
            ch2 = cf.two_body_channel[ci, cj]
            i2 = offset + (ch2 - 1) * NP2 + 1
            i21 = i2 - 1
            if use_spline
                fijp2, fij2 = evaluate_p2(cf, rij)
                fij2 == 0 && continue
            else
                fij2 = cf.f2(rij, cf.rcut2)
                fij2 == 0 && continue
                fijp2 = fij2 .^ cf.p2
            end
            @turbo for m in 1:NP2
                fvec[i21 + m, iat] += fijp2[m]
            end
            if do_grad
                gij = cf.g2(rij, cf.rcut2)
                @fastmath gfij = cf.p2 .* fijp2 ./ fij2 .* gij
                for dir in 1:D
                    @turbo for m in 1:NP2
                        gvec[i21 + m, dir, 1, iat] -= modvij[dir] * gfij[m]
                    end
                    @turbo for m in 1:NP2
                        gvec[i21 + m, dir, ineigth_j, iat] += modvij[dir] * gfij[m]
                    end
                end
                outer2 = vij * modvij'
                for jj in 1:3
                    for ii in 1:3
                        @turbo for m in 1:NP2
                            stotv[i21 + m, ii, jj, iat] += outer2[ii, jj] * gfij[m]
                        end
                    end
                end
            end
        end

        # ─── Three-body features ───
        has_3b || continue

        # Pre-compute three-body pij factors
        if use_spline
            pij, ftmp = evaluate_p3(cf, rij)
            ftmp == 0 && continue
        else
            ftmp = cf.f3(rij, cf.rcut3)
            ftmp == 0 && continue
            pij = ftmp .^ cf.p3
        end
        inv_fij = 1.0 / ftmp

        (;gs1, gs2, gs3) = bufs

        for (kat, kextend, rik, vik) in CellBase.eachneighbourvector(nl, iat)
            rik > maxrcut && continue
            @assert rik > tiny_dist

            if kextend <= jextend
                continue
            end

            vjk = nl.ea.positions[kextend] .- nl.ea.positions[jextend]
            rjk_sq = sum(vjk .^ 2)
            rjk_sq > maxrcut_sq && continue
            rjk = sqrt(rjk_sq)
            rjk > maxrcut && continue
            @assert rjk > tiny_dist

            ineigth_k = revmap[kat] + 1

            modvik = vik / rik
            modvjk = vjk / rjk

            # Compute pik
            if use_spline
                pik, ftmp = evaluate_p3(cf, rik)
                ftmp == 0 && continue
            else
                ftmp = cf.f3(rik, cf.rcut3)
                ftmp == 0 && continue
                pik = ftmp .^ cf.p3
            end
            inv_fik = 1.0 / ftmp

            # Compute qjk
            if use_spline
                qjk, ftmp = evaluate_q3(cf, rjk)
                ftmp == 0 && continue
            else
                ftmp = cf.f3(rjk, cf.rcut3)
                ftmp == 0 && continue
                qjk = ftmp .^ cf.q3
            end
            inv_fjk = 1.0 / ftmp

            # Resolve the channel
            cj = atom_species_ints[jat]
            ck = atom_species_ints[kat]
            ch3 = cf.three_body_channel[ci, cj, ck]
            ch3 == 0 && continue

            i3 = offset + total_fe2 + (ch3 - 1) * tbfe3 + 1
            i31 = i3 - 1

            # Feature vector update
            ijkp_buf = pij .* pik
            for m = 1:NP3
                ijkp_m = ijkp_buf[m]
                oo = (m-1) * NQ3 + i31
                @inbounds @simd for o = 1:NQ3
                    fvec[oo + o, iat] += qjk[o] * ijkp_m
                end
            end

            if do_grad
                gij = cf.g3(rij, cf.rcut3) * inv_fij
                gik = cf.g3(rik, cf.rcut3) * inv_fik
                gjk = cf.g3(rjk, cf.rcut3) * inv_fjk
                gtmp1 = cf.p3 .* gij
                gtmp2 = cf.p3 .* gik
                gtmp3 = cf.q3 .* gjk

                _precompute_3b_factors!(
                    bufs.ffact_center, bufs.ffact_j, bufs.ffact_k,
                    modvij, modvik, modvjk,
                    gtmp1, gtmp2, gtmp3,
                    Val(NP3), Val(NQ3),
                    cell,
                )
                # Forces
                i31 = i3 - 1
                _gvec_update!(gvec, iat, i31, ineigth_j, ineigth_k, ijkp_buf, qjk, bufs.ffact_center, 
                              bufs.ffact_j, bufs.ffact_k, Val(D), Val(NP3), Val(NQ3))

                # Precompute stress factors
                _precompute_gs!(gs1, gs2, gs3, ijkp_buf, qjk, gtmp1, gtmp2, gtmp3, Val(NP3), Val(NQ3))

                # Stress update TODO: use virial for large cell
                outerij = vij * modvij'
                outerik = vik * modvik'
                outerjk = vjk * modvjk'
                _update_stotv!(stotv, i31, iat, outerij, outerik, outerjk, gs1, gs2, gs3, Val(NP3), Val(NQ3))
            end
        end # i,j,k pair
    end
    workspace
end

# ─── Wrapper: compute_fv_gv (standalone) ───────────────────────────────────────

function compute_fv_gv(
    cf::CellFeature,
    cell::Cell;
    nl=NeighbourList(cell, suggest_rcut(cf; shell=0.0); savevec=true),
    core=nothing,
    offset=nothing,
    use_spline=true,
)
    _offset = isnothing(offset) ? one_body_offset(cf) : offset
    workspace = get_workspace(cf, nl; core, offset=_offset)
    atom_species_ints = [cf.spec_to_idx[s] for s in species(cell)]
    _populate_feat_bufs!(workspace, cf, nl)
    nat = natoms(cell)
    chunks = _atom_chunks(nat)
    Threads.@threads for ci in eachindex(chunks)
        bufs = workspace.feat_bufs[ci]
        for iat in chunks[ci]
            compute_fv_gv_one!(
                workspace, cf, iat, cell, nl;
                offset=_offset, atom_species_ints=atom_species_ints, _bufs=bufs,
                use_spline=use_spline,
            )
        end
    end
    workspace
end

# ─── Wrapper: compute_fv! (workspace, CellFeature) ─────────────────────────────

function compute_fv!(
    workspace::GradientWorkspace,
    cf::CellFeature,
    cell::Cell;
    nl,
    offset=workspace.one_body_offset,
    use_spline=true,
    kwargs...,
)
    reset!(workspace)
    atom_species_ints = [cf.spec_to_idx[s] for s in species(cell)]
    _populate_feat_bufs!(workspace, cf, nl)
    nat = natoms(cell)
    chunks = _atom_chunks(nat)
    Threads.@threads for ci in eachindex(chunks)
        bufs = workspace.feat_bufs[ci]
        for iat in chunks[ci]
            compute_fv_gv_one!(
                workspace,
                cf,
                iat,
                cell,
                nl;
                do_grad=false,
                offset,
                atom_species_ints=atom_species_ints,
                _bufs=bufs,
                use_spline=use_spline,
                kwargs...,
            )
        end
    end
    workspace.fvec
end

# ─── Wrapper: compute_fv (standalone) ──────────────────────────────────────────

function compute_fv(cf::CellFeature, cell::Cell; nl, kwargs...)
    workspace = get_workspace(cf, nl, false; kwargs...)
    compute_fv!(workspace, cf, cell; nl, kwargs...)
end

# ─── Wrapper: get_workspace (CellFeature) ──────────────────────────────────────

function get_workspace(
    cf::CellFeature,
    nl,
    do_grad=true;
    fvec=nothing,
    core=nothing,
    offset=0,
)
    if fvec === nothing
        nfe = nfeatures(cf)
        _fvec = zeros(
            nfe,
            length(nl.nneigh),
        )
    else
        _fvec = fvec
    end
    ndims = length(nl.ea.positions[1])
    nn_max = do_grad ? maximum(x -> length(unique(x)), eachcol(nl.orig_indices)) + 1 : 1
    ws = GradientWorkspace(_fvec, nn_max; core, do_grad, ndims, one_body_offset=offset)
    _populate_feat_bufs!(ws, cf, nl)
    ws
end

# ─── Wrapper: compute_fv_gv! (workspace, CellFeature) ──────────────────────────

function compute_fv_gv!(
    workspace::GradientWorkspace,
    cf::CellFeature,
    cell::Cell;
    nl=NeighbourList(cell, suggest_rcut(cf; shell=0.0); savevec=true),
    offset=workspace.one_body_offset,
    use_spline=true,
)
    reset!(workspace)
    atom_species_ints = [cf.spec_to_idx[s] for s in species(cell)]
    _populate_feat_bufs!(workspace, cf, nl)
    nat = natoms(cell)
    chunks = _atom_chunks(nat)
    Threads.@threads for ci in eachindex(chunks)
        bufs = workspace.feat_bufs[ci]
        for iat in chunks[ci]
            compute_fv_gv_one!(
                workspace, cf, iat, cell, nl;
                offset=offset, atom_species_ints=atom_species_ints, _bufs=bufs,
                use_spline=use_spline,
            )
        end
    end
    workspace
end

# ─── Force and stress update (unchanged) ───────────────────────────────────────

function _force_update!(workspace, gv; offset=workspace.one_body_offset)
    gvec = workspace.gvec
    fill!(workspace.forces, 0)
    for iv in axes(gvec, 4)
        for ia = 1:workspace.gvec_nn[iv]
            iat = workspace.gvec_index[ia, iv]
            for i = 1+offset:size(gvec, 1)
                for elm in axes(workspace.forces, 1)
                    workspace.forces[elm, iat] -= gvec[i, elm, ia, iv] * gv[i, iv]
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

function _stress_update!(workspace::GradientWorkspace, gv; offset=workspace.one_body_offset)
    stotv = workspace.stotv
    fill!(workspace.stress, 0)
    for iat in axes(stotv, 4)
        for i = 1+offset:size(stotv, 1)
            for a in axes(workspace.stress, 1), b in axes(workspace.stress, 2)
                workspace.stress[a, b, iat] -= stotv[i, a, b, iat] * gv[i, iat]
            end
        end
    end
    workspace.tot_stress .= sum(workspace.stress, dims=3)
    if !isnothing(workspace.hardcore.core_repulsion)
        workspace.tot_stress .+= sum(workspace.hardcore.score, dims=3)
    end
end

function _substract_force_drift(forces)
    drift = sum(forces, dims=2)
    maximum(drift) > 1e-4 && @warn "Drift forces are large: $drift"
    forces .-= drift ./ size(forces, 2)
end

function _precompute_gs!(gs1, gs2, gs3, ijkp_buf, qjk, gtmp1, gtmp2, gtmp3, ::Val{NP3}, ::Val{NQ3}) where {NP3, NQ3}
    @fastmath @inbounds for m = 1:NP3
        ijkp = ijkp_buf[m]
        gscale1_m = gtmp1[m] * ijkp
        gscale2_m = gtmp2[m] * ijkp
        mm = (m  - 1) * NQ3 
        @turbo for o = 1:NQ3
            gs1[mm + o] = gscale1_m * qjk[o]
            gs2[mm + o] = gscale2_m * qjk[o]
            gs3[mm + o] = gtmp3[o] * ijkp * qjk[o]
        end
    end
end

function _update_stotv!(stotv, i31, iat, outerij, outerik, outerjk, gs1, gs2, gs3, ::Val{NP3}, ::Val{NQ3}) where {NP3, NQ3}
    for ii in 1:3
        for jj in 1:3
            @turbo for nn in 1:(NP3 * NQ3)
                stotv[i31 + nn, jj, ii, iat] += outerij[jj, ii] * gs1[nn] + outerik[jj, ii] * gs2[nn] + outerjk[jj, ii] * gs3[nn]
            end
        end
    end
end

function _gvec_update!(gvec, iat, i31, ineigth_j, ineigth_k, ijkp_buf, qjk, ffact_center, ffact_j, ffact_k, ::Val{D}, ::Val{NP3}, ::Val{NQ3}) where {D, NP3, NQ3}
    vout = qjk * ijkp_buf' 
    for dir in 1:D
        @turbo for nn = 1:(NP3 * NQ3)
            gvec[i31 + nn, dir, 1, iat] += vout[nn] * ffact_center[nn, dir]
        end
        @turbo for nn = 1:(NP3 * NQ3)
            gvec[i31 + nn, dir, ineigth_j, iat] += vout[nn] * ffact_j[nn, dir]
        end
        @turbo for nn = 1:(NP3 * NQ3)
            gvec[i31 + nn, dir, ineigth_k, iat] += vout[nn] * ffact_k[nn, dir]
        end
    end
end
