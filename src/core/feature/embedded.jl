import Random
import LinearAlgebra

struct RandomEmbeddedCellFeature{T,V,W,CF,X} <: AbstractCellFeatureSpec
    elements::Vector{Symbol}
    embedding::Matrix{T}
    pair_channels::V
    triplet_channels::W
    cell_feature::CF
    constructor_kwargs::X
end

function Base.:(==)(A::RandomEmbeddedCellFeature, B::RandomEmbeddedCellFeature)
    for name in fieldnames(typeof(A))
        if getproperty(A, name) != getproperty(B, name)
            return false
        end
    end
    return true
end

function _normalize_embedded_sequences(p2, p3, q3; geometry_sequence=false)
    if geometry_sequence
        p2 = genp(p2)
        p3 = genp(p3)
        q3 = genp(q3)
    end
    Tuple(p2), Tuple(p3), Tuple(q3)
end

function _embedded_basis(
    elements;
    p2=2:8,
    p3=p2,
    q3=p3,
    rcut2=4.0,
    rcut3=rcut2,
    f2=fr,
    g2=gfr,
    f3=fr,
    g3=gfr,
    geometry_sequence=false,
    spline_ngrid::Int=3000,
)
    elements = sort(unique(map(Symbol, elements)))
    isempty(elements) && throw(ArgumentError("RandomEmbeddedCellFeature requires at least one element."))
    basis = CellFeature(
        elements;
        p2,
        p3,
        q3,
        rcut2,
        rcut3,
        f2,
        g2,
        f3,
        g3,
        geometry_sequence,
        spline_ngrid,
    )
    elements, basis
end

function _random_embedding_matrix(T, nelem::Int, nembed::Int; seed::Integer=0, optimize::Bool=true)
    rng = Random.MersenneTwister(seed)

    if !optimize
        # Fast path — random Gaussian with row scaling, no separation optimization
        weight = randn(rng, T, nelem, nembed)
        for i in 1:nelem
            scale = T(0.5) + rand(rng, T)
            @views weight[i, :] .*= scale / sqrt(sum(abs2, weight[i, :]))
        end
        return weight
    end

    if nelem <= nembed
        # Orthogonal vectors via QR decomposition for maximum separation
        raw = randn(rng, T, nembed, nembed)
        Q = Matrix{T}(LinearAlgebra.qr(raw).Q)
        weight = Q[1:nelem, :]
    else
        # More elements than dimensions — repulsion on the unit sphere
        weight = randn(rng, T, nelem, nembed)
        for i in 1:nelem
            nrm = sqrt(sum(abs2, @view weight[i, :]))
            nrm > 0 && (@views weight[i, :] ./= nrm)
        end
        step = T(0.1)
        for _ in 1:50
            for i in 1:nelem
                force = zeros(T, nembed)
                for j in 1:nelem
                    i == j && continue
                    cos_ij = sum(@views weight[i, :] .* weight[j, :])
                    @views force .+= cos_ij .* weight[j, :]
                end
                @views weight[i, :] .-= step .* force
                nrm = sqrt(sum(abs2, @view weight[i, :]))
                nrm > 1e-10 && (@views weight[i, :] ./= nrm)
            end
        end
    end

    # Scale each row to a random length in [0.5, 1.5)
    for i in 1:nelem
        scale = T(0.5) + rand(rng, T)
        @views weight[i, :] .*= scale
    end
    weight
end

function _pair_channels(nembed::Int)
    channels = Tuple{Int,Int}[]
    for a = 1:nembed
        for b = 1:nembed
            push!(channels, (a, b))
        end
    end
    Tuple(channels)
end

function _unordered_channels(nembed::Int)
    channels = Tuple{Int,Int}[]
    for a = 1:nembed
        for b = a:nembed
            push!(channels, (a, b))
        end
    end
    Tuple(channels)
end

function _triplet_channels(nembed::Int)
    channels = Tuple{Int,Int,Int}[]
    for a = 1:nembed
        for (b, c) in _unordered_channels(nembed)
            push!(channels, (a, b, c))
        end
    end
    Tuple(channels)
end

function RandomEmbeddedCellFeature(
    elements;
    nembed::Int=4,
    seed::Integer=0,
    T::DataType=Float64,
    optimize_embedding::Bool=true,
    p2=2:8,
    p3=p2,
    q3=p3,
    rcut2=4.0,
    rcut3=rcut2,
    f2=fr,
    g2=gfr,
    f3=fr,
    g3=gfr,
    geometry_sequence=false,
    spline_ngrid::Int=3000,
)
    proto_elements, basis = _embedded_basis(
        elements;
        p2,
        p3,
        q3,
        rcut2,
        rcut3,
        f2,
        g2,
        f3,
        g3,
        geometry_sequence,
        spline_ngrid,
    )
    embedding = _random_embedding_matrix(T, length(proto_elements), nembed; seed, optimize=optimize_embedding)
    stored_kwargs = (
        ;
        nembed,
        seed,
        T,
        p2=Tuple(basis.p2),
        p3=Tuple(basis.p3),
        q3=Tuple(basis.q3),
        rcut2=basis.rcut2,
        rcut3=basis.rcut3,
        geometry_sequence,
        spline_ngrid,
    )
    RandomEmbeddedCellFeature(
        proto_elements,
        embedding,
        _pair_channels(nembed),
        _triplet_channels(nembed),
        basis,
        stored_kwargs,
    )
end

Base.show(io::IO, x::RandomEmbeddedCellFeature) = Base.show(io, MIME("text/plain"), x)

function Base.show(io::IO, ::MIME"text/plain", x::RandomEmbeddedCellFeature)
    println(io, "$(typeof(x))")
    println(io, "  Elements:")
    println(io, "    $(x.elements)")
    println(io, "  Embedded channels: $(embedding_size(x))")
    println(io, "  Pair channels: $(length(x.pair_channels))")
    println(io, "  Triplet channels: $(length(x.triplet_channels))")
    println(io, "  Basis CellFeature:")
    println(io, "    p2=$(Tuple(x.cell_feature.p2)), p3=$(Tuple(x.cell_feature.p3)), q3=$(Tuple(x.cell_feature.q3))")
    println(io, "    rcut2=$(x.cell_feature.rcut2), rcut3=$(x.cell_feature.rcut3)")
end

embedding_size(cf::RandomEmbeddedCellFeature) = size(cf.embedding, 2)

nfeatures_per_channel(cf::RandomEmbeddedCellFeature, body) = 
    body == 2 ? nfeatures_per_channel(cf.cell_feature, 2) :
    body == 3 ? nfeatures_per_channel(cf.cell_feature, 3) : 0

nchannels(cf::RandomEmbeddedCellFeature, body) =
    body == 2 ? length(cf.pair_channels) :
    body == 3 ? length(cf.triplet_channels) : 0

"""
    one_body_offset(cf)

Return the offset for one-body features in the force/stress chain rule.
For single embedding, one-body features are contiguous at the start.
"""
one_body_offset(cf::RandomEmbeddedCellFeature) = embedding_size(cf)

function _species_index_map(elements)
    Dict(s => i for (i, s) in enumerate(elements))
end

_atom_species_ints(cf::RandomEmbeddedCellFeature, cell::Cell) = [cf.cell_feature.spec_to_idx[s] for s in species(cell)]

function feature_size(cf::RandomEmbeddedCellFeature)
    nembed = embedding_size(cf)
    n2proto = nfeatures_per_channel(cf, 2)
    n3proto = nfeatures_per_channel(cf, 3)
    (
        nembed,
        n2proto * length(cf.pair_channels),
        n3proto * length(cf.triplet_channels),
    )
end

nfeatures(cf::RandomEmbeddedCellFeature) = sum(feature_size(cf))

function suggest_rcut(cf::RandomEmbeddedCellFeature; shell=1.0)
    suggest_rcut(cf.cell_feature; shell)
end

function one_body_vectors!(v::AbstractMatrix, cell::Cell, cf::RandomEmbeddedCellFeature; offset=0)
    spec_index = cf.cell_feature.spec_to_idx
    for (iat, s) in enumerate(species(cell))
        ispec = spec_index[s]
        @views v[offset+1:offset+embedding_size(cf), iat] .= cf.embedding[ispec, :]
    end
    v
end

function feature_vector(cf::RandomEmbeddedCellFeature, cell::Cell; nmax=500, skin=1.0)
    rcut = suggest_rcut(cf; shell=skin)
    nl = NeighbourList(cell, rcut, nmax; savevec=true)
    vecs = zeros(sum(feature_size(cf)), length(cell))
    wk = get_workspace(cf, nl, false; fvec=vecs)
    compute_fv!(wk, cf, cell; nl)
    one_body_vectors!(vecs, cell, cf)
    vecs
end

function _pair_channel_weight(cf::RandomEmbeddedCellFeature, si::Int, sj::Int, a::Int, b::Int)
    e = cf.embedding
    return e[si, a] * e[sj, b]
end

function _triplet_channel_weight(
    cf::RandomEmbeddedCellFeature,
    si::Int,
    sj::Int,
    sk::Int,
    a::Int,
    b::Int,
    c::Int,
)
    e = cf.embedding
    center = e[si, a]
    if b == c
        return center * e[sj, b] * e[sk, b]
    end
    return center * (e[sj, b] * e[sk, c] + e[sj, c] * e[sk, b])
end

function _embedded_fill_vout!(vout, ijkp_buf, qjk, np3::Int, nq3::Int)
    @inbounds for m in 1:np3
        ijkp = ijkp_buf[m]
        mm = (m - 1) * nq3
        @simd for o in 1:nq3
            vout[mm + o] = ijkp * qjk[o]
        end
    end
end

function _embedded_precompute_3b_factors!(
    ffact_center,
    ffact_j,
    ffact_k,
    modvij,
    modvik,
    modvjk,
    p3,
    q3,
    gij,
    gik,
    gjk,
    np3::Int,
    nq3::Int,
    ndims::Int,
)
    @inbounds for axis in 1:ndims
        for m in 1:np3
            g1 = p3[m] * gij * modvij[axis]
            g2 = p3[m] * gik * modvik[axis]
            mm = (m - 1) * nq3
            @simd for o in 1:nq3
                g3 = q3[o] * gjk * modvjk[axis]
                ffact_center[mm + o, axis] = -(g1 + g2)
                ffact_j[mm + o, axis] = g1 - g3
                ffact_k[mm + o, axis] = g2 + g3
            end
        end
    end
end

function _embedded_update_3b_gvec!(
    gvec,
    iat,
    i31,
    ineigh_j,
    ineigh_k,
    vout,
    ffact_center,
    ffact_j,
    ffact_k,
    wtrip,
    npnq::Int,
    ndims::Int,
)
    @inbounds for dir in 1:ndims
        @simd for nn in 1:npnq
            sv = wtrip * vout[nn]
            gvec[i31 + nn, dir, 1, iat] += sv * ffact_center[nn, dir]
            gvec[i31 + nn, dir, ineigh_j, iat] += sv * ffact_j[nn, dir]
            gvec[i31 + nn, dir, ineigh_k, iat] += sv * ffact_k[nn, dir]
        end
    end
end

function _embedded_precompute_gs!(
    gs1,
    gs2,
    gs3,
    ijkp_buf,
    qjk,
    p3,
    q3,
    gij,
    gik,
    gjk,
    np3::Int,
    nq3::Int,
)
    @inbounds for m in 1:np3
        g1 = p3[m] * gij
        g2 = p3[m] * gik
        ijkp = ijkp_buf[m]
        mm = (m - 1) * nq3
        @simd for o in 1:nq3
            idx = mm + o
            vout = ijkp * qjk[o]
            gs1[idx] = g1 * vout
            gs2[idx] = g2 * vout
            gs3[idx] = q3[o] * gjk * vout
        end
    end
end

function _embedded_update_3b_stotv!(
    stotv,
    iat,
    i31,
    outerij,
    outerik,
    outerjk,
    gs1,
    gs2,
    gs3,
    wtrip,
    npnq::Int,
)
    @inbounds for elm2 in 1:3
        for elm1 in 1:3
            c1 = outerij[elm1, elm2]
            c2 = outerik[elm1, elm2]
            c3 = outerjk[elm1, elm2]
            @simd for nn in 1:npnq
                stotv[i31 + nn, elm1, elm2, iat] += wtrip * (c1 * gs1[nn] + c2 * gs2[nn] + c3 * gs3[nn])
            end
        end
    end
end

function compute_embedded_fv_gv_one!(
    workspace,
    cf::RandomEmbeddedCellFeature,
    iat::Int,
    cell::Cell,
    nl::NeighbourList;
    offset=workspace.one_body_offset,
    do_grad=workspace.do_grad,
    zero_workspace::Bool=true,
    compute_hardcore::Bool=true,
    atom_species_ints=nothing,
    _bufs=nothing,
    use_spline=true,
)
    do_grad && (@assert nl.has_vectors)

    (; fvec, gvec, gvec_index, gvec_nn, stotv, hardcore) = workspace
    (; core_repulsion, fcore, score, ecore) = hardcore

    if do_grad && zero_workspace
        gvec[:, :, :, iat] .= 0.0
        stotv[:, :, :, iat] .= 0.0
    end

    basis = cf.cell_feature
    sym = species(cell)
    tiny_dist = 1e-14
    has_3b = !isempty(basis.p3) && !isempty(basis.q3)

    np2 = length(basis.p2)
    np3 = length(basis.p3)
    nq3 = length(basis.q3)
    n2proto = np2
    n3proto = np3 * nq3
    ndims = size(gvec, 2)

    maxrcut = has_3b ? max(basis.rcut2, basis.rcut3) : basis.rcut2
    maxrcut_sq = maxrcut * maxrcut

    if atom_species_ints === nothing
        atom_species_ints = _atom_species_ints(cf, cell)
    end

    # Ensure pre-allocated per-atom feature computation buffers
    if _bufs !== nothing
        bufs = _bufs
    else
        _ensure_feat_buf!(workspace, np3, nq3, has_3b ? 1 : 0, nl)
        bufs = workspace.feat_bufs[Threads.threadid()]
    end

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

    si = atom_species_ints[iat]
    e = cf.embedding
    total_fe2 = length(cf.pair_channels) * np2
    block0 = offset + total_fe2

    pij = has_3b ? view(bufs.pij, 1:np3, 1) : view(bufs.ijkp_buf, 1:0)
    pik = has_3b ? view(bufs.pik, 1:np3, 1) : view(bufs.ijkp_buf, 1:0)
    qjk = has_3b ? view(bufs.qjk, 1:nq3, 1) : view(bufs.ijkp_buf, 1:0)

    for (jat, jextend, rij, vij) in CellBase.eachneighbourvector(nl, iat)
        rij > maxrcut && continue
        @assert rij > tiny_dist

        sj = atom_species_ints[jat]
        ineigh_j = revmap[jat] + 1
        modvij = vij / rij

        if !isnothing(core_repulsion) && compute_hardcore
            _hard_core_update!(ecore, fcore, score, iat, jat, rij, vij, modvij, core_repulsion, sym[iat], sym[jat])
        end

        if n2proto > 0
            if use_spline
                fijp2, fij2 = evaluate_p2(basis, rij)
            else
                fij2 = basis.f2(rij, basis.rcut2)
                fijp2 = np2 == 0 ? () : (fij2 .^ basis.p2)
            end
            if fij2 != 0
                gij = basis.g2(rij, basis.rcut2)
                outer2 = do_grad ? vij * modvij' : nothing
                for (ich, (a, b)) in enumerate(cf.pair_channels)
                    wpair = e[si, a] * e[sj, b]
                    wpair == 0 && continue
                    i2 = offset + (ich - 1) * np2 + 1
                    i21 = i2 - 1
                    @inbounds @simd for m = 1:np2
                        fvec[i21 + m, iat] += fijp2[m] * wpair
                    end
                    do_grad || continue

                    gfij = basis.p2 .* fijp2 ./ fij2 .* (gij * wpair)
                    @inbounds for dir in 1:ndims
                        md = modvij[dir]
                        @simd for m = 1:np2
                            delta = md * gfij[m]
                            gvec[i21 + m, dir, 1, iat] -= delta
                            gvec[i21 + m, dir, ineigh_j, iat] += delta
                        end
                    end

                    @inbounds for elm2 in 1:3
                        for elm1 in 1:3
                            c = outer2[elm1, elm2]
                            @simd for m = 1:np2
                                stotv[i21 + m, elm1, elm2, iat] += c * gfij[m]
                            end
                        end
                    end
                end
            end
        end

        has_3b || continue
        if use_spline
            pij_vals, fijtmp = evaluate_p3(basis, rij)
        else
            fijtmp = basis.f3(rij, basis.rcut3)
            pij_vals = np3 == 0 ? () : (fijtmp .^ basis.p3)
        end
        fijtmp == 0 && continue
        inv_fij = 1.0 / fijtmp
        for m = 1:np3
            pij[m] = pij_vals[m]
        end

        for (kat, kextend, rik, vik) in CellBase.eachneighbourvector(nl, iat)
            rik > maxrcut && continue
            @assert rik > tiny_dist
            kextend <= jextend && continue

            vjk = nl.ea.positions[kextend] .- nl.ea.positions[jextend]
            rjk_sq = sum(vjk .^ 2)
            rjk_sq > maxrcut_sq && continue
            rjk = sqrt(rjk_sq)
            rjk > maxrcut && continue
            @assert rjk > tiny_dist

            sk = atom_species_ints[kat]
            ineigh_k = revmap[kat] + 1
            modvik = vik / rik
            modvjk = vjk / rjk

            if use_spline
                pik_vals, fiktmp = evaluate_p3(basis, rik)
            else
                fiktmp = basis.f3(rik, basis.rcut3)
                pik_vals = np3 == 0 ? () : (fiktmp .^ basis.p3)
            end
            fiktmp == 0 && continue
            inv_fik = 1.0 / fiktmp
            for m = 1:np3
                pik[m] = pik_vals[m]
            end

            if use_spline
                qjk_vals, fjktmp = evaluate_q3(basis, rjk)
            else
                fjktmp = basis.f3(rjk, basis.rcut3)
                qjk_vals = nq3 == 0 ? () : (fjktmp .^ basis.q3)
            end
            fjktmp == 0 && continue
            inv_fjk = 1.0 / fjktmp
            for o = 1:nq3
                qjk[o] = qjk_vals[o]
            end

            gij = basis.g3(rij, basis.rcut3) * inv_fij
            gik = basis.g3(rik, basis.rcut3) * inv_fik
            gjk = basis.g3(rjk, basis.rcut3) * inv_fjk
            ijkp_buf = bufs.ijkp_buf
            @inbounds @simd for m in 1:np3
                ijkp_buf[m] = pij[m] * pik[m]
            end
            vout = bufs.gs1
            _embedded_fill_vout!(vout, ijkp_buf, qjk, np3, nq3)
            if do_grad
                _embedded_precompute_3b_factors!(
                    bufs.ffact_center,
                    bufs.ffact_j,
                    bufs.ffact_k,
                    modvij,
                    modvik,
                    modvjk,
                    basis.p3,
                    basis.q3,
                    gij,
                    gik,
                    gjk,
                    np3,
                    nq3,
                    ndims,
                )
            end
            outerij = do_grad ? (vij * modvij') : nothing
            outerik = do_grad ? (vik * modvik') : nothing
            outerjk = do_grad ? (vjk * modvjk') : nothing

            for (ich, (a, b, c)) in enumerate(cf.triplet_channels)
                wtrip = _triplet_channel_weight(cf, si, sj, sk, a, b, c)
                wtrip == 0 && continue
                i31 = block0 + (ich - 1) * n3proto
                @inbounds @simd for nn in 1:n3proto
                    fvec[i31 + nn, iat] += wtrip * vout[nn]
                end

                do_grad || continue
                _embedded_update_3b_gvec!(
                    gvec,
                    iat,
                    i31,
                    ineigh_j,
                    ineigh_k,
                    vout,
                    bufs.ffact_center,
                    bufs.ffact_j,
                    bufs.ffact_k,
                    wtrip,
                    n3proto,
                    ndims,
                )
            end

            if do_grad
                _embedded_precompute_gs!(
                    bufs.gs1,
                    bufs.gs2,
                    bufs.gs3,
                    ijkp_buf,
                    qjk,
                    basis.p3,
                    basis.q3,
                    gij,
                    gik,
                    gjk,
                    np3,
                    nq3,
                )
                for (ich, (a, b, c)) in enumerate(cf.triplet_channels)
                    wtrip = _triplet_channel_weight(cf, si, sj, sk, a, b, c)
                    wtrip == 0 && continue
                    i31 = block0 + (ich - 1) * n3proto
                    _embedded_update_3b_stotv!(
                        stotv,
                        iat,
                        i31,
                        outerij,
                        outerik,
                        outerjk,
                        bufs.gs1,
                        bufs.gs2,
                        bufs.gs3,
                        wtrip,
                        n3proto,
                    )
                end
            end
        end
    end

    return workspace
end

function get_workspace(
    cf::RandomEmbeddedCellFeature,
    nl,
    do_grad=true;
    fvec=nothing,
    core=nothing,
)
    if fvec === nothing
        fvec = zeros(sum(feature_size(cf)), length(nl.nneigh))
    end
    nn_max = do_grad ? maximum(x -> length(unique(x)), eachcol(nl.orig_indices)) + 1 : 1
    GradientWorkspace(
        fvec,
        nn_max;
        ndims=length(nl.ea.positions[1]),
        core=core,
        do_grad=do_grad,
        one_body_offset=one_body_offset(cf),
    )
end

function compute_fv!(
    workspace,
    cf::RandomEmbeddedCellFeature,
    cell::Cell;
    nl,
    offset=workspace.one_body_offset,
    use_spline=true,
)
    reset!(workspace)
    atom_species_ints = _atom_species_ints(cf, cell)
    _populate_feat_bufs!(workspace, cf.cell_feature, nl)
    nat = natoms(cell)
    chunks = _atom_chunks(nat)
    Threads.@threads for ci in eachindex(chunks)
        bufs = workspace.feat_bufs[ci]
        for iat in chunks[ci]
            compute_embedded_fv_gv_one!(
                workspace,
                cf,
                iat,
                cell,
                nl;
                do_grad=false,
                offset=offset,
                atom_species_ints=atom_species_ints,
                _bufs=bufs,
                use_spline=use_spline,
            )
        end
    end
    return workspace.fvec
end

function compute_fv_gv!(
    workspace,
    cf::RandomEmbeddedCellFeature,
    cell::Cell;
    nl=NeighbourList(cell, suggest_rcut(cf; shell=0.0); savevec=true),
    offset=workspace.one_body_offset,
    use_spline=true,
)
    reset!(workspace)
    atom_species_ints = _atom_species_ints(cf, cell)
    _populate_feat_bufs!(workspace, cf.cell_feature, nl)
    nat = natoms(cell)
    chunks = _atom_chunks(nat)
    Threads.@threads for ci in eachindex(chunks)
        bufs = workspace.feat_bufs[ci]
        for iat in chunks[ci]
            compute_embedded_fv_gv_one!(
                workspace,
                cf,
                iat,
                cell,
                nl;
                offset=offset,
                atom_species_ints=atom_species_ints,
                _bufs=bufs,
                use_spline=use_spline,
            )
        end
    end
    return workspace
end

function compute_fv_gv(cf::RandomEmbeddedCellFeature, cell::Cell; kwargs...)
    nl = get(kwargs, :nl, NeighbourList(cell, suggest_rcut(cf; shell=0.0); savevec=true))
    workspace = get_workspace(cf, nl, true)
    return compute_fv_gv!(workspace, cf, cell; kwargs...)
end

#=
Learnable embedding gradient computation.

Computes dE_total / d(embedding) given gv = dE_total / d(fvec) (from gradinp!).

The embedding gradient for species s and embedding dimension d is:

    gE[s, d] = Σ_{iat, j} gv[j, iat] * ∂fvec[j, iat] / ∂E[s, d]

The chain rule splits into one-body, two-body, and three-body contributions
that mirror the feature computation loops.
=#

"""
    grad_embedding_one!(gE, cf, iat, cell, nl, gv; offset)

Accumulate the embedding gradient contribution from a single center atom `iat`.

This mirrors `compute_embedded_fv_gv_one!` but accumulates into `gE` instead of
`fvec`/`gvec`.  Only the embedding gradient is computed — no position/strain
gradients.

# Arguments
- `gE`: Embedding gradient matrix `(n_elem, n_embed)`, accumulated in-place
- `cf`: The `RandomEmbeddedCellFeature` with current embedding
- `iat`: Center atom index
- `cell`: Crystal structure
- `nl`: Neighbour list
- `gv`: Upstream gradient `dE_total/d(fvec)`, shape `(nfeatures, natoms)`
"""
function grad_embedding_one!(
    gE::Matrix{T},
    cf::RandomEmbeddedCellFeature,
    iat::Int,
    cell::Cell,
    nl::NeighbourList,
    gv::AbstractMatrix{T};
    offset=embedding_size(cf),
    use_spline=true,
) where {T}
    basis = cf.cell_feature
    sym = species(cell)
    e = cf.embedding
    tiny_dist = 1e-14
    maxrcut = suggest_rcut(cf; shell=0.0)
    maxrcut_sq = maxrcut * maxrcut

    np2 = length(basis.p2)
    np3 = length(basis.p3)
    nq3 = length(basis.q3)
    has_3b = np3 > 0 && nq3 > 0
    n2proto = np2
    n3proto = np3 * nq3

    si = basis.spec_to_idx[sym[iat]]

    # --- One-body contribution ---
    # fvec[d, iat] = E[si, d]  =>  ∂fvec[d, iat]/∂E[si, d] = 1
    # One-body features are at [offset - embedding_size + 1 : offset]
    ob_start = offset - embedding_size(cf) + 1
    for d in 1:embedding_size(cf)
        gE[si, d] += gv[ob_start + d - 1, iat]
    end

    # --- Pre-compute three-body caches ---
    pij = zeros(T, np3)
    inv_fij = zero(T)
    pik = zeros(T, np3)
    inv_fik = zero(T)
    qjk = zeros(T, nq3)

    # --- Two-body and three-body loops ---
    for (jat, jextend, rij, vij) in CellBase.eachneighbourvector(nl, iat)
        rij > maxrcut && continue
        @assert rij > tiny_dist

        sj = basis.spec_to_idx[sym[jat]]

        if has_3b
            if use_spline
                pijvals, fijtmp = evaluate_p3(basis, rij)
            else
                fijtmp = basis.f3(rij, basis.rcut3)
                pijvals = fijtmp .^ basis.p3
            end
            fijtmp == 0 && continue
            inv_fij = 1.0 / fijtmp
            for m = 1:np3
                pij[m] = pijvals[m]
            end
        end

        # --- Two-body embedding gradient ---
        # fvec[i, iat] += raw * E[si, a] * E[sj, b]
        # => gE[si, a] += gv[i, iat] * raw * E[sj, b]
        # => gE[sj, b] += gv[i, iat] * raw * E[si, a]
        if n2proto > 0
            if use_spline
                fijp2, fij2 = evaluate_p2(basis, rij)
            else
                fij2 = basis.f2(rij, basis.rcut2)
                fijp2 = fij2 .^ basis.p2
            end
            for (ich, (a, b)) in enumerate(cf.pair_channels)
                base_i = offset + 1 + (ich - 1) * n2proto
                for m = 1:np2
                    raw = fijp2[m]
                    idx = base_i + (m - 1)
                    ggv = gv[idx, iat] * raw
                    if ggv != zero(T)
                        gE[si, a] += ggv * e[sj, b]
                        gE[sj, b] += ggv * e[si, a]
                    end
                end
            end
        end

        has_3b || continue

        # --- Three-body embedding gradient ---

        for (kat, kextend, rik, vik) in CellBase.eachneighbourvector(nl, iat)
            rik > maxrcut && continue
            @assert rik > tiny_dist
            kextend <= jextend && continue

            vjk = nl.ea.positions[kextend] .- nl.ea.positions[jextend]
            rjk_sq = sum(vjk .^ 2)
            rjk_sq > maxrcut_sq && continue
            rjk = sqrt(rjk_sq)
            rjk > maxrcut && continue
            @assert rjk > tiny_dist

            sk = basis.spec_to_idx[sym[kat]]

            if use_spline
                pikvals, fiktmp = evaluate_p3(basis, rik)
            else
                fiktmp = basis.f3(rik, basis.rcut3)
                pikvals = fiktmp .^ basis.p3
            end
            fiktmp == 0 && continue

            if use_spline
                qjkvals, fjktmp = evaluate_q3(basis, rjk)
            else
                fjktmp = basis.f3(rjk, basis.rcut3)
                qjkvals = fjktmp .^ basis.q3
            end
            fjktmp == 0 && continue
            inv_fik = 1.0 / fiktmp
            for m = 1:np3
                pik[m] = pikvals[m]
            end
            for o = 1:nq3
                qjk[o] = qjkvals[o]
            end

            block0 = offset + n2proto * length(cf.pair_channels)
            for (ich, (a, b, c)) in enumerate(cf.triplet_channels)
                base_i = block0 + 1 + (ich - 1) * n3proto
                for m = 1:np3
                    ijkp = pij[m] * pik[m]
                    for o = 1:nq3
                        raw = ijkp * qjk[o]
                        idx = base_i + (m - 1) * nq3 + (o - 1)
                        ggv = gv[idx, iat] * raw
                        ggv == zero(T) && continue

                        if b == c
                            # wtrip = E[si, a] * E[sj, b] * E[sk, b]
                            gE[si, a] += ggv * e[sj, b] * e[sk, b]
                            gE[sj, b] += ggv * e[si, a] * e[sk, b]
                            gE[sk, b] += ggv * e[si, a] * e[sj, b]
                        else
                            # wtrip = E[si, a] * (E[sj, b]*E[sk, c] + E[sj, c]*E[sk, b])
                            gE[si, a] += ggv * (e[sj, b] * e[sk, c] + e[sj, c] * e[sk, b])
                            gE[sj, b] += ggv * e[si, a] * e[sk, c]
                            gE[sj, c] += ggv * e[si, a] * e[sk, b]
                            gE[sk, b] += ggv * e[si, a] * e[sj, c]
                            gE[sk, c] += ggv * e[si, a] * e[sj, b]
                        end
                    end
                end
            end
        end
    end
    return gE
end

"""
    grad_embedding!(gE, cf, cell, nl, gv)

Compute the gradient of the total energy with respect to the embedding matrix.

    gE[s, d] = dE_total / d(embedding[s, d])

given `gv = dE_total / d(fvec)` from `gradinp!`.
"""
function grad_embedding!(
    gE::Matrix{T},
    cf::RandomEmbeddedCellFeature,
    cell::Cell,
    nl::NeighbourList,
    gv::AbstractMatrix{T};
    offset=embedding_size(cf),
    use_spline=true,
) where {T}
    fill!(gE, zero(T))
    for iat in 1:natoms(cell)
        grad_embedding_one!(gE, cf, iat, cell, nl, gv; offset, use_spline)
    end
    return gE
end

"""
    recompute_features!(fvec_out, cf, cell, workspace, nl)

Recompute features for a single structure with the current embedding.
"""
function recompute_features!(
    fvec_out,
    cf::RandomEmbeddedCellFeature,
    cell::Cell,
    workspace,
    nl,
)
    compute_fv!(workspace, cf, cell; nl)
    one_body_vectors!(workspace.fvec, cell, cf)
    if fvec_out !== workspace.fvec
        fvec_out .= workspace.fvec
    end
    return fvec_out
end

"""
    compute_feature_stats!(mean_vec, std_vec, structures, cf, workspaces, nls)

Compute per-feature mean and standard deviation across all structures
using the current embedding in `cf`.
"""
function compute_feature_stats!(
    mean_vec::AbstractVector{T},
    std_vec::AbstractVector{T},
    structures,
    cf::RandomEmbeddedCellFeature,
    workspaces,
    nls,
) where {T}
    nf = nfeatures(cf)
    fill!(mean_vec, zero(T))
    fill!(std_vec, zero(T))
    n_atoms_total = 0

    # Welford's online algorithm
    for i in eachindex(structures)
        recompute_features!(workspaces[i].fvec, cf, structures[i], workspaces[i], nls[i])
        fvec = workspaces[i].fvec
        nat = size(fvec, 2)
        for iat in 1:nat
            n_atoms_total += 1
            delta_n = 1.0 / n_atoms_total
            for j in 1:nf
                val = fvec[j, iat]
                delta = val - mean_vec[j]
                mean_vec[j] += delta * delta_n
                delta2 = val - mean_vec[j]
                std_vec[j] += delta * delta2
            end
        end
    end

    if n_atoms_total > 1
        std_vec .= sqrt.(std_vec ./ (n_atoms_total - 1))
    end
    # Reset near-zero std to 1 to avoid division by near-zero
    for j in eachindex(std_vec)
        if std_vec[j] < 1e-6
            std_vec[j] = 1.0
        end
    end
    return mean_vec, std_vec
end
