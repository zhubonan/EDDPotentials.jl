import Random
import LinearAlgebra

struct RandomEmbeddedCellFeature{T,V,W,P2,P3,X} <: AbstractCellFeatureSpec
    elements::Vector{Symbol}
    embedding::Matrix{T}
    pair_channels::V
    triplet_channels::W
    two_body_proto::P2
    three_body_proto::P3
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

function _embedded_prototypes(
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
)
    elements = sort(unique(map(Symbol, elements)))
    isempty(elements) && throw(ArgumentError("RandomEmbeddedCellFeature requires at least one element."))

    p2t, p3t, q3t = _normalize_embedded_sequences(p2, p3, q3; geometry_sequence)
    dummy = elements[1]

    proto2 = isempty(p2t) ? nothing : TwoBodyFeature(f2, g2, p2t, (dummy, dummy), rcut2)
    proto3 =
        isempty(p3t) || isempty(q3t) ? nothing :
        ThreeBodyFeature(f3, g3, p3t, q3t, (dummy, dummy, dummy), rcut3)

    kwargs = (; nembed=nothing, seed=nothing, T=Float64, p2=p2t, p3=p3t, q3=q3t, rcut2, rcut3, geometry_sequence)
    elements, proto2, proto3, kwargs
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
    kwargs...,
)
    proto_elements, proto2, proto3, proto_kwargs = _embedded_prototypes(elements; kwargs...)
    embedding = _random_embedding_matrix(T, length(proto_elements), nembed; seed, optimize=optimize_embedding)
    stored_kwargs = merge(proto_kwargs, (; nembed, seed, T))
    RandomEmbeddedCellFeature(
        proto_elements,
        embedding,
        _pair_channels(nembed),
        _triplet_channels(nembed),
        proto2,
        proto3,
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
    println(io, "  Two-body prototype:")
    println(io, "    $(x.two_body_proto)")
    println(io, "  Three-body prototype:")
    println(io, "    $(x.three_body_proto)")
end

embedding_size(cf::RandomEmbeddedCellFeature) = size(cf.embedding, 2)

"""
    one_body_offset(cf)

Return the offset for one-body features in the force/stress chain rule.
For single embedding, one-body features are contiguous at the start.
"""
one_body_offset(cf::RandomEmbeddedCellFeature) = embedding_size(cf)

function _species_index_map(elements)
    Dict(s => i for (i, s) in enumerate(elements))
end

function feature_size(cf::RandomEmbeddedCellFeature)
    nembed = embedding_size(cf)
    n2proto = isnothing(cf.two_body_proto) ? 0 : nfeatures(cf.two_body_proto)
    n3proto = isnothing(cf.three_body_proto) ? 0 : nfeatures(cf.three_body_proto)
    (
        nembed,
        n2proto * length(cf.pair_channels),
        n3proto * length(cf.triplet_channels),
    )
end

nfeatures(cf::RandomEmbeddedCellFeature) = sum(feature_size(cf))

function suggest_rcut(cf::RandomEmbeddedCellFeature; shell=1.0)
    vals = Float64[]
    isnothing(cf.two_body_proto) || push!(vals, cf.two_body_proto.rcut)
    isnothing(cf.three_body_proto) || push!(vals, cf.three_body_proto.rcut)
    isempty(vals) && return shell
    maximum(vals) + shell
end

function one_body_vectors!(v::AbstractMatrix, cell::Cell, cf::RandomEmbeddedCellFeature; offset=0)
    spec_index = _species_index_map(cf.elements)
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
)
    do_grad && (@assert nl.has_vectors)

    (; fvec, gvec, gvec_index, gvec_nn, stotv, hardcore) = workspace
    (; core, fcore, score, ecore) = hardcore

    if do_grad && zero_workspace
        gvec[:, :, :, iat] .= 0.0
        stotv[:, :, :, iat] .= 0.0
    end

    feat2 = cf.two_body_proto
    feat3 = cf.three_body_proto
    spec_index = _species_index_map(cf.elements)
    sym = species(cell)
    tiny_dist = 1e-14
    maxrcut = suggest_rcut(cf; shell=0.0)

    n2proto = isnothing(feat2) ? 0 : nfeatures(feat2)
    n3proto = isnothing(feat3) ? 0 : nfeatures(feat3)

    neighbour_indices = unique(nl.orig_indices[1:nl.nneigh[iat], iat])
    if do_grad
        gvec_index[1, iat] = iat
        gvec_index[2:length(neighbour_indices)+1, iat] .= neighbour_indices
        gvec_nn[iat] = length(neighbour_indices) + 1
    end

    si = spec_index[sym[iat]]

    npmax3 = isnothing(feat3) ? 0 : feat3.np
    nqmax3 = isnothing(feat3) ? 0 : feat3.nq
    pij = zeros(eltype(fvec), npmax3)
    inv_fij = zero(eltype(fvec))
    pik = zeros(eltype(fvec), npmax3)
    inv_fik = zero(eltype(fvec))
    qjk = zeros(eltype(fvec), nqmax3)
    inv_fjk = zero(eltype(fvec))

    for (jat, jextend, rij, vij) in CellBase.eachneighbourvector(nl, iat)
        rij > maxrcut && continue
        @assert rij > tiny_dist

        sj = spec_index[sym[jat]]
        ineigh_j = findfirst(isequal(jat), neighbour_indices) + 1
        modvij = vij / rij

        if !isnothing(feat3)
            fijtmp = feat3.f(rij, feat3.rcut)
            inv_fij = 1.0 / fijtmp
            for m = 1:feat3.np
                pij[m] = fast_pow(fijtmp, feat3.p[m])
            end
        end

        if !isnothing(core) && compute_hardcore
            _hard_core_update!(ecore, fcore, score, iat, jat, rij, vij, modvij, core)[1]
        end

        if n2proto > 0
            fij2 = feat2.f(rij, feat2.rcut)
            gij = feat2.g(rij, feat2.rcut)
            for (ich, (a, b)) in enumerate(cf.pair_channels)
                wpair = _pair_channel_weight(cf, si, sj, a, b)
                i = offset + 1 + (ich - 1) * n2proto
                for m = 1:feat2.np
                    raw = fast_pow(fij2, feat2.p[m])
                    val = raw * wpair
                    fvec[i, iat] += val

                    if do_grad && raw != 0.0 && wpair != 0.0
                        gfij = feat2.p[m] * fast_pow(fij2, feat2.p[m] - 1) * gij * wpair
                        for elm in axes(gvec, 1)
                            gvec[elm, i, 1, iat] -= modvij[elm] * gfij
                            gvec[elm, i, ineigh_j, iat] += modvij[elm] * gfij
                        end
                        for elm2 in axes(stotv, 2), elm1 in axes(stotv, 1)
                            stotv[elm1, elm2, i, iat] += vij[elm1] * modvij[elm2] * gfij
                        end
                    end
                    i += 1
                end
            end
        end

        isnothing(feat3) && continue

        for (kat, kextend, rik, vik) in CellBase.eachneighbourvector(nl, iat)
            rik > maxrcut && continue
            @assert rik > tiny_dist
            kextend <= jextend && continue

            vjk = nl.ea.positions[kextend] .- nl.ea.positions[jextend]
            rjk = norm(vjk)
            rjk > maxrcut && continue
            @assert rjk > tiny_dist

            sk = spec_index[sym[kat]]
            ineigh_k = findfirst(isequal(kat), neighbour_indices) + 1
            modvik = vik / rik
            modvjk = vjk / rjk

            fiktmp = feat3.f(rik, feat3.rcut)
            fjktmp = feat3.f(rjk, feat3.rcut)
            inv_fik = 1.0 / fiktmp
            inv_fjk = 1.0 / fjktmp
            for m = 1:feat3.np
                pik[m] = fast_pow(fiktmp, feat3.p[m])
            end
            for o = 1:feat3.nq
                qjk[o] = fast_pow(fjktmp, feat3.q[o])
            end

            gij = feat3.g(rij, feat3.rcut) * inv_fij
            gik = feat3.g(rik, feat3.rcut) * inv_fik
            gjk = feat3.g(rjk, feat3.rcut) * inv_fjk

            block0 = offset + n2proto * length(cf.pair_channels)
            for (ich, (a, b, c)) in enumerate(cf.triplet_channels)
                wtrip = _triplet_channel_weight(cf, si, sj, sk, a, b, c)
                i = block0 + 1 + (ich - 1) * n3proto
                for m = 1:feat3.np
                    ijkp = pij[m] * pik[m]
                    dlog_ij = feat3.p[m] * gij
                    dlog_ik = feat3.p[m] * gik
                    for o = 1:feat3.nq
                        raw = ijkp * qjk[o]
                        val = raw * wtrip
                        fvec[i, iat] += val

                        if do_grad && raw != 0.0 && wtrip != 0.0
                            t1 = modvij * dlog_ij * val
                            t2 = modvik * dlog_ik * val
                            t3 = modvjk * (feat3.q[o] * gjk) * val

                            for elm in axes(gvec, 1)
                                gvec[elm, i, 1, iat] -= t1[elm] + t2[elm]
                                gvec[elm, i, ineigh_j, iat] += t1[elm] - t3[elm]
                                gvec[elm, i, ineigh_k, iat] += t2[elm] + t3[elm]
                            end

                            for elm2 in axes(stotv, 2), elm1 in axes(stotv, 1)
                                stotv[elm1, elm2, i, iat] += (
                                    vij[elm1] * t1[elm2] +
                                    vik[elm1] * t2[elm2] +
                                    vjk[elm1] * t3[elm2]
                                )
                            end
                        end

                        i += 1
                    end
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
    nn_max = maximum(x -> length(unique(x)), eachcol(nl.orig_indices)) + 1
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
)
    reset!(workspace)
    for iat = 1:natoms(cell)
        compute_embedded_fv_gv_one!(workspace, cf, iat, cell, nl; do_grad=false, offset=offset)
    end
    return workspace.fvec
end

function compute_fv_gv!(
    workspace,
    cf::RandomEmbeddedCellFeature,
    cell::Cell;
    nl=NeighbourList(cell, suggest_rcut(cf; shell=0.0); savevec=true),
    offset=workspace.one_body_offset,
)
    reset!(workspace)
    for iat = 1:natoms(cell)
        compute_embedded_fv_gv_one!(workspace, cf, iat, cell, nl; offset=offset)
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
) where {T}
    feat2 = cf.two_body_proto
    feat3 = cf.three_body_proto
    spec_index = _species_index_map(cf.elements)
    sym = species(cell)
    e = cf.embedding
    tiny_dist = 1e-14
    maxrcut = suggest_rcut(cf; shell=0.0)

    n2proto = isnothing(feat2) ? 0 : nfeatures(feat2)
    n3proto = isnothing(feat3) ? 0 : nfeatures(feat3)

    si = spec_index[sym[iat]]

    # --- One-body contribution ---
    # fvec[d, iat] = E[si, d]  =>  ∂fvec[d, iat]/∂E[si, d] = 1
    # One-body features are at [offset - embedding_size + 1 : offset]
    ob_start = offset - embedding_size(cf) + 1
    for d in 1:embedding_size(cf)
        gE[si, d] += gv[ob_start + d - 1, iat]
    end

    # --- Pre-compute three-body caches ---
    npmax3 = isnothing(feat3) ? 0 : feat3.np
    nqmax3 = isnothing(feat3) ? 0 : feat3.nq
    pij = zeros(T, npmax3)
    inv_fij = zero(T)
    pik = zeros(T, npmax3)
    inv_fik = zero(T)
    qjk = zeros(T, nqmax3)

    # --- Two-body and three-body loops ---
    for (jat, jextend, rij, vij) in CellBase.eachneighbourvector(nl, iat)
        rij > maxrcut && continue
        @assert rij > tiny_dist

        sj = spec_index[sym[jat]]

        if !isnothing(feat3)
            fijtmp = feat3.f(rij, feat3.rcut)
            inv_fij = 1.0 / fijtmp
            for m = 1:feat3.np
                pij[m] = fast_pow(fijtmp, feat3.p[m])
            end
        end

        # --- Two-body embedding gradient ---
        # fvec[i, iat] += raw * E[si, a] * E[sj, b]
        # => gE[si, a] += gv[i, iat] * raw * E[sj, b]
        # => gE[sj, b] += gv[i, iat] * raw * E[si, a]
        if n2proto > 0
            fij2 = feat2.f(rij, feat2.rcut)
            for (ich, (a, b)) in enumerate(cf.pair_channels)
                base_i = offset + 1 + (ich - 1) * n2proto
                for m = 1:feat2.np
                    raw = fast_pow(fij2, feat2.p[m])
                    idx = base_i + (m - 1)
                    ggv = gv[idx, iat] * raw
                    if ggv != zero(T)
                        gE[si, a] += ggv * e[sj, b]
                        gE[sj, b] += ggv * e[si, a]
                    end
                end
            end
        end

        isnothing(feat3) && continue

        # --- Three-body embedding gradient ---

        for (kat, kextend, rik, vik) in CellBase.eachneighbourvector(nl, iat)
            rik > maxrcut && continue
            @assert rik > tiny_dist
            kextend <= jextend && continue

            vjk = nl.ea.positions[kextend] .- nl.ea.positions[jextend]
            rjk = norm(vjk)
            rjk > maxrcut && continue
            @assert rjk > tiny_dist

            sk = spec_index[sym[kat]]

            fiktmp = feat3.f(rik, feat3.rcut)
            fjktmp = feat3.f(rjk, feat3.rcut)
            inv_fik = 1.0 / fiktmp
            for m = 1:feat3.np
                pik[m] = fast_pow(fiktmp, feat3.p[m])
            end
            for o = 1:feat3.nq
                qjk[o] = fast_pow(fjktmp, feat3.q[o])
            end

            block0 = offset + n2proto * length(cf.pair_channels)
            for (ich, (a, b, c)) in enumerate(cf.triplet_channels)
                base_i = block0 + 1 + (ich - 1) * n3proto
                for m = 1:feat3.np
                    ijkp = pij[m] * pik[m]
                    for o = 1:feat3.nq
                        raw = ijkp * qjk[o]
                        idx = base_i + (m - 1) * feat3.nq + (o - 1)
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
) where {T}
    fill!(gE, zero(T))
    for iat in 1:natoms(cell)
        grad_embedding_one!(gE, cf, iat, cell, nl, gv; offset)
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
