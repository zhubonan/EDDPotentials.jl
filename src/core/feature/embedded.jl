import Random

struct RandomEmbeddedCellFeature{C,T,V,W,X} <: AbstractCellFeatureSpec
    elements::Vector{Symbol}
    base::C
    embedding::Matrix{T}
    pair_channels::V
    triplet_channels::W
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

function _validate_embeddable(cf::CellFeature)
    if !isempty(cf.two_body)
        ref = cf.two_body[1]
        ok = all(
            f -> f.p == ref.p && f.rcut == ref.rcut && f.f == ref.f && f.g == ref.g,
            cf.two_body,
        )
        ok || error(
            "RandomEmbeddedCellFeature requires uniform two-body feature settings across species.",
        )
    end
    if !isempty(cf.three_body)
        ref = cf.three_body[1]
        ok = all(
            f ->
                f.p == ref.p &&
                    f.q == ref.q &&
                    f.rcut == ref.rcut &&
                    f.f == ref.f &&
                    f.g == ref.g,
            cf.three_body,
        )
        ok || error(
            "RandomEmbeddedCellFeature requires uniform three-body feature settings across species.",
        )
    end
    cf
end

function _random_embedding_matrix(T, nelem::Int, nembed::Int; seed::Integer=0)
    rng = Random.MersenneTwister(seed)
    weight = randn(rng, T, nelem, nembed)
    for i = 1:nelem
        rownorm = sqrt(sum(abs2, @view weight[i, :]))
        rownorm > 0 || continue
        @views weight[i, :] ./= rownorm
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
    cf::CellFeature;
    nembed::Int=4,
    seed::Integer=0,
    T::DataType=Float64,
)
    _validate_embeddable(cf)
    embedding = _random_embedding_matrix(T, length(cf.elements), nembed; seed)
    kwargs = (; nembed, seed, T)
    RandomEmbeddedCellFeature(
        copy(cf.elements),
        cf,
        embedding,
        _pair_channels(nembed),
        _triplet_channels(nembed),
        kwargs,
    )
end

function RandomEmbeddedCellFeature(
    elements;
    nembed::Int=4,
    seed::Integer=0,
    T::DataType=Float64,
    kwargs...,
)
    RandomEmbeddedCellFeature(CellFeature(elements; kwargs...); nembed, seed, T)
end

Base.show(io::IO, x::RandomEmbeddedCellFeature) = Base.show(io, MIME("text/plain"), x)

function Base.show(io::IO, ::MIME"text/plain", x::RandomEmbeddedCellFeature)
    println(io, "$(typeof(x))")
    println(io, "  Elements:")
    println(io, "    $(x.elements)")
    println(io, "  Embedded channels: $(embedding_size(x))")
    println(io, "  Base feature:")
    show(io, MIME("text/plain"), x.base)
end

embedding_size(cf::RandomEmbeddedCellFeature) = size(cf.embedding, 2)

function _species_index_map(elements)
    Dict(s => i for (i, s) in enumerate(elements))
end

function feature_size(cf::RandomEmbeddedCellFeature)
    nembed = embedding_size(cf)
    n2proto = isempty(cf.base.two_body) ? 0 : nfeatures(cf.base.two_body[1])
    n3proto = isempty(cf.base.three_body) ? 0 : nfeatures(cf.base.three_body[1])
    (
        nembed,
        n2proto * length(cf.pair_channels),
        n3proto * length(cf.triplet_channels),
    )
end

nfeatures(cf::RandomEmbeddedCellFeature) = sum(feature_size(cf))

suggest_rcut(cf::RandomEmbeddedCellFeature; shell=1.0) = suggest_rcut(cf.base; shell)

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

function pair_embedding_weights(cf::RandomEmbeddedCellFeature)
    weights = zeros(eltype(cf.embedding), length(cf.base.two_body), length(cf.pair_channels))
    spec_index = _species_index_map(cf.elements)
    for (ife, feat) in enumerate(cf.base.two_body)
        si = spec_index[feat.sij_idx[1]]
        sj = spec_index[feat.sij_idx[2]]
        for (ich, (a, b)) in enumerate(cf.pair_channels)
            weights[ife, ich] = _pair_channel_weight(cf, si, sj, a, b)
        end
    end
    weights
end

function triplet_embedding_weights(cf::RandomEmbeddedCellFeature)
    weights = zeros(eltype(cf.embedding), length(cf.base.three_body), length(cf.triplet_channels))
    spec_index = _species_index_map(cf.elements)
    for (ife, feat) in enumerate(cf.base.three_body)
        si = spec_index[feat.sijk_idx[1]]
        sj = spec_index[feat.sijk_idx[2]]
        sk = spec_index[feat.sijk_idx[3]]
        for (ich, (a, b, c)) in enumerate(cf.triplet_channels)
            weights[ife, ich] = _triplet_channel_weight(cf, si, sj, sk, a, b, c)
        end
    end
    weights
end

function _contract_full_feature_matrix(cf::RandomEmbeddedCellFeature, full::AbstractMatrix)
    n1full, n2full, n3full = feature_size(cf.base)
    n1, n2, n3 = feature_size(cf)
    out = zeros(eltype(full), n1 + n2 + n3, size(full, 2))

    out[1:n1, :] .= transpose(cf.embedding) * full[1:n1full, :]

    if n2 > 0
        n2proto = nfeatures(cf.base.two_body[1])
        w2 = pair_embedding_weights(cf)
        for i = axes(full, 2)
            m2 = reshape(
                view(full, n1full+1:n1full+n2full, i),
                n2proto,
                length(cf.base.two_body),
            )
            tmp2 = reshape(
                view(out, n1+1:n1+n2, i),
                n2proto,
                length(cf.pair_channels),
            )
            mul!(tmp2, m2, w2)
        end
    end

    if n3 > 0
        n3proto = nfeatures(cf.base.three_body[1])
        w3 = triplet_embedding_weights(cf)
        for i = axes(full, 2)
            m3 = reshape(
                view(full, n1full+n2full+1:n1full+n2full+n3full, i),
                n3proto,
                length(cf.base.three_body),
            )
            tmp3 = reshape(
                view(out, n1+n2+1:n1+n2+n3, i),
                n3proto,
                length(cf.triplet_channels),
            )
            mul!(tmp3, m3, w3)
        end
    end

    return out
end

function _feature_rcut(features2, features3)
    vals = Float64[]
    isempty(features2) || append!(vals, map(x -> x.rcut, features2))
    isempty(features3) || append!(vals, map(x -> x.rcut, features3))
    isempty(vals) && return 0.0
    maximum(vals)
end

function compute_embedded_fv_gv_one!(
    workspace,
    cf::RandomEmbeddedCellFeature,
    iat::Int,
    cell::Cell,
    nl::NeighbourList;
    offset=workspace.one_body_offset,
    do_grad=workspace.do_grad,
)
    do_grad && (@assert nl.has_vectors)

    (; fvec, gvec, gvec_index, gvec_nn, stotv, hardcore) = workspace
    (; core, fcore, score, ecore) = hardcore

    if do_grad
        gvec[:, :, :, iat] .= 0.0
        stotv[:, :, :, iat] .= 0.0
    end

    base2 = cf.base.two_body
    base3 = cf.base.three_body
    spec_index = _species_index_map(cf.elements)
    sym = species(cell)
    tiny_dist = 1e-14
    maxrcut = _feature_rcut(base2, base3)

    n2proto = isempty(base2) ? 0 : nfeatures(base2[1])
    n3proto = isempty(base3) ? 0 : nfeatures(base3[1])
    feat2 = isempty(base2) ? nothing : base2[1]
    feat3 = isempty(base3) ? nothing : base3[1]

    neighbour_indices = unique(nl.orig_indices[1:nl.nneigh[iat], iat])
    if do_grad
        gvec_index[1, iat] = iat
        gvec_index[2:length(neighbour_indices)+1, iat] .= neighbour_indices
        gvec_nn[iat] = length(neighbour_indices) + 1
    end

    si = spec_index[sym[iat]]

    npmax3 = isempty(base3) ? 0 : feat3.np
    nqmax3 = isempty(base3) ? 0 : feat3.nq
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

        if !isempty(base3)
            fijtmp = feat3.f(rij, feat3.rcut)
            inv_fij = 1.0 / fijtmp
            for m = 1:feat3.np
                pij[m] = fast_pow(fijtmp, feat3.p[m])
            end
        end

        if !isnothing(core)
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

        isempty(base3) && continue

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
        one_body_offset=feature_size(cf)[1],
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
