import Random

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
    elements;
    nembed::Int=4,
    seed::Integer=0,
    T::DataType=Float64,
    kwargs...,
)
    proto_elements, proto2, proto3, proto_kwargs = _embedded_prototypes(elements; kwargs...)
    embedding = _random_embedding_matrix(T, length(proto_elements), nembed; seed)
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
)
    do_grad && (@assert nl.has_vectors)

    (; fvec, gvec, gvec_index, gvec_nn, stotv, hardcore) = workspace
    (; core, fcore, score, ecore) = hardcore

    if do_grad
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
