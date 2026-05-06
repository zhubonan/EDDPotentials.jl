#=
Multi-embedding wrapper: combines multiple RandomEmbeddedCellFeature instances
into a single feature specification with M*m total features.

Feature vector layout (per atom):
  [emb1_ob | emb1_2b | emb1_3b | emb2_ob | emb2_2b | emb2_3b | ...]

Each embedding's features are simply concatenated. The workspace uses
`one_body_offset = 0` so that `reset!` zeros the entire fvec, and the
force/stress chain rule iterates over all features safely (one-body features
have zero gvec/stotv, so they contribute nothing).
=#

"""
    MultiRandomEmbeddedCellFeature{E} <: AbstractCellFeatureSpec

Wrapper that combines M independent `RandomEmbeddedCellFeature` instances,
each with a different random embedding matrix. The total feature vector is
the simple concatenation of each embedding's features, giving
`M * nfeatures(single)` total features per atom.

# Constructors

```julia
MultiRandomEmbeddedCellFeature(elements; nembed=4, n_multi=1, seeds=0:n_multi-1, kwargs...)
MultiRandomEmbeddedCellFeature(embeddings::Vector{<:RandomEmbeddedCellFeature})
```
"""
struct MultiRandomEmbeddedCellFeature{E<:RandomEmbeddedCellFeature} <: AbstractCellFeatureSpec
    embeddings::Vector{E}
end

function Base.:(==)(A::MultiRandomEmbeddedCellFeature, B::MultiRandomEmbeddedCellFeature)
    length(A.embeddings) == length(B.embeddings) || return false
    all(A.embeddings .== B.embeddings)
end

function Base.show(io::IO, ::MIME"text/plain", x::MultiRandomEmbeddedCellFeature)
    println(io, "$(typeof(x)) with $(length(x.embeddings)) embeddings:")
    for (i, emb) in enumerate(x.embeddings)
        seed = get(emb.constructor_kwargs, :seed, 0)
        println(io, "  [$i] channels=$(embedding_size(emb)), seed=$seed")
    end
end

Base.show(io::IO, x::MultiRandomEmbeddedCellFeature) =
    Base.show(io, MIME("text/plain"), x)

# --- Constructor from parameters ---

function MultiRandomEmbeddedCellFeature(
    elements;
    nembed::Int=4,
    n_multi::Int=1,
    seeds::AbstractVector{<:Integer}=collect(0:n_multi-1),
    T::DataType=Float64,
    kwargs...,
)
    length(seeds) == n_multi ||
        throw(ArgumentError("Length of seeds ($(length(seeds))) must equal n_multi ($n_multi)"))
    embeddings = [
        RandomEmbeddedCellFeature(elements; nembed, seed=s, T, kwargs...) for s in seeds
    ]
    MultiRandomEmbeddedCellFeature(embeddings)
end

# Lower-level constructor from a vector (with validation)
function MultiRandomEmbeddedCellFeature(embeddings::AbstractVector{<:RandomEmbeddedCellFeature})
    isempty(embeddings) && throw(ArgumentError("At least one embedding is required"))
    ref_elements = sort(embeddings[1].elements)
    for emb in embeddings
        sort(emb.elements) == ref_elements ||
            throw(ArgumentError("All embeddings must have the same element list"))
    end
    MultiRandomEmbeddedCellFeature(collect(embeddings))
end

# --- Interface methods ---

embedding_size(cf::MultiRandomEmbeddedCellFeature) = sum(embedding_size, cf.embeddings)

"""
    one_body_offset(cf)

Return the offset for one-body features in the force/stress chain rule.
For simple concatenation layout, this is 0 (one-body gvec is zero,
so iterating over all features is safe and correct).
"""
one_body_offset(cf::MultiRandomEmbeddedCellFeature) = 0

function feature_size(cf::MultiRandomEmbeddedCellFeature)
    sizes = feature_size.(cf.embeddings)
    (sum(s -> s[1], sizes), sum(s -> s[2], sizes), sum(s -> s[3], sizes))
end

nfeatures(cf::MultiRandomEmbeddedCellFeature) = sum(nfeatures, cf.embeddings)

suggest_rcut(cf::MultiRandomEmbeddedCellFeature; shell=1.0) =
    suggest_rcut(cf.embeddings[1]; shell)

"""
    _nbody_offset(cf, k) -> Int

Return the starting feature index for embedding k's n-body features
(0-indexed). With simple concatenation, this is the sum of all features
from embeddings 1..k-1 plus the one-body features of embedding k.
"""
function _nbody_offset(cf::MultiRandomEmbeddedCellFeature, k::Int)
    feat_before = sum(nfeatures(cf.embeddings[j]) for j in 1:k-1; init=0)
    feat_before + embedding_size(cf.embeddings[k])
end

"""
    _nbody_size(cf, k) -> Int

Number of n-body features for embedding k.
"""
_nbody_size(cf::MultiRandomEmbeddedCellFeature, k::Int) =
    nfeatures(cf.embeddings[k]) - embedding_size(cf.embeddings[k])

# --- one_body_vectors! ---

function one_body_vectors!(v::AbstractMatrix, cell::Cell, cf::MultiRandomEmbeddedCellFeature; offset=0)
    local_offset = offset
    for emb in cf.embeddings
        one_body_vectors!(v, cell, emb; offset=local_offset)
        local_offset += nfeatures(emb)
    end
    v
end

# --- feature_vector ---

function feature_vector(cf::MultiRandomEmbeddedCellFeature, cell::Cell; nmax=500, skin=1.0)
    rcut = suggest_rcut(cf; shell=skin)
    nl = NeighbourList(cell, rcut, nmax; savevec=true)
    vecs = zeros(sum(feature_size(cf)), length(cell))
    wk = get_workspace(cf, nl, false; fvec=vecs)
    compute_fv!(wk, cf, cell; nl)
    one_body_vectors!(vecs, cell, cf)
    vecs
end

# --- get_workspace ---

function get_workspace(
    cf::MultiRandomEmbeddedCellFeature,
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
        one_body_offset=0,
    )
end

# --- compute_fv! (no gradients) ---

function compute_fv!(
    workspace,
    cf::MultiRandomEmbeddedCellFeature,
    cell::Cell;
    nl,
    offset=nothing,
    use_spline=true,
)
    reset!(workspace)
    one_body_vectors!(workspace.fvec, cell, cf)
    for (k, emb) in enumerate(cf.embeddings)
        emb_offset = _nbody_offset(cf, k)
        for iat = 1:natoms(cell)
            compute_embedded_fv_gv_one!(
                workspace, emb, iat, cell, nl;
                do_grad=false, offset=emb_offset,
                compute_hardcore=(k == 1),
            )
        end
    end
    workspace.fvec
end

# --- compute_fv_gv! (with gradients) ---

function compute_fv_gv!(
    workspace,
    cf::MultiRandomEmbeddedCellFeature,
    cell::Cell;
    nl,
    offset=nothing,
    use_spline=true,
)
    reset!(workspace)
    one_body_vectors!(workspace.fvec, cell, cf)
    for iat = 1:natoms(cell)
        if workspace.do_grad
            workspace.gvec[:, :, :, iat] .= 0.0
            workspace.stotv[:, :, :, iat] .= 0.0
        end

        for (k, emb) in enumerate(cf.embeddings)
            emb_offset = _nbody_offset(cf, k)
            compute_embedded_fv_gv_one!(
                workspace, emb, iat, cell, nl;
                offset=emb_offset,
                zero_workspace=false,
                compute_hardcore=(k == 1),
            )
        end
    end
    workspace
end

function compute_fv_gv(cf::MultiRandomEmbeddedCellFeature, cell::Cell; kwargs...)
    nl = get(kwargs, :nl, NeighbourList(cell, suggest_rcut(cf; shell=0.0); savevec=true))
    workspace = get_workspace(cf, nl, true)
    return compute_fv_gv!(workspace, cf, cell; nl)
end

# --- grad_embedding! ---

"""
    grad_embedding!(gE_vec, cf, cell, nl, gv)

Compute gradient of total energy w.r.t. each embedding matrix.
`gE_vec` is a `Vector{Matrix{T}}` with one matrix per embedding.
"""
function grad_embedding!(
    gE_vec::Vector{<:AbstractMatrix{T}},
    cf::MultiRandomEmbeddedCellFeature,
    cell::Cell,
    nl::NeighbourList,
    gv::AbstractMatrix{T};
    offset=nothing,
) where {T}
    for (i, emb) in enumerate(cf.embeddings)
        emb_offset = _nbody_offset(cf, i)
        grad_embedding!(gE_vec[i], emb, cell, nl, gv; offset=emb_offset)
    end
    gE_vec
end

# --- recompute_features! ---

function recompute_features!(
    fvec_out,
    cf::MultiRandomEmbeddedCellFeature,
    cell::Cell,
    workspace,
    nl,
)
    compute_fv!(workspace, cf, cell; nl)  # already calls one_body_vectors!
    if fvec_out !== workspace.fvec
        fvec_out .= workspace.fvec
    end
    return fvec_out
end

# --- compute_feature_stats! ---

function compute_feature_stats!(
    mean_vec::AbstractVector{T},
    std_vec::AbstractVector{T},
    structures,
    cf::MultiRandomEmbeddedCellFeature,
    workspaces,
    nls,
) where {T}
    nf = nfeatures(cf)
    fill!(mean_vec, zero(T))
    fill!(std_vec, zero(T))
    n_atoms_total = 0

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
    for j in eachindex(std_vec)
        if std_vec[j] < 1e-6
            std_vec[j] = 1.0
        end
    end
    return mean_vec, std_vec
end
