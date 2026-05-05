
"""
    CellFeature

Collection of Feature specifications and cell.
"""
abstract type AbstractCellFeatureSpec end

"""
    CellFeature{T, F2, G2, F3, G3, NELEM, NP2, NP3, NQ3}

Parametric feature specification for EDDP. Centralizes two-body and three-body
feature parameters (p/q/rcut/f/g) which are identical across all species channels.

# Type Parameters
- `T`: Float type of p2/p3/q3
- `F2`, `G2`: Type of f2/g2 basis functions  
- `F3`, `G3`: Type of f3/g3 basis functions
- `NELEM`: Number of elements
- `NP2`: Number of two-body exponents
- `NP3`: Number of three-body p exponents
- `NQ3`: Number of three-body q exponents
"""
struct CellFeature{T, F2, G2, F3, G3, NELEM, NP2, NP3, NQ3} <: AbstractCellFeatureSpec
    # ── Species ──
    elements::NTuple{NELEM, Symbol}
    spec_to_idx::Dict{Symbol, Int}

    # ── Two-body channels (ordered pairs, row-major) ──
    two_body_species::Vector{NTuple{2, Int}}
    two_body_channel::Matrix{Int}

    # ── Three-body channels (center-ordered, neighbor-unordered) ──
    three_body_species::Vector{NTuple{3, Int}}
    three_body_channel::Array{Int, 3}

    # ── Parameters ──
    rcut2::Float64
    rcut3::Float64
    p2::SVector{NP2, T}
    p3::SVector{NP3, T}
    q3::SVector{NQ3, T}
    f2::F2
    g2::G2
    f3::F3
    g3::G3
end

# ─── Lookup methods ───────────────────────────────────────────────────────────

spec_index(cf::CellFeature, sym::Symbol) = cf.spec_to_idx[sym]

spec_symbol(cf::CellFeature{<:Any, <:Any, <:Any, <:Any, <:Any, NELEM}, idx::Int) where {NELEM} =
    cf.elements[idx]

"""
    two_body_channel(cf, ci::Int, cj::Int) -> Int

Return the channel index (1-based) for the ordered two-body pair (ci, cj).
"""
function two_body_channel(cf::CellFeature, ci::Int, cj::Int)
    cf.two_body_channel[ci, cj]
end

"""
    two_body_range(cf, ci::Int, cj::Int) -> UnitRange{Int}

Return the feature vector range for the ordered two-body pair (ci, cj).
"""
function two_body_range(cf::CellFeature{<:Any, <:Any, <:Any, <:Any, <:Any, NELEM, NP2},
                        ci::Int, cj::Int) where {NELEM, NP2}
    ch = two_body_channel(cf, ci, cj)
    start = NELEM + (ch - 1) * NP2 + 1
    start:(start + NP2 - 1)
end

"""
    two_body_range(cf, si::Symbol, sj::Symbol) -> UnitRange{Int}

Symbol-based variant of two_body_range.
"""
function two_body_range(cf::CellFeature, si::Symbol, sj::Symbol)
    ci = spec_index(cf, si)
    cj = spec_index(cf, sj)
    two_body_range(cf, ci, cj)
end

"""
    three_body_channel(cf, ci::Int, cj::Int, ck::Int) -> Int

Return the channel index (1-based) for the three-body triplet (ci, cj, ck).
Symmetric in cj↔ck (unordered neighbors).
"""
function three_body_channel(cf::CellFeature, ci::Int, cj::Int, ck::Int)
    cf.three_body_channel[ci, cj, ck]
end

"""
    three_body_range(cf, ci::Int, cj::Int, ck::Int) -> UnitRange{Int}

Return the feature vector range for the three-body triplet (ci, cj, ck).
"""
function three_body_range(cf::CellFeature{<:Any, <:Any, <:Any, <:Any, <:Any, NELEM, NP2, NP3, NQ3},
                          ci::Int, cj::Int, ck::Int) where {NELEM, NP2, NP3, NQ3}
    ch = three_body_channel(cf, ci, cj, ck)
    total_fe2 = NELEM^2 * NP2
    tbfe = NP3 * NQ3
    start = NELEM + total_fe2 + (ch - 1) * tbfe + 1
    start:(start + tbfe - 1)
end

"""
    three_body_range(cf, si::Symbol, sj::Symbol, sk::Symbol) -> UnitRange{Int}

Symbol-based variant of three_body_range.
"""
function three_body_range(cf::CellFeature, si::Symbol, sj::Symbol, sk::Symbol)
    ci = spec_index(cf, si)
    cj = spec_index(cf, sj)
    ck = spec_index(cf, sk)
    three_body_range(cf, ci, cj, ck)
end

# ─── Deprecated lookup aliases for backward compatibility ──────────────────────

lookup(cf::CellFeature, i::Symbol) = spec_index(cf, i)

function lookup(cf::CellFeature, i::Symbol, j::Symbol)
    rng = two_body_range(cf, i, j)
    (first(rng), last(rng))
end

function lookup(cf::CellFeature, i::Symbol, j::Symbol, k::Symbol)
    rng = three_body_range(cf, i, j, k)
    (first(rng), last(rng))
end

# ─── Constructor ───────────────────────────────────────────────────────────────

"""
    CellFeature(elements; kwargs...)

High-level specification for EDDP features combining two-body and three-body interactions.

CellFeature is the main interface for defining features in EDDP. It automatically
generates all necessary two-body and three-body channel specifications for the
given elements.

# Keyword Arguments
- `p2=2:8`: Exponents for two-body features
- `p3=p2`: Exponents for three-body p
- `q3=p3`: Exponents for three-body q
- `rcut2=4.0`: Cutoff radius for two-body features
- `rcut3=rcut2`: Cutoff radius for three-body features
- `f2=fr`: Basis function for two-body interactions
- `g2=gfr`: Gradient of f2
- `f3=fr`: Basis function for three-body interactions
- `g3=gfr`: Gradient of f3
- `geometry_sequence=false`: Convert p/q ranges to geometric sequences
"""
function CellFeature(
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
    # Apply geometry sequence for the powers
    if geometry_sequence
        p2 = genp(p2)
        p3 = genp(p3)
        q3 = genp(q3)
    end

    p2t = Tuple(p2)
    p3t = Tuple(p3)
    q3t = Tuple(q3)

    T = isempty(p2t) ? (isempty(p3t) ? (isempty(q3t) ? Float64 : eltype(q3t)) : eltype(p3t)) : eltype(p2t)

    # Sort the elements to ensure stability
    elements_vec = sort(unique(map(Symbol, elements)))
    isempty(elements_vec) && throw(ArgumentError("CellFeature requires at least one element."))

    NELEM = length(elements_vec)
    elements_nt = Tuple(elements_vec)

    spec_to_idx = Dict{Symbol, Int}(s => i for (i, s) in enumerate(elements_vec))

    # Two-body channels: all ordered pairs (ci, cj), row-major
    n2 = NELEM * NELEM
    two_body_species_vec = Vector{NTuple{2, Int}}(undef, n2)
    idx = 1
    for ci in 1:NELEM
        for cj in 1:NELEM
            two_body_species_vec[idx] = (ci, cj)
            idx += 1
        end
    end

    two_body_channel = zeros(Int, NELEM, NELEM)
    for (ch, (ci, cj)) in enumerate(two_body_species_vec)
        two_body_channel[ci, cj] = ch
    end

    # Three-body channels: center-ordered, neighbor-unordered (ci, cj, ck) with cj ≤ ck
    n3 = NELEM * NELEM * (NELEM + 1) ÷ 2
    three_body_species_vec = Vector{NTuple{3, Int}}(undef, n3)
    idx = 1
    for ci in 1:NELEM
        for cj in 1:NELEM
            for ck in cj:NELEM
                three_body_species_vec[idx] = (ci, cj, ck)
                idx += 1
            end
        end
    end

    three_body_channel = zeros(Int, NELEM, NELEM, NELEM)
    for (ch, (ci, cj, ck)) in enumerate(three_body_species_vec)
        three_body_channel[ci, cj, ck] = ch
        three_body_channel[ci, ck, cj] = ch  # symmetry
    end

    return CellFeature(
        elements_nt, spec_to_idx,
        two_body_species_vec, two_body_channel,
        three_body_species_vec, three_body_channel,
        rcut2, rcut3,
        SVector{length(p2t), T}(p2t),
        SVector{length(p3t), T}(p3t),
        SVector{length(q3t), T}(q3t),
        f2, g2, f3, g3,
    )
end

# ─── Equality ──────────────────────────────────────────────────────────────────

function Base.:(==)(A::CellFeature, B::CellFeature)
    A.elements == B.elements || return false
    A.spec_to_idx == B.spec_to_idx || return false
    A.two_body_species == B.two_body_species || return false
    A.two_body_channel == B.two_body_channel || return false
    A.three_body_species == B.three_body_species || return false
    A.three_body_channel == B.three_body_channel || return false
    A.rcut2 == B.rcut2 || return false
    A.rcut3 == B.rcut3 || return false
    A.p2 == B.p2 || return false
    A.p3 == B.p3 || return false
    A.q3 == B.q3 || return false
    A.f2 == B.f2 || return false
    A.g2 == B.g2 || return false
    A.f3 == B.f3 || return false
    A.g3 == B.g3 || return false
    return true
end

# ─── Addition ──────────────────────────────────────────────────────────────────

"""
    +(a::CellFeature, b::CellFeature)

Combine two `CellFeature` objects. Requires matching p2/p3/q3/rcut2/rcut3.
Merges elements and rebuilds species channels.
"""
function Base.:+(a::CellFeature, b::CellFeature)
    a.p2 == b.p2 && a.p3 == b.p3 && a.q3 == b.q3 &&
        a.rcut2 == b.rcut2 && a.rcut3 == b.rcut3 ||
        throw(ArgumentError("Cannot combine CellFeatures with different p/q/rcut."))
    merged = sort(unique([a.elements..., b.elements...]))
    CellFeature(collect(merged);
        p2=a.p2, p3=a.p3, q3=a.q3,
        rcut2=a.rcut2, rcut3=a.rcut3,
        f2=a.f2, g2=a.g2, f3=a.f3, g3=a.g3,
        geometry_sequence=false)
end

# ─── Display ───────────────────────────────────────────────────────────────────

Base.show(io::IO, x::CellFeature) = Base.show(io, MIME("text/plain"), x)

function Base.show(io::IO, ::MIME"text/plain", cf::CellFeature)
    println(io, "$(typeof(cf))")
    println(io, "  Elements:  $(collect(cf.elements))")
    println(io, "  Two-body channels: $(length(cf.two_body_species))  (p2=$(cf.p2))")
    println(io, "  Three-body channels: $(length(cf.three_body_species))  (p3=$(cf.p3), q3=$(cf.q3))")
    println(io, "  rcut2: $(cf.rcut2)  rcut3: $(cf.rcut3)")
end

# ─── Feature names ─────────────────────────────────────────────────────────────

"""
    feature_names(cf::CellFeature; show_value=true)

Return the name for the features.
"""
function feature_names(cf::CellFeature; show_value=true)
    names = String[]
    nelem = length(cf.elements)

    for (ch, (ci, cj)) in enumerate(cf.two_body_species)
        si = cf.elements[ci]
        sj = cf.elements[cj]
        pair = "$(si)-$(sj)"
        for (m, p) in enumerate(cf.p2)
            if show_value
                push!(names, "2b-$pair-$p")
            else
                push!(names, "2b-$pair-$m")
            end
        end
    end

    for (ch, (ci, cj, ck)) in enumerate(cf.three_body_species)
        si = cf.elements[ci]
        sj = cf.elements[cj]
        sk = cf.elements[ck]
        trip = "$(si)-$(sj)-$(sk)"
        for (m, p) in enumerate(cf.p3)
            for (o, q) in enumerate(cf.q3)
                if show_value
                    push!(names, "3b-$trip-$p-$q")
                else
                    push!(names, "3b-$trip-$m-$o")
                end
            end
        end
    end

    return names
end

"""
    nfeatures_per_channel(cf, bodyorder)

Return the number of features per species channel for the given body order (2 or 3).
"""
nfeatures_per_channel(cf::CellFeature{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, NP2, NP3, NQ3}, body) where {NP2, NP3, NQ3} =
    body == 2 ? Int(NP2) : body == 3 ? Int(NP3 * NQ3) : 0

"""
    nchannels(cf, bodyorder)

Return the number of species channels for the given body order (2 or 3).
"""
nchannels(cf::CellFeature{<:Any, <:Any, <:Any, <:Any, <:Any, NELEM, NP2, NP3, NQ3}, body) where {NELEM, NP2, NP3, NQ3} =
    body == 2 ? Int(NELEM^2) : body == 3 ? Int(NELEM * NELEM * (NELEM + 1) ÷ 2) : 0

# ─── Feature counts ────────────────────────────────────────────────────────────

"""
    nfeatures(cf::CellFeature)

Return the total number of features.
"""
function nfeatures(cf::CellFeature{<:Any, <:Any, <:Any, <:Any, <:Any, NELEM, NP2, NP3, NQ3}) where {NELEM, NP2, NP3, NQ3}
    n2 = NELEM^2 * NP2
    n3 = NELEM * NELEM * (NELEM + 1) ÷ 2 * NP3 * NQ3
    NELEM + n2 + n3
end

"""
    nbodyfeatures(cf, nbody)

Return the number of N-body features.
"""
function nbodyfeatures(cf::CellFeature{<:Any, <:Any, <:Any, <:Any, <:Any, NELEM, NP2, NP3, NQ3},
                       nbody) where {NELEM, NP2, NP3, NQ3}
    if nbody == 1
        return NELEM
    elseif nbody == 2
        return NELEM^2 * NP2
    elseif nbody == 3
        return NELEM * NELEM * (NELEM + 1) ÷ 2 * NP3 * NQ3
    end
    return 0
end

"""
    feature_size(cf)

Return (n_onebody, n_twobody, n_threebody).
"""
function feature_size(cf::CellFeature{<:Any, <:Any, <:Any, <:Any, <:Any, NELEM, NP2, NP3, NQ3}) where {NELEM, NP2, NP3, NQ3}
    n2 = NELEM^2 * NP2
    n3 = NELEM * NELEM * (NELEM + 1) ÷ 2 * NP3 * NQ3
    (NELEM, n2, n3)
end

one_body_offset(cf::CellFeature{<:Any, <:Any, <:Any, <:Any, <:Any, NELEM}) where {NELEM} = NELEM

"""
    total_two_body_features(cf)

Return the total number of two-body features (excluding one-body).
"""
total_two_body_features(cf::CellFeature{<:Any, <:Any, <:Any, <:Any, <:Any, NELEM, NP2}) where {NELEM, NP2} =
    NELEM^2 * NP2

# ─── Cutoff ────────────────────────────────────────────────────────────────────

"""
    suggest_rcut(cf; shell=1.0)

Get a suggested cut off radius for NN list.
"""
function suggest_rcut(cf::CellFeature; shell=1.0)
    max(cf.rcut2, cf.rcut3) + shell
end

# ─── Feature vector computation ────────────────────────────────────────────────

"""
    feature_vector(cf::CellFeature, cell::Cell; nmax=500, skin=1.0)

Return a matrix of vectors describing the environment of each atom.
"""
function feature_vector(cf::CellFeature, cell::Cell; nmax=500, skin=1.0)
    rcut = suggest_rcut(cf; shell=skin)
    nl = NeighbourList(cell, rcut, nmax; savevec=true)
    ntotal = nfeatures(cf)
    vecs = zeros(ntotal, length(cell))
    n1 = one_body_offset(cf)
    wk = get_workspace(cf, nl, false; fvec=vecs)
    compute_fv!(wk, cf, cell; nl, offset=n1)
    one_body_vectors!(vecs, cell, cf)
    vecs
end

# ─── One-body vectors ──────────────────────────────────────────────────────────

"""
    one_body_vectors(cell::Cell, cf::CellFeature; offset=0)

Construct one-body features for the structure. One-hot encoding of species.
"""
function one_body_vectors(cell::Cell, cf::CellFeature; offset=0)
    vecs = zeros(length(cf.elements), length(cell))
    one_body_vectors!(vecs, cell, cf; offset)
end

"""
    one_body_vectors!(v, cell::Cell, cf::CellFeature; offset=0)

Fill in-place one-body features.
"""
function one_body_vectors!(v::AbstractMatrix, cell::Cell, cf::CellFeature; offset=0)
    symbols = species(cell)
    for (iat, s) in enumerate(symbols)
        idx = get(cf.spec_to_idx, s, nothing)
        idx !== nothing && (v[idx + offset, iat] = 1.0)
    end
    v
end

# ─── Legacy compat: suggest_rcut for raw feature tuples ────────────────────────

"""
    suggest_rcut(features...; shell=1.0)

Get a suggested rcut for a collection of features.
"""
function suggest_rcut(features...; shell=1.0)
    maximum(x.rcut for x in features) + shell
end
