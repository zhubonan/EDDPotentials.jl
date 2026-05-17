
"""
Core repulsion energy
"""
function core_repulsion(r, c)
    if r < c
        return (2.0 * (c / r - 1.0))^12
    end
    return 0.0
end

"""
Core repulsion gradient
"""
function gcore_repulsion(r, c)
    if r < c
        return -12 * (2 * (c / r - 1.0))^11 / (r * r) * c * 2
    end
    return 0.0
end

struct CoreRepulsion{F,G}
    f::F
    g::G
    "Global fallback cutoff"
    rcut::Float64
    a::Float64
    "Per-element core radii. Empty dict means use rcut for all pairs."
    radii::Dict{Symbol,Float64}
end

"""
    CoreRepulsion(rcut)

Construct a core repulsion with a single cut off radius applied to all element pairs.
"""
function CoreRepulsion(rcut=2.0)
    CoreRepulsion(core_repulsion, gcore_repulsion, Float64(rcut), 1.0, Dict{Symbol,Float64}())
end

"""
    CoreRepulsion(radii::Dict{Symbol,Float64})

Construct a core repulsion with per-element radii. The pair cutoff for atoms of
species `si` and `sj` is computed as `radii[si] + radii[sj]` (sum combining rule).
Species not present in `radii` default to a radius of 0.25 Å.
"""
function CoreRepulsion(radii::Dict{Symbol,Float64})
    max_pair = isempty(radii) ? 2.0 : 2 * maximum(values(radii))
    CoreRepulsion(core_repulsion, gcore_repulsion, max_pair, 1.0, radii)
end

"""
    CoreRepulsion(radii::Dict{String,Float64})

Convenience constructor accepting String keys (e.g. from TOML config).
"""
CoreRepulsion(radii::Dict{String,Float64}) = CoreRepulsion(Dict(Symbol(k) => v for (k, v) in radii))

"""
    _make_core_repulsion(core_size)

Convert a core_size value (Float64 or Dict{String,Float64}) to a CoreRepulsion object.
"""
_make_core_repulsion(core_size::Float64) = CoreRepulsion(core_size)
_make_core_repulsion(core_size::Dict{String,Float64}) = CoreRepulsion(core_size)

"""
    _pair_rcut(core, si, sj)

Compute the pair-specific cutoff for species `si` and `sj`.
Uses sum combining rule: `radii[si] + radii[sj]`.
Falls back to global `rcut` when per-element radii are not defined.
Species missing from `radii` default to 0.25 Å.
"""
function _pair_rcut(core::CoreRepulsion, si::Symbol, sj::Symbol)
    if isempty(core.radii)
        return core.rcut
    end
    return get(core.radii, si, 0.25) + get(core.radii, sj, 0.25)
end

"""
Compute and update the hard core repulsion with species-aware radii.
"""
function _hard_core_update!(ecore, fcore, score, iat, jat, rij, vij, modvij, core, si::Symbol, sj::Symbol)
    pair_rcut = _pair_rcut(core, si, sj)
    pair_rcut == 0.0 && return nothing

    # forces
    # Note that we add the negative size here since F=-dE/dx
    # Each pair is visited twice (i→j and j→i) so apply force on
    # the central atom only with no extra factor.
    gcore = -core.g(rij, pair_rcut) * core.a
    if iat != jat
        for elm in axes(modvij, 1)
            fcore[elm, iat] -= gcore * modvij[elm]
        end
    end
    for elm1 in axes(score, 1)
        for elm2 in axes(score, 2)
            score[elm1, elm2, iat] += vij[elm1] * modvij[elm2] * gcore / 2.0
        end
    end
    ecore[iat] += core.f(rij, pair_rcut) * core.a / 2.0
end
