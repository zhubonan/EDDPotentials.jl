#=
Formula-space utilities for multi-element RSS composition control.
=#

using Random

"""
    FormulaSamplingContext

Precomputed context for formula-space sampling and runtime seed-file rewriting.
"""
struct FormulaSamplingContext
    formulas::Vector{String}
    remove_directives::Vector{String}
end

_seedfiles(seedfile::AbstractString) = [seedfile]
_seedfiles(seedfile::AbstractVector{<:AbstractString}) = collect(seedfile)

"""
    _resolve_seedfiles(workdir, seedfile) -> Vector{String}

Resolve the configured seed file(s), including glob patterns.
"""
function _resolve_seedfiles(workdir::AbstractString, seedfile)
    out = String[]
    for name in _seedfiles(seedfile)
        for path in glob_allow_abs(joinpath(workdir, name))
            push!(out, path)
        end
    end
    isempty(out) && error("No valid seed files found from `state.seedfile=$(seedfile)`")
    out
end

function _resolve_single_seedfile(workdir::AbstractString, seedfile)
    files = _resolve_seedfiles(workdir, seedfile)
    length(files) == 1 || error(
        "Formula sampling requires exactly one seed file, got $(length(files)): $(files)",
    )
    files[1]
end

"""
    _parse_int_range(text) -> Union{Nothing,Tuple{Int,Int}}

Parse an integer range expression such as `1-8`, `8`, `{1,2,2,3}`, or `1,2,3`.
Returns `(min, max)` when parseable.
"""
function _parse_int_range(text::AbstractString)
    s = strip(text)
    isempty(s) && return nothing
    s = replace(s, "{" => "", "}" => "")
    if occursin("-", s)
        m = match(r"^\s*(-?\d+)\s*-\s*(-?\d+)\s*$", s)
        if m !== nothing
            a = parse(Int, m.captures[1])
            b = parse(Int, m.captures[2])
            return (min(a, b), max(a, b))
        end
    end
    if occursin(",", s)
        vals = Int[]
        for tok in split(s, ",")
            t = strip(tok)
            isempty(t) && continue
            m = match(r"^-?\d+$", t)
            m === nothing && return nothing
            push!(vals, parse(Int, t))
        end
        isempty(vals) && return nothing
        return (minimum(vals), maximum(vals))
    end
    m = match(r"^-?\d+$", s)
    m === nothing && return nothing
    v = parse(Int, s)
    return (v, v)
end

"""
    _extract_seed_directive(seed_text, key) -> Union{Nothing,String}

Extract a `#KEY=...` value from a seed file text.
"""
function _extract_seed_directive(seed_text::AbstractString, key::AbstractString)
    for line in split(seed_text, "\n")
        m = match(Regex("(?i)^\\s*#" * key * "\\s*=\\s*(.*)"), line)
        if m !== nothing
            return strip(m.captures[1])
        end
    end
    return nothing
end

"""
    _parse_natom_value(val) -> Union{Nothing,Tuple{Int,Int}}

Parse `#NATOM=...`. Supports simple integer/range only.
Exact per-species directives are intentionally ignored in formula-sampling mode
because composition is controlled by `#FORMULA`.
"""
function _parse_natom_value(val::AbstractString)
    range = _parse_int_range(val)
    range !== nothing && return range
    nothing
end

"""
    _load_seed_constraints(path) -> NamedTuple

Extract NATOM/NFORM constraints from one seed file.
"""
function _load_seed_constraints(path::AbstractString)
    text = read(path, String)
    natom = let x = _extract_seed_directive(text, "NATOM")
        x === nothing ? nothing : _parse_natom_value(x)
    end
    nform = let x = _extract_seed_directive(text, "NFORM")
        x === nothing ? (1, typemax(Int)) : _parse_int_range(x)
    end
    nform === nothing && (nform = (1, typemax(Int)))
    (; natom_range=natom, nform_range=nform)
end

"""
    _formula_atom_count(comp::Composition) -> Int
"""
_formula_atom_count(comp::Composition) = round(Int, sum(comp[e] for e in keys(comp)))

"""
    _fits_seed_constraints(comp, constraints) -> Bool

Whether a reduced formula can satisfy NATOM/NFORM in at least one unit-count.
"""
function _fits_seed_constraints(comp::Composition, constraints)
    f_atoms = _formula_atom_count(comp)
    f_atoms > 0 || return false
    (nform_min, nform_max) = constraints.nform_range
    if constraints.natom_range === nothing
        return nform_min <= nform_max
    end
    (natom_min, natom_max) = constraints.natom_range
    lo = max(nform_min, ceil(Int, natom_min / f_atoms))
    hi = min(nform_max, floor(Int, natom_max / f_atoms))
    lo <= hi
end

"""
    _has_neutral_oxidation_state(comp, oxidation_states) -> Bool

Return true if the composition can be charge neutral under allowed oxidation states.
"""
function _has_neutral_oxidation_state(comp::Composition, oxidation_states::Dict{Symbol,Vector{Int}})
    elems = sort(collect(keys(comp)))
    counts = Int[round(Int, comp[e]) for e in elems]
    ox_lists = Vector{Vector{Int}}()
    for e in elems
        haskey(oxidation_states, e) || return false
        vals = oxidation_states[e]
        isempty(vals) && return false
        push!(ox_lists, vals)
    end

    # Recursive enumeration over element oxidation-state assignments.
    function _search(i::Int, q::Int)
        if i > length(elems)
            return q == 0
        end
        ni = counts[i]
        for ox in ox_lists[i]
            if _search(i + 1, q + ni * ox)
                return true
            end
        end
        return false
    end
    _search(1, 0)
end

"""
    _enumerate_reduced_formulas(elements, max_coeff) -> Vector{Composition}

Enumerate unique reduced formulas over integer coefficients `0:max_coeff`.
"""
function _enumerate_reduced_formulas(elements::Vector{Symbol}, max_coeff::Int)
    max_coeff >= 1 || throw(ArgumentError("`max_reduced_formula_coeff` must be >= 1"))
    ranges = ntuple(_ -> 0:max_coeff, length(elements))
    uniq = Dict{String,Composition}()
    for coeffs in Iterators.product(ranges...)
        vec = Int[coeffs...]
        all(==(0), vec) && continue
        nz = filter(!=(0), vec)
        g = foldl(gcd, nz)
        red = div.(vec, g)
        comp = Composition(Dict(elements[i] => red[i] for i in eachindex(elements) if red[i] > 0))
        rcomp = reduce_composition(comp)
        uniq[string(formula(rcomp))] = rcomp
    end
    sort([c for c in values(uniq)], by=x -> formula(x))
end

"""
    _canonical_formula_list(formulas) -> Vector{Composition}

Parse and canonicalize user-provided formula strings.
"""
function _canonical_formula_list(formulas::Vector{String})
    uniq = Dict{String,Composition}()
    for f in formulas
        c = reduce_composition(Composition(strip(f)))
        uniq[string(formula(c))] = c
    end
    sort([c for c in values(uniq)], by=x -> formula(x))
end

"""
    _build_formula_pool(state, elements) -> Vector{String}

Build the candidate reduced formula list subject to constraints.
"""
function _build_formula_pool(state::BuilderState, elements::Vector{Symbol}; seedfile=nothing)
    if !state.formula_sampling_enabled
        return String[]
    end

    comps = if isempty(state.formula_list)
        _enumerate_reduced_formulas(elements, state.max_reduced_formula_coeff)
    else
        _canonical_formula_list(state.formula_list)
    end

    # Keep formulas containing only configured search elements.
    allowed = Set(elements)
    comps = [c for c in comps if all(in(allowed), keys(c))]

    # Filter by NATOM/NFORM of at least one seed.
    seed_selector = isnothing(seedfile) ? state.seedfile : seedfile
    seed_path = _resolve_single_seedfile(state.workdir, seed_selector)
    seed_constraints = [_load_seed_constraints(seed_path)]
    comps = [
        c for c in comps if any(_fits_seed_constraints(c, cs) for cs in seed_constraints)
    ]

    # Optional oxidation-state neutrality filter.
    if state.require_charge_neutral && !isempty(state.oxidation_states)
        ox = Dict{Symbol,Vector{Int}}(
            Symbol(k) => Int[v for v in vals] for (k, vals) in pairs(state.oxidation_states)
        )
        comps = [c for c in comps if _has_neutral_oxidation_state(c, ox)]
    end

    forms = string.(formula.(comps))
    isempty(forms) && error(
        "Formula sampling is enabled but no valid formula remains after constraints/filtering.",
    )
    sort(forms)
end

"""
    _inject_formula_directive(seed_text, formula_name; remove_directives=["NATOM","SPECIES","FORMULA"])

Inject a `#FORMULA=...` directive while removing conflicting directives.
"""
function _inject_formula_directive(
    seed_text::AbstractString,
    formula_name::AbstractString;
    remove_directives::Vector{String}=String["NATOM", "SPECIES", "FORMULA"],
)
    lines = split(seed_text, "\n")
    filtered = String[]
    for line in lines
        keep = true
        for key in remove_directives
            if occursin(Regex("(?im)^\\s*#" * key * "\\s*="), line)
                keep = false
                break
            end
        end
        keep && push!(filtered, line)
    end
    out = String["#FORMULA=$(formula_name)"]
    append!(out, filtered)
    join(out, "\n")
end

"""
    _formula_sampling_context(state, cf_elements; seedfile=nothing) -> Union{Nothing,FormulaSamplingContext}

Prepare formula sampling context for generation and RSS.
"""
function _formula_sampling_context(
    state::BuilderState,
    cf_elements::Vector{String};
    seedfile=nothing,
)
    state.formula_sampling_enabled || return nothing
    elements = Symbol.(cf_elements)
    formulas = _build_formula_pool(state, elements; seedfile)
    FormulaSamplingContext(formulas, String["NATOM", "SPECIES", "FORMULA"])
end

"""
    _make_seed_text_transform(ctx) -> Union{Nothing,Function}

Return a function that rewrites seed text with one sampled formula per build call.
"""
function _make_seed_text_transform(ctx::Union{Nothing,FormulaSamplingContext})
    ctx === nothing && return nothing
    function _transform(seed_text::AbstractString)
        sampled = rand(ctx.formulas)
        _inject_formula_directive(seed_text, sampled; remove_directives=ctx.remove_directives)
    end
end
