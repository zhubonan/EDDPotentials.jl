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
    varvol_by_formula::Dict{String,Float64}
end

function _sample_formula_seed(ctx::FormulaSamplingContext, seed_text::AbstractString)
    sampled = rand(ctx.formulas)
    varvol = get(ctx.varvol_by_formula, sampled, nothing)
    rewritten = _inject_formula_directive(
        seed_text,
        sampled;
        remove_directives=ctx.remove_directives,
        varvol,
    )
    return (; formula=sampled, varvol, seed_text=rewritten)
end

function _print_formula_sampling_sample(
    io::IO,
    idx::Int,
    sample,
    seedfile_path::AbstractString,
)
    println(io, "Sample $idx")
    println(io, "----------------------------------------")
    println(io, "User settings")
    println(io, "formula = $(sample.formula)")
    if !isnothing(sample.varvol)
        println(io, "varvol = $(sample.varvol)")
    end
    println(io, "seedfile = $(seedfile_path)")
    println(io, "----------------------------------------")
    println(io, "buildcell input")
    println(io, sample.seed_text)
    println(io, "----------------------------------------")
end

function _run_buildcell_diagnostic(seed_text::AbstractString; timeout::Real=10)
    mktemp() do _, io
        write(io, seed_text)
        seekstart(io)
        return read(
            pipeline(
                `timeout $(timeout) buildcell`,
                stdin=io,
                stderr=devnull,
            ),
            String,
        )
    end
end

function _resolve_formula_diagnostic_save_dir(
    save,
    state::BuilderState,
)
    save === false && return nothing
    save === true && return joinpath(state.workdir, "formula_sampling_diagnostics")
    save isa AbstractString && return String(save)
    throw(ArgumentError("`save` must be `false`, `true`, or a directory path string"))
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
    isempty(out) && error("No valid seed files found from RSS seedfile setting: $(seedfile)")
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
    max_coeff >= 1 || throw(ArgumentError("`formula_max_reduced_formula_coeff` must be >= 1"))
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
    _build_formula_pool(rss, elements) -> Vector{String}

Build the candidate reduced formula list subject to constraints.
"""
function _build_formula_pool(rss::RssSetting, elements::Vector{Symbol}; seedfile=nothing)
    if !rss.formula_sampling_enabled
        return String[]
    end

    comps = if isempty(rss.formula_list)
        _enumerate_reduced_formulas(elements, rss.formula_max_reduced_formula_coeff)
    else
        _canonical_formula_list(rss.formula_list)
    end

    # Keep formulas containing only configured search elements.
    allowed = Set(elements)
    comps = [c for c in comps if all(in(allowed), keys(c))]

    # Optional oxidation-state neutrality filter.
    if rss.formula_require_charge_neutral && !isempty(rss.formula_oxidation_states)
        ox = Dict{Symbol,Vector{Int}}(
            Symbol(k) => Int[v for v in vals] for (k, vals) in pairs(rss.formula_oxidation_states)
        )
        comps = [c for c in comps if _has_neutral_oxidation_state(c, ox)]
    end

    forms = string.(formula.(comps))
    isempty(forms) && error(
        "Formula sampling is enabled but no valid formula remains after constraints/filtering.",
    )
    sort(forms)
end

function _formula_state_rss(state::BuilderState)
    rs = deepcopy(state.rss)
    if state.seedfile !== nothing && rs.seedfile == "null"
        rs.seedfile = deepcopy(state.seedfile)
    end
    if state.formula_sampling_enabled && !rs.formula_sampling_enabled
        rs.formula_sampling_enabled = true
        rs.formula_max_reduced_formula_coeff = state.formula_max_reduced_formula_coeff
        rs.formula_list = deepcopy(state.formula_list)
        rs.formula_target_atom_volumes = deepcopy(state.formula_target_atom_volumes)
        rs.formula_oxidation_states = deepcopy(state.formula_oxidation_states)
        rs.formula_require_charge_neutral = state.formula_require_charge_neutral
    end
    return rs
end

_build_formula_pool(state::BuilderState, elements::Vector{Symbol}; seedfile=nothing) =
    _build_formula_pool(_formula_state_rss(state), elements; seedfile)

"""
    _inject_formula_directive(seed_text, formula_name; remove_directives=["NATOM","SPECIES","FORMULA"], varvol=nothing)

Inject a `#FORMULA=...` directive while removing conflicting directives.
When `varvol` is provided, also inject `#VARVOL=...`.
"""
function _inject_formula_directive(
    seed_text::AbstractString,
    formula_name::AbstractString;
    remove_directives::Vector{String}=String["NATOM", "SPECIES", "FORMULA"],
    varvol=nothing,
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
    if !isnothing(varvol)
        push!(out, "#VARVOL=$(_format_seed_value(Float64(varvol)))")
    end
    append!(out, filtered)
    join(out, "\n")
end

function _format_seed_value(x::Float64)
    xi = round(Int, x)
    isapprox(x, xi; atol=0.0, rtol=sqrt(eps(Float64))) && return string(xi)
    return string(x)
end

function _formula_varvol(
    formula_name::AbstractString,
    target_atom_volumes::Dict{Symbol,Float64},
)
    comp = reduce_composition(Composition(strip(formula_name)))
    vol = 0.0
    missing = Symbol[]
    for el in sort!(collect(keys(comp)))
        if haskey(target_atom_volumes, el)
            vol += target_atom_volumes[el]
        else
            push!(missing, el)
        end
    end
    isempty(missing) || error(
        "Missing `formula_target_atom_volumes` entries for formula `$(formula_name)`: $(join(string.(missing), ", "))",
    )
    return vol
end

function _formula_varvol_map(
    formulas::Vector{String},
    target_atom_volumes::Dict{Symbol,Float64},
)
    out = Dict{String,Float64}()
    for f in formulas
        out[f] = _formula_varvol(f, target_atom_volumes)
    end
    return out
end

"""
    _formula_sampling_context(workdir, rss, cf_elements; seedfile=nothing) -> Union{Nothing,FormulaSamplingContext}

Prepare formula sampling context for generation and RSS.
"""
function _formula_sampling_context(
    workdir::AbstractString,
    rss::RssSetting,
    cf_elements::Vector{String};
    seedfile=nothing,
)
    rss.formula_sampling_enabled || return nothing
    seedfile_cfg = isnothing(seedfile) ? rss.seedfile : seedfile
    _resolve_single_seedfile(workdir, seedfile_cfg)
    elements = Symbol.(cf_elements)
    formulas = _build_formula_pool(rss, elements; seedfile)
    target_atom_volumes = Dict{Symbol,Float64}(
        Symbol(k) => Float64(v) for (k, v) in pairs(rss.formula_target_atom_volumes)
    )
    varvol_by_formula = isempty(target_atom_volumes) ?
                        Dict{String,Float64}() :
                        _formula_varvol_map(formulas, target_atom_volumes)
    FormulaSamplingContext(
        formulas,
        String["NATOM", "SPECIES", "FORMULA", "VARVOL", "TARGVOL"],
        varvol_by_formula,
    )
end

function _formula_sampling_context(
    state::BuilderState,
    cf_elements::Vector{String};
    seedfile=nothing,
)
    _formula_sampling_context(state.workdir, _formula_state_rss(state), cf_elements; seedfile)
end

"""
    _make_seed_text_transform(ctx) -> Union{Nothing,Function}

Return a function that rewrites seed text with one sampled formula per build call.
"""
function _make_seed_text_transform(ctx::Union{Nothing,FormulaSamplingContext})
    ctx === nothing && return nothing
    function _transform(seed_text::AbstractString)
        _sample_formula_seed(ctx, seed_text).seed_text
    end
end

"""
    diagnose_formula_sampling(state, cf_elements; seedfile=nothing, nsamples=3, io=stdout, save=false)

Print sampled formulas and rewritten seed text using the same formula-sampling code
path as RSS/buildcell generation. Returns a vector of named tuples
`(formula, varvol, seed_text, seedfile_path, saved_seed_path, saved_buildcell_output_path)`.
"""
function diagnose_formula_sampling(
    state::BuilderState,
    cf_elements::Vector{String};
    seedfile=nothing,
    nsamples::Int=3,
    io::IO=stdout,
    save=false,
)
    nsamples >= 1 || throw(ArgumentError("`nsamples` must be >= 1"))
    ctx = _formula_sampling_context(state, cf_elements; seedfile)
    ctx === nothing && error("Formula sampling is disabled for this BuilderState.")

    rs = _formula_state_rss(state)
    seedfile_cfg = isnothing(seedfile) ? rs.seedfile : seedfile
    seedfile_path = _resolve_single_seedfile(state.workdir, seedfile_cfg)
    seed_text = read(seedfile_path, String)
    save_dir = _resolve_formula_diagnostic_save_dir(save, state)
    if !isnothing(save_dir)
        mkpath(save_dir)
    end
    results = NamedTuple[]
    for i in 1:nsamples
        sample = _sample_formula_seed(ctx, seed_text)
        saved_seed_path = nothing
        saved_buildcell_output_path = nothing
        if !isnothing(save_dir)
            stem = lpad(string(i), 3, '0')
            saved_seed_path = joinpath(save_dir, "sample-$(stem)-seed.cell")
            write(saved_seed_path, sample.seed_text)
            buildcell_output = _run_buildcell_diagnostic(
                sample.seed_text;
                timeout=state.build_timeout,
            )
            saved_buildcell_output_path = joinpath(save_dir, "sample-$(stem)-buildcell.cell")
            write(saved_buildcell_output_path, buildcell_output)
        end
        result = (;
            sample...,
            original_seed_text=seed_text,
            seedfile_path,
            saved_seed_path,
            saved_buildcell_output_path,
        )
        push!(results, result)
        _print_formula_sampling_sample(io, i, result, seedfile_path)
        if !isnothing(saved_seed_path)
            println(io, "saved_seed = $(saved_seed_path)")
            println(io, "saved_buildcell_output = $(saved_buildcell_output_path)")
            println(io, "----------------------------------------")
        end
    end
    return results
end
