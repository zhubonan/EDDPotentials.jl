#=
Routines for building structures
=#

using StatsBase: Weights, sample

_has_symmetry_generate(text::AbstractString) =
    any(line -> uppercase(strip(line)) == "SYMMETRY_GENERATE", split(text, '\n'))

_has_symmetry_generate(lines) =
    any(line -> uppercase(strip(line)) == "SYMMETRY_GENERATE", lines)

function _cell_block(lines, block_name::AbstractString)
    in_block = false
    block_lines = String[]
    block_upper = uppercase(block_name)
    for line in lines
        stripped = strip(line)
        upper = uppercase(stripped)
        if startswith(upper, "%BLOCK")
            parts = split(upper)
            if length(parts) >= 2 && parts[2] == block_upper
                in_block = true
            end
        elseif in_block && startswith(upper, "%ENDBLOCK")
            break
        elseif in_block
            push!(block_lines, String(line))
        end
    end
    block_lines
end

function _setting_true(lines, key::AbstractString)
    key_upper = uppercase(key)
    pattern = Regex("^" * key_upper * raw"\s*:\s*TRUE\b")
    for line in lines
        text = uppercase(strip(split(String(line), "#"; limit=2)[1]))
        occursin(pattern, text) && return true
    end
    false
end

function _parse_inline_position_tags(lines, nat::Integer)
    fix_mask = falses(3, nat)
    nomove = falses(nat)
    pos_blocks = _cell_block(lines, "POSITIONS_ABS")
    isempty(pos_blocks) && (pos_blocks = _cell_block(lines, "POSITIONS_FRAC"))

    iatom = 0
    for line in pos_blocks
        data_and_comment = split(String(line), "#"; limit=2)
        data = strip(data_and_comment[1])
        isempty(data) && continue
        length(split(data)) >= 4 || continue
        iatom += 1
        iatom > nat && break
        length(data_and_comment) == 2 || continue

        comment = uppercase(data_and_comment[2])
        if occursin('%', comment)
            tag_text = split(comment, "%"; limit=2)[2]
            tags = split(tag_text)
            if any(==("FIX"), tags)
                fix_mask[:, iatom] .= true
            end
            if any(==("NOMOVE"), tags)
                nomove[iatom] = true
            end
        end
    end
    fix_mask, nomove
end

function _parse_ionic_constraints(lines, cell)
    block = _cell_block(lines, "IONIC_CONSTRAINTS")
    mask = falses(3, length(cell))
    isempty(block) && return mask

    specie_counts = Dict{Symbol,Int}()
    occurrence_to_atom = Dict{Tuple{Symbol,Int},Int}()
    for (iat, sym) in enumerate(species(cell))
        count = get(specie_counts, sym, 0) + 1
        specie_counts[sym] = count
        occurrence_to_atom[(sym, count)] = iat
    end

    for line in block
        tokens = split(strip(split(String(line), "#"; limit=2)[1]))
        length(tokens) >= 6 || continue
        sym = Symbol(tokens[2])
        occurrence = parse(Int, tokens[3])
        atom = get(occurrence_to_atom, (sym, occurrence), 0)
        atom == 0 && continue
        direction = parse.(Float64, tokens[4:6])
        for dim in 1:3
            abs(direction[dim]) > 1e-8 && (mask[dim, atom] = true)
        end
    end
    mask
end

function _parse_cell_constraint_mask(lines)
    block = _cell_block(lines, "CELL_CONSTRAINTS")
    isempty(block) && return nothing

    values = Int[]
    for line in block
        for token in split(strip(split(String(line), "#"; limit=2)[1]))
            push!(values, parse(Int, token))
        end
    end
    length(values) >= 6 || return nothing
    Tuple(values[1:6] .== 0)
end

function _seed_lattice_cell_constraints(lines)
    fix_tag = nothing
    for block_name in ("LATTICE_CART", "LATTICE_ABC")
        for line in _cell_block(lines, block_name)
            tag = uppercase(strip(String(line)))
            if tag in ("#FIX", "#CFIX", "#ABFIX")
                fix_tag = Symbol(tag[2:end])
            end
        end
    end

    fix_tag === :FIX && return Dict{Symbol,Any}(
        :fix_all_cell => true,
        :fix_volume => false,
        :cell_mask => nothing,
        :source => :seed_lattice_fix,
    )
    fix_tag === :CFIX && return Dict{Symbol,Any}(
        :fix_all_cell => false,
        :fix_volume => false,
        :cell_mask => (false, false, true, true, true, false),
        :source => :seed_lattice_fix,
    )
    fix_tag === :ABFIX && return Dict{Symbol,Any}(
        :fix_all_cell => false,
        :fix_volume => false,
        :cell_mask => (true, true, false, false, false, true),
        :source => :seed_lattice_fix,
    )
    nothing
end

function _record_seed_lattice_constraints!(cell, seed_lines)
    haskey(cell.metadata, :buildcell_cell_constraints) && return cell
    info = _seed_lattice_cell_constraints(seed_lines)
    info === nothing && return cell
    cell.metadata[:buildcell_cell_constraints] = info
    cell
end

function _record_buildcell_constraints!(cell, lines)
    fix_mask, nomove = _parse_inline_position_tags(lines, length(cell))
    fix_mask .|= _parse_ionic_constraints(lines, cell)

    if any(fix_mask)
        cell.arrays[:buildcell_fix_mask] = fix_mask
    end
    if any(nomove)
        cell.arrays[:buildcell_nomove_mask] = nomove
    end

    fix_all_cell = _setting_true(lines, "FIX_ALL_CELL")
    fix_volume = _setting_true(lines, "FIX_VOL")
    cell_mask = _parse_cell_constraint_mask(lines)
    if fix_all_cell || fix_volume || cell_mask !== nothing
        cell.metadata[:buildcell_cell_constraints] = Dict{Symbol,Any}(
            :fix_all_cell => fix_all_cell,
            :fix_volume => fix_volume,
            :cell_mask => cell_mask,
        )
    end
    cell
end

"""
    read_buildcell_cell(lines)

Read buildcell output while preserving AIRSS/CASTEP constraint intent in the
returned `Cell` metadata and arrays.
"""
function read_buildcell_cell(lines)
    cell = CellBase.read_cell(lines)
    _record_buildcell_constraints!(cell, lines)
end

"""
    build_one(seedfile;timeout=10, init_structure_transform=nothing, seed_text_transform=nothing)

Build a single structure via `buildcell`.
"""
function build_one(
    seedfile;
    timeout=10,
    init_structure_transform=nothing,
    max_attemp=100,
    seed_text_transform=nothing,
)
    local cell::Cell{Float64,3}
    i = 1
    seed_text = read(seedfile, String)
    seed_lines = split(seed_text, "\n")
    seed_symmetry_generate = _has_symmetry_generate(seed_text)
    while true
        if i >= max_attemp
            throw(ErrorException("Maximum attempt for building structure exceeded!"))
        end
        this_seed = isnothing(seed_text_transform) ? seed_text : seed_text_transform(seed_text)
        lines = try
            cellout = read(
                pipeline(
                    `timeout $(timeout) buildcell`,
                    stdin=IOBuffer(this_seed),
                    stderr=devnull,
                ),
                String,
            )
            split(cellout, "\n")
        catch err
            if typeof(err) <: ProcessFailedException
                @warn " `buildcell` failed to make the structure"
            else
                throw(err)
            end
        end
        isnothing(lines) && continue

        # Generate a unique label
        cell = read_buildcell_cell(lines)
        _record_seed_lattice_constraints!(cell, seed_lines)
        wants_symmetry_generate = seed_symmetry_generate || _has_symmetry_generate(lines)
        if wants_symmetry_generate
            cell.metadata[:symmetry_generate] = true
        end

        if !isnothing(init_structure_transform)
            cell = init_structure_transform(cell)
            if isnothing(cell)
                # This generated structure is no good....
                i += 1
                continue
            end
            if wants_symmetry_generate
                cell.metadata[:symmetry_generate] = true
            end
        end
        break
    end
    return cell
end


"""
    _select_seed(names::AbstractVector, weights::AbstractVector)

Select a random seed from a vector of names and expand glob pattern if needed, and the resolved paths.

# Examples 

Selecting seeds with pattern `Si-*.cell` with equal weightings.
```julia
_select_seed(["Si-*.cell"], [1.0])
```

"""
function _select_seed(
    names::AbstractVector,
    weights::AbstractVector=repeat([1.0], length(names)),
)

    actual_names = String[]
    actual_weights = Float64[]
    for (i, name) in enumerate(names)
        for j in glob_allow_abs(name)
            push!(actual_names, j)
            if i <= length(weights)
                i_ = i
            else
                i_ = length(weights)
            end
            push!(actual_weights, weights[i_])
        end
    end
    actual_weights ./= sum(actual_weights)
    @assert !isempty(actual_names) "No valid file found with $names"
    sample(actual_names, Weights(actual_weights)), actual_names
end

_select_seed(names::AbstractString, weights=[1.0]) = _select_seed([names], weights)



"""
    build_random_structures(seedfile, outdir;n=1, show_progress=false, timeout=60, seedfile_weights)

Build multiple random structures in the target folder. A glob pattern may be used for the
`seedfile` argument. When `nprocs > 1`, multiple `buildcell` subprocesses are launched
concurrently.

"""
function build_random_structures(
    seedfile::Union{AbstractString,Vector},
    outdir;
    n=1,
    show_progress=false,
    timeout=60,
    outfmt="res",
    seedfile_weights=[1.0],
    nprocs=1,
    seed_text_transform=nothing,
)
    if n <= 0
        return
    end

    if show_progress
        prog = Progress(n)
        prog_lock = ReentrantLock()
    end

    function build_and_write_one!()
        this_seed = _select_seed(seedfile, seedfile_weights)[1]
        cell = build_one(this_seed; timeout, seed_text_transform)
        label = get_label(stem(this_seed))
        cell.metadata[:label] = label
        if outfmt == "res"
            write_res(joinpath(outdir, "$(label).res"), cell)
        elseif outfmt == "cell"
            CellBase.write_cell(joinpath(outdir, "$(label).cell"), cell)
        else
            throw(ArgumentError("Unsupported output format: $outfmt"))
        end

        if show_progress
            lock(prog_lock) do
                ProgressMeter.next!(prog)
            end
        end
        return
    end

    nworkers = min(max(nprocs, 1), n)
    if nworkers == 1
        for _ = 1:n
            build_and_write_one!()
        end
        return
    end

    jobs = Channel{Int}(n)
    for i = 1:n
        put!(jobs, i)
    end
    close(jobs)

    tasks = [
        @async begin
            for _ in jobs
                build_and_write_one!()
            end
        end for _ = 1:nworkers
    ]

    foreach(fetch, tasks)
end
