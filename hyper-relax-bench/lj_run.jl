include("protocols.jl")
include("lj_systems.jl")
include("lj_protocols.jl")

using .LJSystems
using .Protocols
using EDDPotentials
using EDDPotentials: opt_fire, opt_tpsd
using CellBase
using Unitful: ustrip
using StatsBase: mean, median
using LinearAlgebra
using JSON

const RESULTS_DIR = joinpath(@__DIR__, "lj_results_v2")

const PROTOCOLS = [
    ("svc-fire",   opt_fire, "standard-vc"),
    ("svc-tpsd",   opt_tpsd, "standard-vc"),
    ("h1d-fire",   opt_fire, "hyper-1d"),
    ("h1d-tpsd",   opt_tpsd, "hyper-1d"),
    ("h1d-fire-dyn", opt_fire, "hyper-1d-dyn"),
    ("h1d-tpsd-dyn", opt_tpsd, "hyper-1d-dyn"),
]

function _fmt(v; digits=3)
    isnan(v) ? "NaN" : string(round(v; digits=digits))
end

function _print_table(headers, rows)
    widths = [length(h) for h in headers]
    for row in rows
        for (j, v) in enumerate(row)
            widths[j] = max(widths[j], length(string(v)))
        end
    end
    pad(s, w) = rpad(string(s), w)
    line = join(map(w -> repeat("─", w + 2), widths), "┼")
    println("  ", join(map((h, w) -> pad(h, w), headers, widths), " │ "))
    println("  ", line)
    for row in rows
        println("  ", join(map((v, w) -> pad(v, w), row, widths), " │ "))
    end
end

function _run_protocol(fn, cell, outdir, stem, proto; relax_fn=opt_fire, kwargs...)
    try
        m = fn(cell, outdir; relax_fn, kwargs...)
        m["structure"] = stem
        @info "    done" converged=m["converged"] iter=m["iterations"] wall=round(m["wall_s"]; digits=1)
        result_file = joinpath(outdir, "result.json")
        open(result_file, "w") do io
            JSON.print(io, m, 2)
        end
        return m
    catch e
        @warn "    FAILED" proto exception=(e, catch_backtrace())
        return Dict{String,Any}(
            "structure" => stem,
            "protocol" => proto,
            "converged" => false,
            "iterations" => 0,
            "n_traj_frames" => 0,
            "final_energy" => NaN,
            "vol_ratio" => NaN,
            "fmax" => NaN,
            "min_dist" => NaN,
            "wall_s" => NaN,
            "error" => string(typeof(e), ": ", e),
        )
    end
end

function _run_all(structures; skip_done=true)
    all_metrics = Dict{String,Any}[]
    skipped = 0

    for (idx, (label, cell)) in enumerate(structures)
        stem = label
        @info "[$idx/$(length(structures))]" stem volume=ustrip(CellBase.volume(cell)) nions=CellBase.nions(cell)

        struct_dir = joinpath(RESULTS_DIR, stem)

        for (proto_key, relax_fn, _) in PROTOCOLS
            proto_dir = joinpath(struct_dir, proto_key)
            done = isfile(joinpath(proto_dir, "result.json"))

            if skip_done && done
                skipped += 1
                continue
            end

            @info "  Running $proto_key..."
            if startswith(proto_key, "svc-")
                m = _run_protocol(lj_run_standard_vc, cell, proto_dir, stem, proto_key; relax_fn)
            elseif endswith(proto_key, "-dyn")
                m = _run_protocol(lj_run_hyper_1d, cell, proto_dir, stem, proto_key;
                                  relax_fn, dynamic_alpha=true, force_ratio=3.0, max_alpha=1e4)
            else
                m = _run_protocol(lj_run_hyper_1d, cell, proto_dir, stem, proto_key; relax_fn)
            end
            m["protocol_key"] = proto_key
            push!(all_metrics, m)
        end

        idx % 5 == 0 && (GC.gc(); @info "  GC done, continuing...")
    end

    @info "Completed: $(length(all_metrics)) new runs, $skipped skipped"
    return all_metrics
end

function _collect_results(structures)
    all_metrics = Dict{String,Any}[]
    for (label, _) in structures
        stem = label
        struct_dir = joinpath(RESULTS_DIR, stem)
        for (proto_key, _, _) in PROTOCOLS
            result_file = joinpath(struct_dir, proto_key, "result.json")
            if isfile(result_file)
                m = JSON.parsefile(result_file)
                m["protocol_key"] = proto_key
                push!(all_metrics, m)
            end
        end
    end
    return all_metrics
end

function _analyze(all_metrics; fcc_e_per_atom=LJSystems.FCC_E_PER_ATOM)
    headers = ["structure", "proto", "conv", "iter", "E/N", "gap", "fmax", "wall_s"]

    for proto_key in ["svc-fire", "svc-tpsd", "h1d-fire", "h1d-tpsd", "h1d-fire-dyn", "h1d-tpsd-dyn"]
        data = filter(m -> get(m, "protocol_key", "") == proto_key && !haskey(m, "error"), all_metrics)
        isempty(data) && continue
        println("\n--- ", proto_key, " ---")
        rows = Vector{String}[]
        for m in data
            nat = 8
            epa = Float64(m["final_energy"]) / nat
            gap = epa - fcc_e_per_atom
            push!(rows, [
                m["structure"],
                proto_key,
                m["converged"] ? "Y" : "N",
                string(m["iterations"]),
                _fmt(epa; digits=3),
                _fmt(gap; digits=3),
                _fmt(Float64(m["fmax"]); digits=4),
                _fmt(Float64(m["wall_s"]); digits=1),
            ])
        end
        _print_table(headers, rows)
    end

    struct_names = unique([m["structure"] for m in all_metrics if !haskey(m, "error")])
    comparisons = [
        ("svc-fire", "svc-tpsd"),
        ("h1d-fire", "h1d-tpsd"),
        ("h1d-fire", "h1d-fire-dyn"),
        ("h1d-tpsd", "h1d-tpsd-dyn"),
        ("svc-fire", "h1d-fire"),
        ("svc-fire", "h1d-fire-dyn"),
        ("svc-tpsd", "h1d-tpsd-dyn"),
    ]

    for (key_a, key_b) in comparisons
        data_a = Dict(m["structure"] => m for m in all_metrics
                      if get(m, "protocol_key", "") == key_a && !haskey(m, "error"))
        data_b = Dict(m["structure"] => m for m in all_metrics
                      if get(m, "protocol_key", "") == key_b && !haskey(m, "error"))
        matched = intersect(collect(keys(data_a)), collect(keys(data_b)))
        isempty(matched) && continue

        println("\n--- Comparison: $key_a vs $key_b ($(length(matched)) structures) ---")
        comp_headers = ["structure", "winner", "dE/N", "Ea/N", "Eb/N", "d_iter", "wall_ratio"]
        rows = Vector{String}[]
        dE_all = Float64[]
        diter_all = Int[]
        wratio_all = Float64[]
        n_a = 0; n_b = 0; n_tie = 0

        for name in matched
            a = data_a[name]
            b = data_b[name]
            nat = 8
            ea = Float64(a["final_energy"]) / nat
            eb = Float64(b["final_energy"]) / nat
            dE = eb - ea
            diter = Int(b["iterations"]) - Int(a["iterations"])
            wr = Float64(b["wall_s"]) / max(Float64(a["wall_s"]), 0.01)
            push!(dE_all, dE)
            push!(diter_all, diter)
            push!(wratio_all, wr)

            if dE < -0.001
                winner = key_b; n_b += 1
            elseif dE > 0.001
                winner = key_a; n_a += 1
            else
                winner = "tie"; n_tie += 1
            end

            push!(rows, [
                name,
                winner,
                _fmt(dE; digits=3),
                _fmt(ea; digits=3),
                _fmt(eb; digits=3),
                string(diter),
                _fmt(wr; digits=1),
            ])
        end
        _print_table(comp_headers, rows)

        println("\n  Summary ($(length(dE_all)) structures):")
        n_ca = count(m -> m["converged"], [data_a[n] for n in matched])
        n_cb = count(m -> m["converged"], [data_b[n] for n in matched])
        println("  Convergence:   $key_a $n_ca/$(length(matched))  |  $key_b $n_cb/$(length(matched))")
        println("  Winner:        $key_a $n_a  |  $key_b $n_b  |  tie $n_tie")
        println("  Mean dE/N:     $(round(mean(dE_all); digits=4)) ε")
        println("  Median dE/N:   $(round(median(dE_all); digits=4)) ε")
        println("  Mean wall ratio ($key_b/$key_a): $(round(mean(wratio_all); digits=2))x")
    end

    return nothing
end

function main(; n::Int=100, generate::Bool=true, crystals::Bool=true)
    structures = Tuple{String,Cell}[]

    if crystals
        push!(structures, LJSystems.lj_fcc_cell(; nrep=2))
        push!(structures, LJSystems.lj_diamond_cell(; nrep=2))
        push!(structures, LJSystems.lj_bcc_cell(; nrep=2))
        push!(structures, LJSystems.lj_sc_cell(; nrep=2))
        @info "Added $(length(structures)) crystal structures"
    end

    if generate
        gen_dir = joinpath(@__DIR__, "lj_generated_v2")
        if !isdir(gen_dir) || isempty(readdir(gen_dir))
            mkpath(gen_dir)
            @info "Generating $n random LJ structures via buildcell..."
            gen_structs = LJSystems.generate_structures(; n, outdir=gen_dir)
        else
            @info "Loading existing generated structures..."
            gen_structs = LJSystems.load_generated(; n, outdir=gen_dir)
            if length(gen_structs) < n
                @info "Generating $(n - length(gen_structs)) more structures..."
                LJSystems.generate_structures(; n=n - length(gen_structs), outdir=gen_dir)
                gen_structs = LJSystems.load_generated(; n, outdir=gen_dir)
            end
        end
        append!(structures, gen_structs)
    end

    @info "Total structures: $(length(structures))"
    new_metrics = _run_all(structures)

    all_metrics = _collect_results(structures)
    _analyze(all_metrics)
    println("\nResults written to: $RESULTS_DIR")
end

function parse_args(args)
    n = 100
    generate = true
    crystals = true
    i = 1
    while i <= length(args)
        if args[i] == "--generate"
            generate = true
        elseif args[i] == "--no-generate"
            generate = false
        elseif args[i] == "--no-crystals"
            crystals = false
        elseif args[i] == "--n" && i + 1 <= length(args)
            n = parse(Int, args[i + 1])
            i += 1
        else
            n = parse(Int, args[i])
        end
        i += 1
    end
    return (; n, generate, crystals)
end

kwargs = parse_args(ARGS)
main(; kwargs...)
