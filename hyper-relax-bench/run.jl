include("systems.jl")
include("protocols.jl")

using .Systems
using .Protocols
using EDDPotentials
using CellBase
using Unitful: ustrip
using StatsBase: mean, median

const RESULTS_DIR = joinpath(@__DIR__, "results")

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

    function pad(s, w)
        s = string(s)
        rpad(s, w)
    end

    line = join(map(w -> repeat("─", w + 2), widths), "┼")
    println("  ", join(map((h, w) -> pad(h, w), headers, widths), " │ "))
    println("  ", line)
    for row in rows
        println("  ", join(map((v, w) -> pad(v, w), row, widths), " │ "))
    end
end

function _run_protocol(fn, calc, cell, outdir, stem, proto)
    try
        m = fn(calc, cell, outdir)
        m["structure"] = stem
        @info "    done" converged=m["converged"] iter=m["iterations"] wall=round(m["wall_s"]; digits=1)
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

function _run_all(calc, structures; skip_done=true)
    all_metrics = Dict{String,Any}[]
    skipped = 0

    for (idx, (label, cell)) in enumerate(structures)
        stem = label
        @info "[$idx/$(length(structures))]" stem volume=ustrip(CellBase.volume(cell)) nions=CellBase.nions(cell)

        struct_dir = joinpath(RESULTS_DIR, stem)

        svc_done = isfile(joinpath(struct_dir, "standard-vc", "relaxed.extxyz"))
        h1d_done = isfile(joinpath(struct_dir, "hyper-1d", "relaxed.extxyz"))

        if skip_done && svc_done && h1d_done
            skipped += 1
            continue
        end

        if !svc_done
            @info "  Running standard-vc..."
            m1 = _run_protocol(Protocols.run_standard_vc, calc, cell,
                               joinpath(struct_dir, "standard-vc"), stem, "standard-vc")
            push!(all_metrics, m1)
        end

        if !h1d_done
            @info "  Running hyper-1d..."
            m2 = _run_protocol(Protocols.run_hyper_1d, calc, cell,
                               joinpath(struct_dir, "hyper-1d"), stem, "hyper-1d")
            push!(all_metrics, m2)
        end

        idx % 10 == 0 && (GC.gc(); @info "  GC done, continuing...")
    end

    @info "Completed: $(length(all_metrics)) new runs, $skipped skipped"
    return all_metrics
end

function _collect_results(structures)
    all_metrics = Dict{String,Any}[]
    for (label, _) in structures
        stem = label
        struct_dir = joinpath(RESULTS_DIR, stem)
        for proto in ("standard-vc", "hyper-1d")
            relaxed = joinpath(struct_dir, proto, "relaxed.extxyz")
            if isfile(relaxed)
                m = _read_result_json(struct_dir, proto)
                if m !== nothing
                    push!(all_metrics, m)
                end
            end
        end
    end
    return all_metrics
end

function _read_result_json(struct_dir, proto)
    result_file = joinpath(struct_dir, proto, "result.json")
    if isfile(result_file)
        return JSON.parsefile(result_file)
    end
    nothing
end

function _analyze(all_metrics)
    svc = filter(m -> m["protocol"] == "standard-vc" && !haskey(m, "error"), all_metrics)
    h1d = filter(m -> m["protocol"] == "hyper-1d" && !haskey(m, "error"), all_metrics)

    headers = ["structure", "conv", "iter", "frames", "E_final", "vol_ratio", "fmax", "min_dist", "wall_s"]

    for (proto_label, data) in [("standard-vc", svc), ("hyper-1d", h1d)]
        isempty(data) && continue
        println("\n--- ", proto_label, " ---")
        rows = Vector{String}[]
        for m in data
            push!(rows, [
                m["structure"],
                get(m, "error", nothing) !== nothing ? "ERR" : (m["converged"] ? "Y" : "N"),
                string(m["iterations"]),
                string(m["n_traj_frames"]),
                _fmt(Float64(m["final_energy"]); digits=2),
                _fmt(Float64(m["vol_ratio"])),
                _fmt(Float64(m["fmax"]); digits=4),
                _fmt(Float64(m["min_dist"])),
                _fmt(Float64(m["wall_s"]); digits=1),
            ])
        end
        _print_table(headers, rows)
    end

    matched = intersect([m["structure"] for m in svc], [m["structure"] for m in h1d])
    matched_svc = [m for m in svc if m["structure"] in matched]
    matched_h1d = [m for m in h1d if m["structure"] in matched]

    n_hyper_wins = 0
    n_svc_wins = 0
    n_ties = 0

    if !isempty(matched_svc) && !isempty(matched_h1d)
        println("\n--- Comparison (hyper-1d vs standard-vc) ---")
        comp_headers = ["structure", "winner", "dE", "d_vol_ratio", "d_iter", "wall_ratio"]
        rows = Vector{String}[]
        dE_all = Float64[]
        dvol_all = Float64[]
        diter_all = Int[]
        wratio_all = Float64[]

        for i in 1:length(matched_svc)
            s = matched_svc[i]
            h = matched_h1d[i]
            dE = Float64(h["final_energy"]) - Float64(s["final_energy"])
            dvol = Float64(h["vol_ratio"]) - Float64(s["vol_ratio"])
            diter = Int(h["iterations"]) - Int(s["iterations"])
            wr = Float64(h["wall_s"]) / max(Float64(s["wall_s"]), 0.01)
            push!(dE_all, dE)
            push!(dvol_all, dvol)
            push!(diter_all, diter)
            push!(wratio_all, wr)

            if dE < -0.01
                winner = "hyper"
                n_hyper_wins += 1
            elseif dE > 0.01
                winner = "standard"
                n_svc_wins += 1
            else
                winner = "tie"
                n_ties += 1
            end

            push!(rows, [
                s["structure"],
                winner,
                _fmt(dE; digits=2),
                _fmt(dvol),
                string(diter),
                _fmt(wr; digits=1),
            ])
        end
        _print_table(comp_headers, rows)

        println("\n--- Statistics ($(length(dE_all)) structures) ---")
        n_conv_svc = count(m -> m["converged"], matched_svc)
        n_conv_h1d = count(m -> m["converged"], matched_h1d)
        println("  Convergence:  standard-vc $n_conv_svc/$(length(matched_svc))  hyper-1d $n_conv_h1d/$(length(matched_h1d))")
        println("  Winner:       hyper-1d lowest $n_hyper_wins  |  standard-vc lowest $n_svc_wins  |  tie $n_ties")
        println("  Energy:       hyper better $n_hyper_wins  |  standard better $n_svc_wins  |  similar $n_ties")
        println("  Mean dE:      $(round(mean(dE_all); digits=3)) eV")
        println("  Median dE:    $(round(median(dE_all); digits=3)) eV")
        hyper_wins_dE = filter(<(-0.01), dE_all)
        if !isempty(hyper_wins_dE)
            println("  Mean dE (hyper wins only): $(round(mean(hyper_wins_dE); digits=3)) eV")
        end
        println("  Mean wall ratio: $(round(mean(wratio_all); digits=1))x")
        println("  Mean iter ratio: $(round(mean(diter_all); digits=0))")
        println("  Mean min_dist:  svc=$(round(mean(Float64[m["min_dist"] for m in matched_svc]); digits=3))  h1d=$(round(mean(Float64[m["min_dist"] for m in matched_h1d]); digits=3))")

        _print_energy_ranking(matched_svc, matched_h1d)
    end

    return (n_hyper_wins, n_svc_wins, n_ties)
end

function _print_energy_ranking(matched_svc, matched_h1d)
    entries = Dict{String,NamedTuple}()
    for m in matched_svc
        entries[m["structure"]] = (svc_E=Float64(m["final_energy"]), svc_conv=m["converged"],
                                    svc_iter=m["iterations"], svc_wall=m["wall_s"],
                                    h1d_E=NaN, h1d_conv=false, h1d_iter=0, h1d_wall=NaN)
    end
    for m in matched_h1d
        if haskey(entries, m["structure"])
            e = entries[m["structure"]]
            entries[m["structure"]] = (; e.svc_E, e.svc_conv, e.svc_iter, e.svc_wall,
                                        h1d_E=Float64(m["final_energy"]), h1d_conv=m["converged"],
                                        h1d_iter=m["iterations"], h1d_wall=Float64(m["wall_s"]))
        end
    end

    sorted = sort(collect(entries); by=((_, v),) -> min(v.svc_E, v.h1d_E))
    top_n = min(10, length(sorted))

    println("\n--- Top $top_n lowest-energy structures ---")
    headers = ["structure", "E_svc", "E_h1d", "winner"]
    rows = Vector{String}[]
    for (name, v) in sorted[1:top_n]
        winner = isnan(v.h1d_E) ? "standard" :
                 v.h1d_E < v.svc_E - 0.01 ? "hyper" :
                 v.svc_E < v.h1d_E - 0.01 ? "standard" : "tie"
        push!(rows, [
            name,
            _fmt(v.svc_E; digits=2),
            isnan(v.h1d_E) ? "---" : _fmt(v.h1d_E; digits=2),
            winner,
        ])
    end
    _print_table(headers, rows)
end

function main(; n::Int=100, generate::Bool=false, skip_done::Bool=true)
    @info "Loading model..."
    calc = Systems.load_boron_calculator()

    if generate
        @info "Generating $n random structures via buildcell..."
        structures = Systems.generate_structures(; n=n)
    else
        structures = Systems.select_structures(; n=n)
        @info "Selected $(length(structures)) structures"
    end

    all_metrics = _run_all(calc, structures; skip_done=skip_done)
    _analyze(all_metrics)

    println("\nResults written to: $RESULTS_DIR")
end

function parse_args(args)
    n = 100
    generate = false
    i = 1
    while i <= length(args)
        if args[i] == "--generate"
            generate = true
        elseif args[i] == "--n" && i + 1 <= length(args)
            n = parse(Int, args[i + 1])
            i += 1
        else
            n = parse(Int, args[i])
        end
        i += 1
    end
    return (; n, generate)
end

kwargs = parse_args(ARGS)
main(; kwargs...)
