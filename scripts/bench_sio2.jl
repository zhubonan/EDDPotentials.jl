#!/usr/bin/env julia
#=
Benchmark feature generation: Julia vs Fortran repose on SiO2-1536

Uses the exact same crystal structure and feature parameters as the
SiO2-1536 benchmark in the repose repository.

Usage:
    julia --project=. scripts/bench_sio2.jl [options]

Options:
    --repose PATH   Path to repose binary
    --ompnp N       Number of OMP threads for Fortran (default: 1)
    --runs N        Number of runs (default: 5)
    --julia-only    Only run Julia benchmark
    --fortran-only  Only run Fortran benchmark
=#

using CellBase
using EDDPotentials
using EDDPotentials.EDDPotentialsCore
using Base.Threads

function find_repose_dir()
    for d in [
        normpath(joinpath(@__DIR__, "..", "..", "..", "eddp-dev", "repose")),
        normpath(joinpath(@__DIR__, "..", "..", "eddp-dev", "repose")),
        normpath(joinpath(@__DIR__, "..", "..", "repose")),
    ]
        isdir(d) && return d
    end
    return ""
end

const REPOSE_DIR = find_repose_dir()
const BENCH_DIR = joinpath(REPOSE_DIR, "benchmarks", "SiO2-1536-ramble")

repose_bin = joinpath(REPOSE_DIR, "bin", "repose")
omp_threads = 1
n_runs = 5
julia_only = false
fortran_only = false

for i in 1:length(ARGS)
    if ARGS[i] == "--repose" && i < length(ARGS)
        global repose_bin = ARGS[i+1]
    elseif ARGS[i] == "--ompnp" && i < length(ARGS)
        global omp_threads = parse(Int, ARGS[i+1])
    elseif ARGS[i] == "--runs" && i < length(ARGS)
        global n_runs = parse(Int, ARGS[i+1])
    elseif ARGS[i] == "--julia-only"
        global julia_only = true
    elseif ARGS[i] == "--fortran-only"
        global fortran_only = true
    end
end

println("=" ^ 70)
println("  SiO2-1536 Feature Generation Benchmark")
println("=" ^ 70)

# --- Feature parameters (matching SiO2 ddp metadata) ---
# composition: O-Si  rmax: 5.0  length: 8 64  powers: 2.000 .. 10.000
fortran_powers = EDDPotentialsCore.genp(2, 10, 8)
println("\nFeature spec: 2 species (O,Si), 8 geometric powers, rcut=5.0")
println("  powers: ", round.(fortran_powers, digits=3))
println("  Julia threads: $(nthreads())")

# =============================================================================
# 1. Load SiO2 cell
# =============================================================================

cell_path = joinpath(BENCH_DIR, "SiO2.cell")
if !isfile(cell_path)
    error("SiO2.cell not found at $cell_path")
end
cell = CellBase.read_cell(cell_path)
nat = natoms(cell)
elem_list = sort(unique(cell.symbols))
println("  Cell: $cell_path")
println("  Atoms: $nat, Species: $elem_list")

# =============================================================================
# 2. Julia benchmark
# =============================================================================

if !fortran_only
    println("\n", "-" ^ 50)
    println("  Julia benchmark")
    println("-" ^ 50)

    cf = CellFeature(
        [:O, :Si];
        p2=fortran_powers,
        p3=fortran_powers,
        q3=fortran_powers,
        rcut2=5.0,
        rcut3=5.0,
        geometry_sequence=false,
    )
    nfe = nfeatures(cf)
    println("  CellFeature: $nfe features")
    @assert nfe == 418 "Expected 418 features, got $nfe"

    spec_lookup = EDDPotentialsCore.FeatureSpeciesLookup(cf.two_body, cf.three_body, elem_list)
    atom_species_ints = [spec_lookup.spec_to_idx[s] for s in species(cell)]

    rcut = EDDPotentialsCore.suggest_rcut(cf; shell=4)
    nl = NeighbourList(cell, rcut; savevec=true)
    avg_nn = sum(nl.nneigh) / nat
    println("  rcut=$rcut, avg neighbours/atom=$(round(avg_nn, digits=1))")

    function julia_compute!(do_grad, cell, nl, cf, lookup, atom_species_ints)
        ws = EDDPotentialsCore.get_workspace(cf, nl, do_grad)
        for iat in 1:natoms(cell)
            EDDPotentialsCore.compute_fv_gv_one!(
                ws, cf.two_body, cf.three_body, iat, cell, nl;
                offset=ws.one_body_offset, do_grad=do_grad,
                spec_lookup=lookup, atom_species_ints=atom_species_ints,
            )
        end
        ws
    end

    julia_compute!(false, cell, nl, cf, spec_lookup, atom_species_ints)

    jl_feat_times = Float64[]
    println("\n  features only:")
    for i in 1:n_runs
        t = @elapsed julia_compute!(false, cell, nl, cf, spec_lookup, atom_species_ints)
        push!(jl_feat_times, t)
        println("    run $i: $(round(t, digits=4)) s  ($(round(t/nat*1e6, digits=1)) us/atom)")
    end
    jl_feat_best = minimum(jl_feat_times)

    jl_grad_times = Float64[]
    julia_compute!(true, cell, nl, cf, spec_lookup, atom_species_ints)
    println("\n  features + gradients:")
    for i in 1:n_runs
        t = @elapsed julia_compute!(true, cell, nl, cf, spec_lookup, atom_species_ints)
        push!(jl_grad_times, t)
        println("    run $i: $(round(t, digits=4)) s  ($(round(t/nat*1e6, digits=1)) us/atom)")
    end
    jl_grad_best = minimum(jl_grad_times)

    if nthreads() > 1
        println("\n  threaded ($(nthreads()) threads), features only:")
        ws_mt = EDDPotentialsCore.get_workspace(cf, nl, false)
        function julia_compute_mt!(ws, cf, cell, nl, lookup, atom_species_ints)
            EDDPotentialsCore.reset!(ws)
            Threads.@threads for iat in 1:natoms(cell)
                EDDPotentialsCore.compute_fv_gv_one!(
                    ws, cf.two_body, cf.three_body, iat, cell, nl;
                    offset=ws.one_body_offset, do_grad=false,
                    spec_lookup=lookup, atom_species_ints=atom_species_ints,
                )
            end
            ws
        end
        julia_compute_mt!(ws_mt, cf, cell, nl, spec_lookup, atom_species_ints)
        jl_mt_times = Float64[]
        for i in 1:n_runs
            t = @elapsed julia_compute_mt!(ws_mt, cf, cell, nl, spec_lookup, atom_species_ints)
            push!(jl_mt_times, t)
            println("    run $i: $(round(t, digits=4)) s  ($(round(t/nat*1e6, digits=1)) us/atom)")
        end
        jl_mt_best = minimum(jl_mt_times)
    end
end

# =============================================================================
# 3. Fortran repose benchmark
# =============================================================================

if !julia_only
    println("\n", "-" ^ 50)
    println("  Fortran repose benchmark")
    println("-" ^ 50)

    if !isfile(repose_bin)
        println("  WARNING: repose not found at $repose_bin")
    else
        println("  repose: $repose_bin")
        ft_results = Dict{String, Vector{Float64}}()

        for n_omp in unique([1, omp_threads])
            n_omp > Sys.CPU_THREADS && continue
            println("\n  --- repose -ompnp $n_omp ---")
            times_list = Float64[]

            for irun in 1:n_runs
                try
                    errfile = joinpath(BENCH_DIR, "stderr_$(n_omp)_$(irun).txt")
                    wrapper = joinpath(BENCH_DIR, "run_$(n_omp)_$(irun).sh")
                    write(wrapper, "#!/bin/sh\ncd \"$(BENCH_DIR)\"\n\"$(repose_bin)\" -n -q -ompnp $(n_omp) -m 1 SiO2 2>\"$(errfile)\"\n")
                    chmod(wrapper, 0o755)
                    run(`sh $wrapper`)

                    err_str = isfile(errfile) ? read(errfile, String) : ""
                    # Parse fine-grained Fortran timing with state tracking.
                    # Labels with colons attached (e.g. "--feat:") have the value at
                    # token [2]; labels with a space before colon (e.g. " -ball :")
                    # have the value at token [3].
                    ball_t = f2b_feat_t = f2b_force_t = f2b_stress_t = NaN
                    f3b_feat_t = f3b_force_t = f3b_stress_t = NaN
                    netw_t = NaN
                    state = ""   # "f2" or "f3" when inside a -f2body/-f3body block
                    for line in split(err_str, "\n")
                        toks = split(line)
                        length(toks) < 3 && continue
                        val_idx = toks[2] == ":" ? 3 : 2
                        if occursin("-netw ", line)
                            state = ""
                            netw_t = parse(Float64, toks[val_idx])
                        elseif occursin("-f2body:", line)
                            state = "f2"
                        elseif occursin("-f3body:", line)
                            state = "f3"
                        elseif occursin("--feat:", line)
                            t = parse(Float64, toks[val_idx])
                            state == "f2" && (f2b_feat_t = t)
                            state == "f3" && (f3b_feat_t = t)
                        elseif occursin("--force:", line)
                            t = parse(Float64, toks[val_idx])
                            state == "f2" && (f2b_force_t = t)
                            state == "f3" && (f3b_force_t = t)
                        elseif occursin("--stress:", line)
                            t = parse(Float64, toks[val_idx])
                            state == "f2" && (f2b_stress_t = t)
                            state == "f3" && (f3b_stress_t = t)
                        elseif occursin("-ball", line)
                            ball_t = parse(Float64, toks[val_idx])
                        end
                    end

                    f2b_t = f2b_feat_t + f2b_force_t      # two-body feat+force (excl. stress)
                    f3b_t = f3b_feat_t + f3b_force_t      # three-body feat+force
                    feat_force_t = f2b_t + f3b_t          # total feat+force
                    feat_only_t  = f2b_feat_t + f3b_feat_t # feature values only

                    f2b_us = round(f2b_t / nat * 1e6, digits=1)
                    f3b_us = round(f3b_t / nat * 1e6, digits=1)
                    feat_us = round(feat_force_t / nat * 1e6, digits=1)
                    println("    run $irun: " *
                            "ball=$(round(ball_t*1000, digits=1))ms  " *
                            "f2b(f=$(round(f2b_feat_t*1000, digits=1)) g=$(round(f2b_force_t*1000, digits=1)))ms  " *
                            "f3b(f=$(round(f3b_feat_t*1000, digits=1)) g=$(round(f3b_force_t*1000, digits=1)))ms  " *
                            "netw=$(round(netw_t*1000, digits=1))ms  ($feat_us us/atom)")
                    push!(times_list, feat_force_t)
                catch e
                    println("    run $irun: FAILED - $e")
                end
            end

            if !isempty(times_list)
                ft_results["feat_omp$n_omp"] = times_list
            end
        end
    end
end

# =============================================================================
# 4. Summary
# =============================================================================

println("\n", "=" ^ 70)
println("  Summary ($nat atoms, 418 features)")
println("=" ^ 70)

if @isdefined(jl_feat_best)
    println("  Julia (1 thread, feat only):    $(round(jl_feat_best*1000, digits=1)) ms  ($(round(jl_feat_best/nat*1e6, digits=1)) us/atom)")
end
if @isdefined(jl_grad_best)
    println("  Julia (1 thread, feat+grad):    $(round(jl_grad_best*1000, digits=1)) ms  ($(round(jl_grad_best/nat*1e6, digits=1)) us/atom)")
end
if @isdefined(jl_mt_best)
    println("  Julia ($(nthreads()) threads, feat only):  $(round(jl_mt_best*1000, digits=1)) ms  ($(round(jl_mt_best/nat*1e6, digits=1)) us/atom)")
end

if @isdefined(ft_results) && !isempty(ft_results)
    for (key, times) in sort(collect(ft_results))
        best = minimum(times)
        best_us = round(best / nat * 1e6, digits=1)
        omp_n = match(r"omp(\d+)", key).captures[1]
        println("  Fortran (OMP=$omp_n, feat+force):  $(round(best*1000, digits=1)) ms  ($best_us us/atom)")
        if @isdefined(jl_grad_best)
            ratio = jl_grad_best / best
            println("    --> Julia feat+grad / Fortran: $(round(ratio, digits=2))x")
        end
    end
end

println()
