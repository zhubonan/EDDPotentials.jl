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
    --ft-steps N    Number of steps per Fortran run (default: 5, amortizes alloc)
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
n_ft_steps = 5
julia_only = false
fortran_only = false

for i in 1:length(ARGS)
    if ARGS[i] == "--repose" && i < length(ARGS)
        global repose_bin = ARGS[i+1]
    elseif ARGS[i] == "--ompnp" && i < length(ARGS)
        global omp_threads = parse(Int, ARGS[i+1])
    elseif ARGS[i] == "--runs" && i < length(ARGS)
        global n_runs = parse(Int, ARGS[i+1])
    elseif ARGS[i] == "--ft-steps" && i < length(ARGS)
        global n_ft_steps = parse(Int, ARGS[i+1])
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

    atom_species_ints = [cf.spec_to_idx[s] for s in species(cell)]

    rcut = EDDPotentialsCore.suggest_rcut(cf; shell=4)
    nl = NeighbourList(cell, rcut; savevec=true)
    avg_nn = sum(nl.nneigh) / nat
    println("  rcut=$rcut, avg neighbours/atom=$(round(avg_nn, digits=1))")

    ws_feat = EDDPotentialsCore.get_workspace(cf, nl, false)
    ws_grad = EDDPotentialsCore.get_workspace(cf, nl, true)

    function julia_compute!(ws, do_grad, cell, nl, cf, atom_species_ints; use_spline=true)
        EDDPotentialsCore.reset!(ws)
        for iat in 1:natoms(cell)
            EDDPotentialsCore.compute_fv_gv_one!(
                ws, cf, iat, cell, nl;
                offset=ws.one_body_offset, do_grad=do_grad,
                atom_species_ints=atom_species_ints,
                use_spline=use_spline,
            )
        end
        ws
    end

    n_warmup = max(3, n_runs ÷ 2)
    for use_spline in (true, false)
        label = use_spline ? "spline" : "direct"
        println("\n  features only ($label), $n_warmup warmup + $n_runs timed:")
        for _ in 1:n_warmup
            julia_compute!(ws_feat, false, cell, nl, cf, atom_species_ints; use_spline)
        end
        feat_times = Float64[]
        for i in 1:n_runs
            GC.gc()
            t = @elapsed julia_compute!(ws_feat, false, cell, nl, cf, atom_species_ints; use_spline)
            push!(feat_times, t)
            println("    run $i: $(round(t, digits=4)) s  ($(round(t/nat*1e6, digits=1)) us/atom)")
        end
        if use_spline
            global jl_feat_best = minimum(feat_times)
        else
            global jl_feat_direct_best = minimum(feat_times)
        end

        println("\n  features + gradients ($label), $n_warmup warmup + $n_runs timed:")
        for _ in 1:n_warmup
            julia_compute!(ws_grad, true, cell, nl, cf, atom_species_ints; use_spline)
        end
        grad_times = Float64[]
        for i in 1:n_runs
            GC.gc()
            t = @elapsed julia_compute!(ws_grad, true, cell, nl, cf, atom_species_ints; use_spline)
            push!(grad_times, t)
            println("    run $i: $(round(t, digits=4)) s  ($(round(t/nat*1e6, digits=1)) us/atom)")
        end
        if use_spline
            global jl_grad_best = minimum(grad_times)
        else
            global jl_grad_direct_best = minimum(grad_times)
        end
    end

    if nthreads() > 1
        for use_spline in (true, false)
            label = use_spline ? "spline" : "direct"
            println("\n  threaded ($(nthreads()) threads), features only ($label):")
            ws_mt = EDDPotentialsCore.get_workspace(cf, nl, false)
            function julia_compute_mt!(ws, cf, cell, nl, atom_species_ints; use_spline=true)
                EDDPotentialsCore.reset!(ws)
                Threads.@threads for iat in 1:natoms(cell)
                    EDDPotentialsCore.compute_fv_gv_one!(
                        ws, cf, iat, cell, nl;
                        offset=ws.one_body_offset, do_grad=false,
                        atom_species_ints=atom_species_ints,
                        use_spline=use_spline,
                    )
                end
                ws
            end
            for _ in 1:n_warmup
                julia_compute_mt!(ws_mt, cf, cell, nl, atom_species_ints; use_spline)
            end
            jl_mt_times = Float64[]
            for i in 1:n_runs
                GC.gc()
                t = @elapsed julia_compute_mt!(ws_mt, cf, cell, nl, atom_species_ints; use_spline)
                push!(jl_mt_times, t)
                println("    run $i: $(round(t, digits=4)) s  ($(round(t/nat*1e6, digits=1)) us/atom)")
            end
            if use_spline
                global jl_mt_best = minimum(jl_mt_times)
            else
                global jl_mt_direct_best = minimum(jl_mt_times)
            end
        end
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
            println("\n  --- repose -ompnp $n_omp -m $n_ft_steps ---")
            times_list = Float64[]

            for irun in 1:n_runs
                try
                    errfile = joinpath(BENCH_DIR, "stderr_$(n_omp)_$(irun).txt")
                    wrapper = joinpath(BENCH_DIR, "run_$(n_omp)_$(irun).sh")
                    write(wrapper, "#!/bin/sh\ncd \"$(BENCH_DIR)\"\n\"$(repose_bin)\" -q -ompnp $(n_omp) -m $(n_ft_steps) SiO2 2>\"$(errfile)\"\n")
                    chmod(wrapper, 0o755)
                    run(`sh $wrapper`)

                    err_str = isfile(errfile) ? read(errfile, String) : ""
                    ball_t = feat_t = netw_t = ddp_t = NaN
                    for line in split(err_str, "\n")
                        toks = split(line)
                        length(toks) < 3 && continue
                        val_idx = toks[2] == ":" ? 3 : 2
                        if occursin("-ball", line)
                            ball_t = parse(Float64, toks[val_idx])
                        elseif occursin("-feat", line)
                            feat_t = parse(Float64, toks[val_idx])
                        elseif occursin("-netw", line)
                            netw_t = parse(Float64, toks[val_idx])
                        elseif occursin("ddp", line)
                            ddp_t = parse(Float64, toks[val_idx])
                        end
                    end

                    feat_t_per_step = feat_t / n_ft_steps
                    feat_us = round(feat_t_per_step / nat * 1e6, digits=1)
                    println("    run $irun: " *
                            "ball=$(round(ball_t*1000/n_ft_steps, digits=1))ms  " *
                            "feat=$(round(feat_t_per_step*1000, digits=1))ms/step  " *
                            "netw=$(round(netw_t*1000/n_ft_steps, digits=1))ms  " *
                            "ddp=$(round(ddp_t*1000/n_ft_steps, digits=1))ms  ($feat_us us/atom)")
                    push!(times_list, feat_t_per_step)
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
    println("  Julia spline (1 thread, feat only):    $(round(jl_feat_best*1000, digits=1)) ms  ($(round(jl_feat_best/nat*1e6, digits=1)) us/atom)")
end
if @isdefined(jl_feat_direct_best)
    println("  Julia direct (1 thread, feat only):    $(round(jl_feat_direct_best*1000, digits=1)) ms  ($(round(jl_feat_direct_best/nat*1e6, digits=1)) us/atom)")
    if @isdefined(jl_feat_best)
        ratio = jl_feat_direct_best / jl_feat_best
        println("    --> spline speedup: $(round(ratio, digits=2))x")
    end
end
if @isdefined(jl_grad_best)
    println("  Julia spline (1 thread, feat+grad):    $(round(jl_grad_best*1000, digits=1)) ms  ($(round(jl_grad_best/nat*1e6, digits=1)) us/atom)")
end
if @isdefined(jl_grad_direct_best)
    println("  Julia direct (1 thread, feat+grad):    $(round(jl_grad_direct_best*1000, digits=1)) ms  ($(round(jl_grad_direct_best/nat*1e6, digits=1)) us/atom)")
    if @isdefined(jl_grad_best)
        ratio = jl_grad_direct_best / jl_grad_best
        println("    --> spline speedup: $(round(ratio, digits=2))x")
    end
end
if @isdefined(jl_mt_best)
    println("  Julia spline ($(nthreads()) threads, feat only):  $(round(jl_mt_best*1000, digits=1)) ms  ($(round(jl_mt_best/nat*1e6, digits=1)) us/atom)")
end
if @isdefined(jl_mt_direct_best)
    println("  Julia direct ($(nthreads()) threads, feat only):  $(round(jl_mt_direct_best*1000, digits=1)) ms  ($(round(jl_mt_direct_best/nat*1e6, digits=1)) us/atom)")
end

if @isdefined(ft_results) && !isempty(ft_results)
    for (key, times) in sort(collect(ft_results))
        best = minimum(times)
        best_us = round(best / nat * 1e6, digits=1)
        omp_n = match(r"omp(\d+)", key).captures[1]
        println("  Fortran (OMP=$omp_n, feat):  $(round(best*1000, digits=1)) ms  ($best_us us/atom)")
        if @isdefined(jl_grad_best)
            ratio = jl_grad_best / best
            println("    --> Julia spline feat+grad / Fortran feat: $(round(ratio, digits=2))x")
        end
    end
end

println()
