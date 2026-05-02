#!/usr/bin/env julia
#=
Profile EDDP feature generation on a carbon diamond supercell (~100 atoms).

Usage:
    julia --project=. scripts/bench_feature_gen.jl
    EDDP_OLD_PATH=1 julia --project=. scripts/bench_feature_gen.jl

Compares:
  1. OLD path: per-feature permequal iteration (original code)
  2. NEW path: O(1) species lookup + threaded atom loop

No model file is needed — only CellBase + EDDPotentials.
=#

using CellBase
using EDDPotentials
using EDDPotentials.EDDPotentialsCore
using Base.Threads

function print_sep(label)
    println("\n$(repeat("-", 60))")
    println("  $label")
    println("$(repeat("-", 60))")
end

# === 1. Build diamond supercell =============================================

print_sep("Building carbon diamond supercell")

unit = CellBase.bulk("MgO", "rocksalt"; a=3.0, cubic=true)   # 8-atom conventional cubic
cell = CellBase.make_supercell(unit, 2, 2, 2)                # 8 × 2 × 2 × 3 = 96 atoms
cell.symbols[3] = :Sr
cell.symbols[15] = :Ti
println("Conventional cell: $(natoms(unit)) atoms")
println("Supercell:         $(natoms(cell)) atoms")

# === 2. Define CellFeature ==================================================

print_sep("Feature specification")

cf = CellFeature([:Mg, :O, :Sr, :Ti]; geometry_sequence=false, rcut2=5.0)
tb_feat = isempty(cf.two_body) ? 0 : nfeatures(cf.two_body[1])
tb_n    = length(cf.two_body)
t3_feat = isempty(cf.three_body) ? 0 : nfeatures(cf.three_body[1])
t3_n    = length(cf.three_body)
nfe     = nfeatures(cf)
println("Total features: $nfe  (1-body: $(length(cf.elements)), " *
        "2-body: $(tb_n)×$(tb_feat)=$(tb_n * tb_feat), " *
        "3-body: $(t3_n)×$(t3_feat)=$(t3_n * t3_feat))")
println("Threads: $(nthreads())")

# === 3. Neighbour lists =====================================================

print_sep("Neighbour list")

rcut     = EDDPotentialsCore.suggest_rcut(cf; shell=2.0)
nl_grad   = NeighbourList(cell, rcut; savevec=true)
nl_nograd = NeighbourList(cell, rcut; savevec=true)
avg_nn    = sum(nl_grad.nneigh) / natoms(cell)
max_nn    = maximum(nl_grad.nneigh)
println("rcut = $rcut Å, avg neighbours/atom = $(round(avg_nn, digits=1)), max = $max_nn")

# === 4. Workspaces & species lookup  ========================================

print_sep("Workspace allocations")

ws_grad   = EDDPotentialsCore.get_workspace(cf, nl_grad, true)
ws_nograd = EDDPotentialsCore.get_workspace(cf, nl_nograd, false)

# Build species lookup separately for manual benchmark control
elements = cf.elements
spec_lookup = EDDPotentialsCore.FeatureSpeciesLookup(cf.two_body, cf.three_body, elements)
atom_species_ints = [spec_lookup.spec_to_idx[s] for s in species(cell)]

println("Gradient workspace: $(round(Base.summarysize(ws_grad) / 1024, digits=1)) KiB")
println("No-grad  workspace: $(round(Base.summarysize(ws_nograd) / 1024, digits=1)) KiB")
println("feat_bufs count: $(length(ws_grad.feat_bufs))")

# === 5. OLD PATH: single-threaded, permequal-based ===========================

print_sep("OLD PATH: single-threaded, per-feature permequal iteration")

function run_old_path!(ws, f2, f3, cell, nl; offset, do_grad)
    EDDPotentialsCore.reset!(ws)
    for iat = 1:natoms(cell)
        EDDPotentialsCore.compute_fv_gv_one!(
            ws, f2, f3, iat, cell, nl;
            offset=offset, do_grad=do_grad, spec_lookup=nothing,
        )
    end
    ws
end

# warm-up
run_old_path!(ws_nograd, cf.two_body, cf.three_body, cell, nl_nograd;
    offset=ws_nograd.one_body_offset, do_grad=false)

println("--- feature vector only (no gradients) ---")
for i in 1:5
    t = @elapsed run_old_path!(ws_nograd, cf.two_body, cf.three_body, cell, nl_nograd;
        offset=ws_nograd.one_body_offset, do_grad=false)
    μs = round(t / natoms(cell) * 1e6, digits=1)
    println("  run $i: $(round(t, digits=4)) s  ($μs μs/atom)")
end

# warm-up
run_old_path!(ws_grad, cf.two_body, cf.three_body, cell, nl_grad;
    offset=ws_grad.one_body_offset, do_grad=true)

println("\n--- feature + gradients ---")
for i in 1:5
    t = @elapsed run_old_path!(ws_grad, cf.two_body, cf.three_body, cell, nl_grad;
        offset=ws_grad.one_body_offset, do_grad=true)
    μs = round(t / natoms(cell) * 1e6, digits=1)
    println("  run $i: $(round(t, digits=4)) s  ($μs μs/atom)")
end

# === 6. NEW PATH: species lookup + threaded atom loop ========================

print_sep("NEW PATH: O(1) species lookup + threaded atom loop")

function run_new_path!(ws, f2, f3, cell, nl, lookup, atom_species_ints; offset, do_grad, threaded)
    EDDPotentialsCore.reset!(ws)
    if threaded
        Threads.@threads for iat = 1:natoms(cell)
            EDDPotentialsCore.compute_fv_gv_one!(
                ws, f2, f3, iat, cell, nl;
                offset=offset, do_grad=do_grad, spec_lookup=lookup, atom_species_ints=atom_species_ints,
            )
        end
    else
        for iat = 1:natoms(cell)
            EDDPotentialsCore.compute_fv_gv_one!(
                ws, f2, f3, iat, cell, nl;
                offset=offset, do_grad=do_grad, spec_lookup=lookup, atom_species_ints=atom_species_ints,
            )
        end
    end
    ws
end

# --- 6a. New path, SINGLE-THREADED (fair comparison) ---
println("--- feature vector only, single-threaded new path ---")
run_new_path!(ws_nograd, cf.two_body, cf.three_body, cell, nl_nograd, spec_lookup, atom_species_ints;
    offset=ws_nograd.one_body_offset, do_grad=false, threaded=false)
for i in 1:5
    t = @elapsed run_new_path!(ws_nograd, cf.two_body, cf.three_body, cell, nl_nograd, spec_lookup, atom_species_ints;
        offset=ws_nograd.one_body_offset, do_grad=false, threaded=false)
    μs = round(t / natoms(cell) * 1e6, digits=1)
    println("  run $i: $(round(t, digits=4)) s  ($μs μs/atom)")
end

println("\n--- feature + gradients, single-threaded new path ---")
run_new_path!(ws_grad, cf.two_body, cf.three_body, cell, nl_grad, spec_lookup, atom_species_ints;
    offset=ws_grad.one_body_offset, do_grad=true, threaded=false)
for i in 1:5
    t = @elapsed run_new_path!(ws_grad, cf.two_body, cf.three_body, cell, nl_grad, spec_lookup, atom_species_ints;
        offset=ws_grad.one_body_offset, do_grad=true, threaded=false)
    μs = round(t / natoms(cell) * 1e6, digits=1)
    println("  run $i: $(round(t, digits=4)) s  ($μs μs/atom)")
end

# --- 6b. New path, THREADED ---
if nthreads() > 1
    print_sep("NEW PATH: threaded ($(nthreads()) threads)")

    run_new_path!(ws_nograd, cf.two_body, cf.three_body, cell, nl_nograd, spec_lookup, atom_species_ints;
        offset=ws_nograd.one_body_offset, do_grad=false, threaded=true)
    for i in 1:5
        t = @elapsed run_new_path!(ws_nograd, cf.two_body, cf.three_body, cell, nl_nograd, spec_lookup, atom_species_ints;
            offset=ws_nograd.one_body_offset, do_grad=false, threaded=true)
        μs = round(t / natoms(cell) * 1e6, digits=1)
        println("  run $i: $(round(t, digits=4)) s  ($μs μs/atom)")
    end

    println("\n--- feature + gradients, threaded ---")
    run_new_path!(ws_grad, cf.two_body, cf.three_body, cell, nl_grad, spec_lookup, atom_species_ints;
        offset=ws_grad.one_body_offset, do_grad=true, threaded=true)
    for i in 1:5
        t = @elapsed run_new_path!(ws_grad, cf.two_body, cf.three_body, cell, nl_grad, spec_lookup, atom_species_ints;
            offset=ws_grad.one_body_offset, do_grad=true, threaded=true)
        μs = round(t / natoms(cell) * 1e6, digits=1)
        println("  run $i: $(round(t, digits=4)) s  ($μs μs/atom)")
    end
end

# === 7. Correctness check ====================================================

print_sep("Correctness: old path vs new path (should be ~0 difference)")

ws_old = EDDPotentialsCore.get_workspace(cf, nl_grad, true)
ws_new = EDDPotentialsCore.get_workspace(cf, nl_grad, true)

run_old_path!(ws_old, cf.two_body, cf.three_body, cell, nl_grad;
    offset=ws_old.one_body_offset, do_grad=true)
run_new_path!(ws_new, cf.two_body, cf.three_body, cell, nl_grad, spec_lookup, atom_species_ints;
    offset=ws_new.one_body_offset, do_grad=true, threaded=false)

diff_fvec = maximum(abs, ws_old.fvec .- ws_new.fvec)
diff_gvec = maximum(abs, ws_old.gvec .- ws_new.gvec)
diff_stotv = maximum(abs, ws_old.stotv .- ws_new.stotv)
println("max|Δfvec|  = $diff_fvec")
println("max|Δgvec|  = $diff_gvec")
println("max|Δstotv| = $diff_stotv")
if max(diff_fvec, diff_gvec, diff_stotv) < 1e-10
    println("✓ Old and new paths give identical results (max diff < 1e-10)")
else
    println("✗ MISMATCH detected! max diff = $(max(diff_fvec, diff_gvec, diff_stotv))")
end

# === 8. Allocation profile ===================================================

print_sep("@timev allocation breakdown")

println("--- new path, single-threaded ---")
@timev run_new_path!(ws_nograd, cf.two_body, cf.three_body, cell, nl_nograd, spec_lookup, atom_species_ints;
    offset=ws_nograd.one_body_offset, do_grad=false, threaded=false)

println("\n--- new path, single-threaded + gradients ---")
@timev run_new_path!(ws_grad, cf.two_body, cf.three_body, cell, nl_grad, spec_lookup, atom_species_ints;
    offset=ws_grad.one_body_offset, do_grad=true, threaded=false)

println("\nDone.")
