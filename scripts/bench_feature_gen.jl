#!/usr/bin/env julia
#=
Profile EDDP feature generation on an MgO rocksalt supercell (~96 atoms).

Usage:
    julia --project=. scripts/bench_feature_gen.jl

Uses the species_lookup O(1) feature matching path + optional threaded atom loop.

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

# === 5. Feature computation benchmarks ========================================

print_sep("Feature computation (species lookup path)")

function run_path!(ws, f2, f3, cell, nl, lookup, atom_species_ints; offset, do_grad, threaded)
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

# --- 5a. Single-threaded ---
println("--- feature vector only, single-threaded ---")
run_path!(ws_nograd, cf.two_body, cf.three_body, cell, nl_nograd, spec_lookup, atom_species_ints;
    offset=ws_nograd.one_body_offset, do_grad=false, threaded=false)
for i in 1:5
    t = @elapsed run_path!(ws_nograd, cf.two_body, cf.three_body, cell, nl_nograd, spec_lookup, atom_species_ints;
        offset=ws_nograd.one_body_offset, do_grad=false, threaded=false)
    μs = round(t / natoms(cell) * 1e6, digits=1)
    println("  run $i: $(round(t, digits=4)) s  ($μs μs/atom)")
end

println("\n--- feature + gradients, single-threaded ---")
run_path!(ws_grad, cf.two_body, cf.three_body, cell, nl_grad, spec_lookup, atom_species_ints;
    offset=ws_grad.one_body_offset, do_grad=true, threaded=false)
for i in 1:5
    t = @elapsed run_path!(ws_grad, cf.two_body, cf.three_body, cell, nl_grad, spec_lookup, atom_species_ints;
        offset=ws_grad.one_body_offset, do_grad=true, threaded=false)
    μs = round(t / natoms(cell) * 1e6, digits=1)
    println("  run $i: $(round(t, digits=4)) s  ($μs μs/atom)")
end

# --- 5b. Threaded ---
if nthreads() > 1
    print_sep("Threaded ($(nthreads()) threads)")

    run_path!(ws_nograd, cf.two_body, cf.three_body, cell, nl_nograd, spec_lookup, atom_species_ints;
        offset=ws_nograd.one_body_offset, do_grad=false, threaded=true)
    for i in 1:5
        t = @elapsed run_path!(ws_nograd, cf.two_body, cf.three_body, cell, nl_nograd, spec_lookup, atom_species_ints;
            offset=ws_nograd.one_body_offset, do_grad=false, threaded=true)
        μs = round(t / natoms(cell) * 1e6, digits=1)
        println("  run $i: $(round(t, digits=4)) s  ($μs μs/atom)")
    end

    println("\n--- feature + gradients, threaded ---")
    run_path!(ws_grad, cf.two_body, cf.three_body, cell, nl_grad, spec_lookup, atom_species_ints;
        offset=ws_grad.one_body_offset, do_grad=true, threaded=true)
    for i in 1:5
        t = @elapsed run_path!(ws_grad, cf.two_body, cf.three_body, cell, nl_grad, spec_lookup, atom_species_ints;
            offset=ws_grad.one_body_offset, do_grad=true, threaded=true)
        μs = round(t / natoms(cell) * 1e6, digits=1)
        println("  run $i: $(round(t, digits=4)) s  ($μs μs/atom)")
    end
end

# === 6. Allocation profile ===================================================

print_sep("@timev allocation breakdown")

println("--- single-threaded ---")
@timev run_path!(ws_nograd, cf.two_body, cf.three_body, cell, nl_nograd, spec_lookup, atom_species_ints;
    offset=ws_nograd.one_body_offset, do_grad=false, threaded=false)

println("\n--- single-threaded + gradients ---")
@timev run_path!(ws_grad, cf.two_body, cf.three_body, cell, nl_grad, spec_lookup, atom_species_ints;
    offset=ws_grad.one_body_offset, do_grad=true, threaded=false)

println("\nDone.")
