module Protocols

using CellBase
using EDDPotentials
using EDDPotentials: HyperRoundDiagnostics, update_metadata!
using Printf
using Unitful: ustrip
using LinearAlgebra: norm, diag
using ExtXYZ

function _build_frame_dict(frame, species_vec; round=0, alpha=0.0, phase="")
    pos = frame.positions
    D, N = size(pos)

    arrays = Dict{String,Any}(
        "pos" => pos[1:3, :],
        "species" => String.(string.(species_vec)),
    )
    if D > 3
        arrays["extra_dim"] = pos[4:end, :]
    end
    if frame.forces !== nothing
        F = frame.forces
        fD = size(F, 1)
        arrays["forces"] = F[1:min(3, fD), :]
        if fD > 3
            arrays["forces_extra"] = F[4:end, :]
        end
    end

    info = Dict{String,Any}(
        "energy" => Float64(frame.energy),
        "fmax" => Float64(frame.fmax),
        "smax" => Float64(frame.smax),
        "iteration" => Int(frame.iteration),
        "extra_dim_norm" => Float64(frame.extra_dim_norm),
    )
    if frame.stress !== nothing
        S = frame.stress
        info["stress"] = vec(Float64[Float64(S[i,j]) for i in 1:3, j in 1:3])
    end
    round > 0 && (info["round"] = Int(round))
    alpha > 0 && (info["alpha"] = Float64(alpha))
    !isempty(phase) && (info["phase"] = String(phase))

    cell_mat = if frame.cell_matrix !== nothing
        collect(permutedims(frame.cell_matrix))
    else
        zeros(3, 3)
    end

    Dict{String,Any}(
        "N_atoms" => N,
        "cell" => cell_mat,
        "pbc" => [true, true, true],
        "info" => info,
        "arrays" => arrays,
    )
end

function _attach_round_diagnostics!(dict, diag)
    info = dict["info"]
    info["max_total_force"] = diag.max_total_force
    info["max_3d_force"] = diag.max_3d_force
    info["max_extra_force"] = diag.max_extra_force
    info["avg_extra_dim_norm"] = diag.avg_extra_dim_norm
    info["extra_norm"] = diag.extra_norm
    info["dr_3d"] = diag.dr_3d
    info["iterations_used"] = diag.iterations_used
    info["converged_early"] = diag.converged_early
    info["alpha_after"] = diag.alpha_after
    dict
end

function _write_trajectory_extxyz(all_frames, species_vec, traj_path;
                                   round_diagnostics=HyperRoundDiagnostics[],
                                   round_frame_indices=Int[],
                                   phase_for_round=String[])
    if isempty(all_frames)
        return
    end

    round_boundary_pos = 1

    open(traj_path, "w") do io
        for (k, frame) in enumerate(all_frames)
            round_num = 0
            phase = ""

            if !isempty(round_frame_indices)
                while round_boundary_pos <= length(round_frame_indices) && k > round_frame_indices[round_boundary_pos]
                    round_boundary_pos += 1
                end
                if round_boundary_pos <= length(round_frame_indices)
                    round_start = round_boundary_pos == 1 ? 1 : round_frame_indices[round_boundary_pos - 1] + 1
                    round_end = round_frame_indices[round_boundary_pos]
                    if round_start <= k <= round_end
                        round_num = round_boundary_pos
                        if !isempty(phase_for_round) && round_boundary_pos <= length(phase_for_round)
                            phase = phase_for_round[round_boundary_pos]
                        end
                    end
                end
            end

            d = _build_frame_dict(frame, species_vec; round=round_num, alpha=0.0, phase=phase)

            if !isempty(round_frame_indices) && k in round_frame_indices
                diag_idx = findfirst(==(k), round_frame_indices)
                if diag_idx !== nothing && diag_idx <= length(round_diagnostics)
                    _attach_round_diagnostics!(d, round_diagnostics[diag_idx])
                    d["info"]["alpha"] = round_diagnostics[diag_idx].alpha_before
                end
            end

            ExtXYZ.write_frame(io, d)
        end
    end
end

function _write_final_structure_extxyz(calc, cell_3d, outpath; label="EDDPotentials", energy=nothing)
    try
        update_metadata!(calc, label)
    catch
    end

    info = Dict{String,Any}(
        "energy" => something(energy, Float64(get(cell_3d.metadata, :enthalpy, 0.0))),
        "pressure" => Float64(get(cell_3d.metadata, :pressure, 0.0)),
        "volume" => Float64(ustrip(CellBase.volume(cell_3d))),
        "label" => String(get(cell_3d.metadata, :label, label)),
        "symmetry" => String(get(cell_3d.metadata, :symm, "(n/a)")),
        "n_dimensions" => 3,
    )

    nat = CellBase.nions(cell_3d)
    sp = CellBase.species(cell_3d)
    pos = CellBase.positions(cell_3d)[1:3, :]

    arrays = Dict{String,Any}(
        "pos" => pos,
        "species" => String.(string.(sp)),
    )

    try
        F = eddp_forces(calc)
        fD = size(F, 1)
        arrays["forces"] = F[1:min(3, fD), :]
    catch
    end

    try
        S = eddp_stress(calc)
        info["stress"] = vec(Float64[Float64(S[i,j]) for i in 1:3, j in 1:3])
    catch
    end

    lattice = CellBase.cellmat(cell_3d)
    cell_mat = collect(permutedims(lattice))

    dict = Dict{String,Any}(
        "N_atoms" => nat,
        "cell" => cell_mat,
        "pbc" => [true, true, true],
        "info" => info,
        "arrays" => arrays,
    )

    ExtXYZ.write_frames(outpath, [dict])
end

function _min_dist_3d(cell)
    dm = CellBase.distance_matrix(cell)
    n = size(dm, 1)
    min_d = Inf
    for i in 1:n, j in i+1:n
        min_d = min(min_d, dm[i, j])
    end
    ustrip(min_d)
end

function run_standard_vc(calc::EDDPCalculator, cell::Cell, outdir::String)
    mkpath(outdir)
    species_vec = copy(CellBase.species(cell))
    vol_init = ustrip(CellBase.volume(cell))

    vc = create_calculator(calc, cell; modifiers=[
        CellRelaxationModifier(cell),
    ])

    wall = @elapsed begin
        opt = opt_fire(vc; itermax=1000, keep_trajectory=true)
    end

    final_cell = get_cell(vc)
    final_cell_3d = Cell(
        Lattice(CellBase.cellmat(final_cell)),
        CellBase.species(final_cell),
        CellBase.positions(final_cell)[1:3, :],
    )

    traj_path = joinpath(outdir, "trajectory.extxyz")
    _write_trajectory_extxyz(opt.trajectory, species_vec, traj_path)

    _write_final_structure_extxyz(vc, final_cell_3d,
        joinpath(outdir, "relaxed.extxyz"); label="standard-vc",
        energy=Float64(EDDPotentials.eddp_energy(vc)))

    try
        CellBase.write_res(joinpath(outdir, "relaxed.res"), final_cell_3d)
    catch e
        @warn "Failed to write SHELX .res file" exception=e
    end

    total_iter = opt.iteration
    converged_flag = EDDPotentials.converged(opt, vc)
    final_energy = EDDPotentials.eddp_energy(vc)
    fmax_final = isempty(opt.trajectory) ? NaN : opt.trajectory[end].fmax
    vol_final = ustrip(CellBase.volume(final_cell_3d))
    min_dist = _min_dist_3d(final_cell_3d)
    n_frames = length(opt.trajectory)

    empty!(opt.trajectory)
    GC.gc()

    return Dict{String,Any}(
        "protocol" => "standard-vc",
        "converged" => converged_flag,
        "iterations" => total_iter,
        "n_traj_frames" => n_frames,
        "final_energy" => final_energy,
        "vol_ratio" => vol_final / vol_init,
        "fmax" => fmax_final,
        "min_dist" => min_dist,
        "wall_s" => wall,
    )
end

function run_hyper_1d(calc::EDDPCalculator, cell::Cell, outdir::String)
    mkpath(outdir)
    species_vec = copy(CellBase.species(cell))
    vol_init = ustrip(CellBase.volume(cell))

    target_iterations = 1000
    schedule_every = 10
    alpha_start = 1e-4

    hcell = CellBase.add_dimensions(deepcopy(cell), 1)

    source = create_calculator(calc, hcell)
    penalty = ExtraDimensionPenaltyModifier(alpha_start; start_dim=4)
    cell_relax = CellRelaxationModifier(hcell)

    wall = @elapsed begin
        metacalc_pos = MetaCalculator(source, [penalty])
        set_modifier_parameter!(metacalc_pos, ExtraDimensionPenaltyModifier, :alpha, 0.5)
        p1 = scheduled_penalty_relax!(metacalc_pos;
            schedule_every, max_iterations=div(target_iterations, 2),
            alpha_start=0.5, ramp_alpha=false, relax_fn=opt_fire,
            relax_kwargs=(; keep_trajectory=true))

        metacalc_full = MetaCalculator(source, [penalty, cell_relax])
        p2 = scheduled_penalty_relax!(metacalc_full;
            schedule_every, max_iterations=target_iterations - div(target_iterations, 2),
            alpha_start, relax_fn=opt_fire,
            relax_kwargs=(; keep_trajectory=true))

        hyper_cell = get_cell(metacalc_full)
        cell_3d = CellBase.remove_dimensions(hyper_cell)
        calc3d_source = create_calculator(calc, cell_3d)
        calc3d_vc = MetaCalculator(calc3d_source, [CellRelaxationModifier(cell_3d)])
        p3_opt = opt_fire(calc3d_vc; itermax=1000, keep_trajectory=true)
    end

    final_cell_3d_raw = get_cell(calc3d_vc)
    final_cell_3d = Cell(
        Lattice(CellBase.cellmat(final_cell_3d_raw)),
        CellBase.species(final_cell_3d_raw),
        CellBase.positions(final_cell_3d_raw)[1:3, :],
    )
    _write_final_structure_extxyz(calc3d_vc, final_cell_3d,
        joinpath(outdir, "relaxed.extxyz"); label="hyper-1d",
        energy=Float64(EDDPotentials.eddp_energy(calc3d_vc)))

    try
        CellBase.write_res(joinpath(outdir, "relaxed.res"), final_cell_3d)
    catch e
        @warn "Failed to write SHELX .res file" exception=e
    end

    all_frames = []
    round_frame_indices = Int[]
    all_diagnostics = HyperRoundDiagnostics[]
    phase_for_round = String[]

    p1_nframes = 0
    for opt_res in p1.results
        append!(all_frames, opt_res.trajectory)
        p1_nframes += length(opt_res.trajectory)
    end
    for (ri, diag) in enumerate(p1.round_diagnostics)
        idx = min(sum(r.iteration for r in p1.results[1:ri]), length(all_frames))
        if idx > 0
            push!(round_frame_indices, idx)
            push!(all_diagnostics, diag)
            push!(phase_for_round, "fixed-cell-hyper")
        end
    end

    for opt_res in p2.results
        append!(all_frames, opt_res.trajectory)
    end
    for (ri, diag) in enumerate(p2.round_diagnostics)
        idx = p1_nframes + min(sum(r.iteration for r in p2.results[1:ri]), length(p2.results[ri].trajectory))
        total_idx = p1_nframes + sum(min(r.iteration, length(r.trajectory)) for r in p2.results[1:ri])
        if total_idx > 0
            push!(round_frame_indices, total_idx)
            push!(all_diagnostics, diag)
            push!(phase_for_round, "vc-hyper")
        end
    end

    append!(all_frames, p3_opt.trajectory)

    if !isempty(p3_opt.trajectory)
        push!(round_frame_indices, length(all_frames))
        push!(all_diagnostics, HyperRoundDiagnostics(
            0, 0.0, 0.0, p3_opt.iteration, false,
            p3_opt.trajectory[end].fmax, p3_opt.trajectory[end].fmax, 0.0,
            0.0, 0.0, 0.0, p3_opt.trajectory[end].energy,
        ))
        push!(phase_for_round, "vc-3d")
    end

    traj_path = joinpath(outdir, "trajectory.extxyz")
    _write_trajectory_extxyz(all_frames, species_vec, traj_path;
        round_diagnostics=all_diagnostics,
        round_frame_indices=round_frame_indices,
        phase_for_round=phase_for_round)

    final_energy = EDDPotentials.eddp_energy(calc3d_vc)
    total_iter = sum(r.iteration for r in p1.results; init=0) +
                 sum(r.iteration for r in p2.results; init=0) +
                 p3_opt.iteration

    pos_hyper = CellBase.positions(hyper_cell)
    extra_norm = size(pos_hyper, 1) > 3 ? maximum(norm.(eachcol(pos_hyper[4:end, :]))) : 0.0

    vol_final = ustrip(CellBase.volume(final_cell_3d))
    min_dist = _min_dist_3d(final_cell_3d)
    converged_flag = EDDPotentials.converged(p3_opt, calc3d_vc) && extra_norm < 0.5
    fmax_final = isempty(p3_opt.trajectory) ? NaN : p3_opt.trajectory[end].fmax
    n_traj_frames = length(all_frames)
    extra_dim_norms = vcat(p1.extra_dim_norms, p2.extra_dim_norms)

    empty!(all_frames)
    empty!(p1.results)
    empty!(p2.results)
    empty!(p3_opt.trajectory)
    GC.gc()

    return Dict{String,Any}(
        "protocol" => "hyper-1d",
        "converged" => converged_flag,
        "iterations" => total_iter,
        "n_traj_frames" => n_traj_frames,
        "final_energy" => final_energy,
        "vol_ratio" => vol_final / vol_init,
        "fmax" => fmax_final,
        "min_dist" => min_dist,
        "wall_s" => wall,
        "extra_norm" => extra_norm,
        "extra_dim_norms" => extra_dim_norms,
    )
end

end
