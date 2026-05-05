using .Protocols
using .Protocols: _fmt, _print_table, _build_frame_dict, _write_trajectory_extxyz,
                   _write_final_structure_extxyz, _attach_round_diagnostics!
using EDDPotentials
using EDDPotentials: HyperRoundDiagnostics, ExtraDimensionPenaltyModifier,
                     CellRelaxationModifier, MetaCalculator, update_metadata!,
                     scheduled_penalty_relax!, opt_fire, opt_tpsd, eddp_energy, eddp_forces,
                     eddp_stress, converged, lj_baseline_calc, get_cell, set_modifier_parameter!
using CellBase
using Unitful: ustrip
using LinearAlgebra: norm, diag
using ExtXYZ

const LJ_SIGMA = 2.0
const LJ_EPSILON = 1.0
const LJ_RCUT = 6.0
const LJ_NMAX = 5000
const LJ_EXTRA_DIM_TOL = 0.1

function _make_lj_calc(cell; sigma=LJ_SIGMA, epsilon=LJ_EPSILON, rcut=LJ_RCUT, nmax=LJ_NMAX)
    lj_baseline_calc(cell; sigma, epsilon, rcut, nmax)
end

function lj_run_standard_vc(cell, outdir;
                             sigma=LJ_SIGMA, epsilon=LJ_EPSILON,
                             rcut=LJ_RCUT, nmax=LJ_NMAX, itermax=500,
                             relax_fn=opt_fire, f_tol=1e-3)
    mkpath(outdir)
    species_vec = copy(CellBase.species(cell))
    vol_init = ustrip(CellBase.volume(cell))

    source = _make_lj_calc(cell; sigma, epsilon, rcut, nmax)
    vc = MetaCalculator(source, [CellRelaxationModifier(cell)])

    wall = @elapsed begin
        opt = relax_fn(vc; itermax=itermax, f_tol=f_tol, keep_trajectory=true)
    end

    final_cell = get_cell(vc)
    final_cell_3d = Cell(
        CellBase.Lattice(CellBase.cellmat(final_cell)),
        CellBase.species(final_cell),
        CellBase.positions(final_cell)[1:3, :],
    )

    traj_path = joinpath(outdir, "trajectory.extxyz")
    _write_trajectory_extxyz(opt.trajectory, species_vec, traj_path)

    _write_final_structure_extxyz(vc, final_cell_3d,
        joinpath(outdir, "relaxed.extxyz"); label="standard-vc",
        energy=Float64(eddp_energy(vc)))

    try
        CellBase.write_res(joinpath(outdir, "relaxed.res"), final_cell_3d)
    catch e
        @warn "Failed to write SHELX .res file" exception=e
    end

    total_iter = opt.iteration
    converged_flag = converged(opt, vc)
    final_energy = eddp_energy(vc)
    fmax_final = isempty(opt.trajectory) ? NaN : opt.trajectory[end].fmax
    vol_final = ustrip(CellBase.volume(final_cell_3d))
    min_dist = Protocols._min_dist_3d(final_cell_3d)
    n_frames = length(opt.trajectory)

    result = Dict{String,Any}(
        "protocol" => "standard-vc",
        "optimizer" => relax_fn === opt_fire ? "fire" : "tpsd",
        "converged" => converged_flag,
        "iterations" => total_iter,
        "n_traj_frames" => n_frames,
        "final_energy" => final_energy,
        "vol_ratio" => vol_final / vol_init,
        "fmax" => fmax_final,
        "min_dist" => min_dist,
        "wall_s" => wall,
    )

    empty!(opt.trajectory)
    GC.gc()

    return result
end

function lj_run_hyper_1d(cell, outdir;
                          sigma=LJ_SIGMA, epsilon=LJ_EPSILON,
                          rcut=LJ_RCUT, nmax=LJ_NMAX, itermax=1000,
                          relax_fn=opt_fire, f_tol=1e-3,
                          dynamic_alpha=false, force_ratio=2.0, max_alpha=1e4)
    mkpath(outdir)
    species_vec = copy(CellBase.species(cell))
    vol_init = ustrip(CellBase.volume(cell))

    schedule_every = 10
    alpha_start = 1e-4

    hcell = CellBase.add_dimensions(deepcopy(cell), 1, 10.0)

    source = _make_lj_calc(hcell; sigma, epsilon, rcut, nmax)
    penalty = ExtraDimensionPenaltyModifier(alpha_start; start_dim=4)
    cell_relax = CellRelaxationModifier(hcell)

    wall = @elapsed begin
        metacalc_pos = MetaCalculator(source, [penalty])
        set_modifier_parameter!(metacalc_pos, ExtraDimensionPenaltyModifier, :alpha, 1e-6)
        p1 = scheduled_penalty_relax!(metacalc_pos;
            schedule_every, max_iterations=div(itermax, 2),
            alpha_start=1e-6, ramp_alpha=true, max_alpha,
            dynamic_alpha, force_ratio,
            relax_fn=relax_fn,
            relax_kwargs=(; f_tol=f_tol, keep_trajectory=true))

        metacalc_full = MetaCalculator(source, [penalty, cell_relax])
        p2 = scheduled_penalty_relax!(metacalc_full;
            schedule_every, max_iterations=itermax - div(itermax, 2),
            alpha_start, max_alpha,
            dynamic_alpha, force_ratio,
            relax_fn=relax_fn,
            relax_kwargs=(; f_tol=f_tol, keep_trajectory=true))

        hyper_cell = get_cell(metacalc_full)
        cell_3d = CellBase.remove_dimensions(hyper_cell)
        calc3d_source = _make_lj_calc(cell_3d; sigma, epsilon, rcut, nmax)
        calc3d_vc = MetaCalculator(calc3d_source, [CellRelaxationModifier(cell_3d)])
        p3_opt = relax_fn(calc3d_vc; itermax=500, f_tol=f_tol, keep_trajectory=true)
    end

    final_cell_3d_raw = get_cell(calc3d_vc)
    final_cell_3d = Cell(
        CellBase.Lattice(CellBase.cellmat(final_cell_3d_raw)),
        CellBase.species(final_cell_3d_raw),
        CellBase.positions(final_cell_3d_raw)[1:3, :],
    )
    _write_final_structure_extxyz(calc3d_vc, final_cell_3d,
        joinpath(outdir, "relaxed.extxyz"); label="hyper-1d",
        energy=Float64(eddp_energy(calc3d_vc)))

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

    final_energy = eddp_energy(calc3d_vc)
    total_iter = sum(r.iteration for r in p1.results; init=0) +
                 sum(r.iteration for r in p2.results; init=0) +
                 p3_opt.iteration

    pos_hyper = CellBase.positions(hyper_cell)
    extra_norm = size(pos_hyper, 1) > 3 ? maximum(norm.(eachcol(pos_hyper[4:end, :]))) : 0.0

    vol_final = ustrip(CellBase.volume(final_cell_3d))
    min_dist = Protocols._min_dist_3d(final_cell_3d)
    converged_flag = converged(p3_opt, calc3d_vc) && extra_norm < LJ_EXTRA_DIM_TOL
    fmax_final = isempty(p3_opt.trajectory) ? NaN : p3_opt.trajectory[end].fmax
    n_traj_frames = length(all_frames)
    extra_dim_norms = vcat(p1.extra_dim_norms, p2.extra_dim_norms)

    result = Dict{String,Any}(
        "protocol" => "hyper-1d",
        "optimizer" => relax_fn === opt_fire ? "fire" : "tpsd",
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

    empty!(all_frames)
    empty!(p1.results)
    empty!(p2.results)
    empty!(p3_opt.trajectory)
    GC.gc()

    return result
end
