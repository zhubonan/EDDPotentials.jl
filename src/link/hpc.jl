#=
File-backed HPC master/client workflow for link.

The master owns phase transitions and shared workdir preparation. Clients are
small state-machine supervisors that read `control.json`, launch one child
process, and report liveness through atomic heartbeat files.
=#

using Dates

const HPC_CLIENT_CMD_PLACEHOLDER = "{{EDDP_HPC_CLIENT_CMD}}"

_hpc_root(bu::Builder) = joinpath(bu.state.workdir, bu.options.hpc.root)
_hpc_control_path(bu::Builder) = joinpath(_hpc_root(bu), "control.json")
_hpc_workers_dir(bu::Builder) = joinpath(_hpc_root(bu), "workers")
_hpc_phase_dir(bu::Builder, iter::Int, phase::AbstractString) =
    joinpath(_hpc_root(bu), "phases", "gen$(iter)-$(phase)")
_hpc_phase_manifest_path(phase_dir) = joinpath(phase_dir, "manifest.json")
_hpc_events_dir(phase_dir) = joinpath(phase_dir, "events")
_hpc_worker_path(bu::Builder, worker_id) =
    joinpath(_hpc_workers_dir(bu), "$(worker_id).json")

_hpc_timestamp() = Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sssZ")

function _hpc_atomic_write(path::AbstractString, text::AbstractString)
    mkpath(dirname(path))
    tmp = path * ".tmp-$(getpid())-$(rand(UInt))"
    open(tmp, "w") do io
        write(io, text)
    end
    mv(tmp, path; force=true)
    return path
end

function _hpc_write_json(path::AbstractString, data)
    io = IOBuffer()
    JSON.print(io, data, 4)
    _hpc_atomic_write(path, String(take!(io)))
end

function _hpc_read_json(path::AbstractString)
    isfile(path) || return nothing
    try
        return JSON.parsefile(path)
    catch err
        @warn "Failed to parse HPC JSON file `$path`: $err"
        return nothing
    end
end

function _hpc_marker(path::AbstractString)
    mkpath(dirname(path))
    touch(path)
    return path
end

function _hpc_write_event(phase_dir, worker_id, kind, data=Dict{String,Any}())
    event_dir = _hpc_events_dir(phase_dir)
    name = "$(replace(_hpc_timestamp(), ':' => ""))-$(worker_id)-$(kind)-$(rand(UInt)).json"
    payload = merge(
        Dict{String,Any}(
            "worker_id" => string(worker_id),
            "kind" => string(kind),
            "created_at" => _hpc_timestamp(),
        ),
        Dict{String,Any}(data),
    )
    _hpc_write_json(joinpath(event_dir, name), payload)
end

function hpc_write_heartbeat(
    bu::Builder,
    worker_id::AbstractString;
    state::AbstractString,
    phase::AbstractString="",
    iteration::Int=bu.state.iteration,
    epoch::Int=0,
    child_pid=nothing,
    claim=nothing,
    model_path=nothing,
    message::AbstractString="",
    run_id::AbstractString="",
)
    payload = Dict{String,Any}(
        "worker_id" => worker_id,
        "run_id" => run_id,
        "builder_uuid" => builder_uuid(bu),
        "host" => get(ENV, "HOSTNAME", get(ENV, "COMPUTERNAME", "")),
        "pid" => getpid(),
        "state" => state,
        "phase" => phase,
        "iteration" => iteration,
        "epoch" => epoch,
        "child_pid" => child_pid === nothing ? "" : string(child_pid),
        "claim" => claim === nothing ? "" : string(claim),
        "model_path" => model_path === nothing ? "" : string(model_path),
        "started_at" => _hpc_timestamp(),
        "updated_at" => _hpc_timestamp(),
        "message" => message,
    )
    _hpc_write_json(_hpc_worker_path(bu, worker_id), payload)
end

function hpc_worker_states(bu::Builder)
    dir = _hpc_workers_dir(bu)
    isdir(dir) || return Dict{String,Any}[]
    states = Dict{String,Any}[]
    for path in sort(glob_allow_abs(joinpath(dir, "*.json")))
        state = _hpc_read_json(path)
        state === nothing || push!(states, state)
    end
    return states
end

function _hpc_publish_control!(
    bu::Builder;
    iteration::Int=bu.state.iteration,
    phase::AbstractString,
    phase_dir::AbstractString,
    status::AbstractString="running",
    epoch::Int=round(Int, time()),
    extra=Dict{String,Any}(),
)
    payload = merge(
        Dict{String,Any}(
            "builder_uuid" => builder_uuid(bu),
            "run_id" => builder_short_uuid(bu) * "-$(iteration)",
            "epoch" => epoch,
            "iteration" => iteration,
            "phase" => phase,
            "phase_dir" => phase_dir,
            "status" => status,
            "updated_at" => _hpc_timestamp(),
        ),
        Dict{String,Any}(extra),
    )
    _hpc_write_json(_hpc_control_path(bu), payload)
    return payload
end

function _hpc_read_control(bu::Builder)
    control = _hpc_read_json(_hpc_control_path(bu))
    control === nothing && return nothing
    get(control, "builder_uuid", "") == builder_uuid(bu) || return nothing
    return control
end

function hpc_prepare_crud_workdir!(bu::Builder, iter::Int=bu.state.iteration)
    state = bu.state
    phase_dir = _hpc_phase_dir(bu, iter, "dft")
    crud_workdir = joinpath(state.workdir, bu.options.hpc.crud_workdir, "gen$(iter)")
    input_dir = input_structure_dir(state.workdir, iter)
    output_dir = output_structure_dir(state.workdir, iter)

    if isfile(joinpath(phase_dir, "READY")) || isfile(joinpath(phase_dir, "DONE"))
        return crud_workdir
    end

    mkpath(phase_dir)
    for dir in ("hopper", "good_castep", "bad_castep")
        path = joinpath(crud_workdir, dir)
        isdir(path) && rm(path; recursive=true)
        mkpath(path)
    end
    mkpath(output_dir)

    seed = splitext(joinpath(state.workdir, state.seedfile_calc))[1]
    cp(seed * ".cell", joinpath(crud_workdir, basename(seed) * ".cell"); force=true)
    cp(seed * ".param", joinpath(crud_workdir, basename(seed) * ".param"); force=true)

    input_files = sort(glob_allow_abs(joinpath(input_dir, "*.res")))
    for file in input_files
        cp(file, joinpath(crud_workdir, "hopper", basename(file)); force=true)
    end

    manifest = Dict{String,Any}(
        "phase" => "dft",
        "iteration" => iter,
        "input_dir" => input_dir,
        "output_dir" => output_dir,
        "crud_workdir" => crud_workdir,
        "mpinp" => state.mpinp,
        "threshold" => state.per_generation_threshold,
        "input_count" => length(input_files),
        "created_at" => _hpc_timestamp(),
    )
    _hpc_write_json(_hpc_phase_manifest_path(phase_dir), manifest)
    _hpc_marker(joinpath(phase_dir, "READY"))
    return crud_workdir
end

function _hpc_dft_counts(crud_workdir)
    good = length(glob_allow_abs(joinpath(crud_workdir, "good_castep", "*.res")))
    bad = length(glob_allow_abs(joinpath(crud_workdir, "bad_castep", "*.res")))
    return good, bad
end

function hpc_collect_crud_results!(bu::Builder, iter::Int=bu.state.iteration)
    phase_dir = _hpc_phase_dir(bu, iter, "dft")
    manifest = _hpc_read_json(_hpc_phase_manifest_path(phase_dir))
    manifest === nothing && throw(ArgumentError("Missing DFT manifest for generation $iter."))
    crud_workdir = manifest["crud_workdir"]
    output_dir = manifest["output_dir"]
    mkpath(output_dir)
    for file in glob_allow_abs(joinpath(crud_workdir, "good_castep", "*.res"))
        mv(file, joinpath(output_dir, basename(file)); force=true)
    end
    _hpc_marker(joinpath(phase_dir, "DONE"))
    return output_dir
end

function _hpc_required_dft_terminal(manifest)
    n = Int(manifest["input_count"])
    n == 0 && return 0
    return max(1, ceil(Int, n * Float64(manifest["threshold"])))
end

function _hpc_required_dft_success(manifest)
    _hpc_required_dft_terminal(manifest)
end

function _hpc_request_phase_stop!(bu::Builder, phase_dir::AbstractString)
    _hpc_marker(joinpath(phase_dir, "STOP"))
end

function _hpc_abort_phase!(bu::Builder, phase_dir::AbstractString, message)
    _hpc_marker(joinpath(phase_dir, "STOP"))
    _hpc_write_json(
        joinpath(phase_dir, "ABORTED.json"),
        Dict{String,Any}("message" => string(message), "aborted_at" => _hpc_timestamp()),
    )
end

function _hpc_phase_worker_is_active(state, control)
    get(state, "phase", "") == get(control, "phase", "") || return false
    Int(get(state, "iteration", -1)) == Int(get(control, "iteration", -2)) || return false
    Int(get(state, "epoch", -1)) == Int(get(control, "epoch", -2)) || return false
    return get(state, "state", "") in (
        "DFT_RUNNING",
        "TRAINING_CLAIM",
        "TRAINING_RUNNING",
        "STOPPING_CHILD",
    )
end

function _hpc_wait_for_phase_workers_stopped!(bu::Builder, control)
    timeout =
        bu.options.hpc.child_grace_seconds +
        bu.options.hpc.child_kill_seconds +
        bu.options.hpc.heartbeat_interval +
        2 * bu.options.hpc.poll_interval
    started = time()
    while true
        active = filter(state -> _hpc_phase_worker_is_active(state, control), hpc_worker_states(bu))
        isempty(active) && return true
        if time() - started >= timeout
            @warn "Timed out waiting for HPC clients to stop" active_workers = length(active)
            return false
        end
        sleep(max(1.0, bu.options.hpc.poll_interval))
    end
end

function _hpc_wait_for_dft!(bu::Builder, phase_dir)
    manifest = _hpc_read_json(_hpc_phase_manifest_path(phase_dir))
    manifest === nothing && throw(ArgumentError("Missing DFT manifest `$phase_dir`."))
    crud_workdir = manifest["crud_workdir"]
    required_terminal = _hpc_required_dft_terminal(manifest)
    while true
        good, bad = _hpc_dft_counts(crud_workdir)
        if good + bad >= required_terminal
            _hpc_request_phase_stop!(bu, phase_dir)
            if good < _hpc_required_dft_success(manifest)
                _hpc_marker(joinpath(phase_dir, "FAILED"))
                throw(
                    ErrorException(
                        "DFT terminal threshold reached for generation $(manifest["iteration"]), but only $good successful calculations are available.",
                    ),
                )
            end
            return good, bad
        end
        if should_stop(bu)
            _hpc_abort_phase!(bu, phase_dir, "STOP requested while waiting for DFT.")
            throw(ErrorException("STOP requested while waiting for DFT phase."))
        end
        sleep(bu.options.hpc.poll_interval)
    end
end

function hpc_run_dft_phase!(bu::Builder, iter::Int=bu.state.iteration)
    phase_dir = _hpc_phase_dir(bu, iter, "dft")
    crud_workdir = hpc_prepare_crud_workdir!(bu, iter)
    manifest = _hpc_read_json(_hpc_phase_manifest_path(phase_dir))
    control = _hpc_publish_control!(
        bu;
        iteration=iter,
        phase="dft",
        phase_dir,
        extra=Dict("dft" => manifest),
    )
    _hpc_submit_clients(bu, "dft", iter)
    try
        _hpc_wait_for_dft!(bu, phase_dir)
    catch
        _hpc_publish_control!(bu; iteration=iter, phase="dft", phase_dir, status="stopping")
        rethrow()
    end
    _hpc_wait_for_phase_workers_stopped!(bu, control)
    hpc_collect_crud_results!(bu, iter)
    _hpc_publish_control!(bu; iteration=iter, phase="idle", phase_dir, status="done")
    return crud_workdir
end

function _hpc_training_phase_dir(bu::Builder, iter=bu.state.iteration)
    _hpc_phase_dir(bu, iter, "train")
end

function _hpc_training_slots_dir(bu::Builder, iter=bu.state.iteration)
    joinpath(_hpc_training_phase_dir(bu, iter), "slots")
end

function _hpc_slot_name(slot_id::Int)
    lpad(string(slot_id), 6, '0')
end

function hpc_prepare_training_phase!(bu::Builder, iter::Int=bu.state.iteration)
    phase_dir = _hpc_training_phase_dir(bu, iter)
    mkpath(joinpath(phase_dir, "slots"))
    mkpath(_hpc_events_dir(phase_dir))
    dataset_path = dataset_name(bu, iter)
    manifest = Dict{String,Any}(
        "phase" => "training",
        "iteration" => iter,
        "dataset_path" => dataset_path,
        "training_dir" => joinpath(bu.state.workdir, TRAINING_DIR),
        "nmodels" => bu.trainer.nmodels,
        "required_models" => _external_required_models(bu.trainer),
        "prefix" => bu.trainer.prefix,
        "created_at" => _hpc_timestamp(),
    )
    _hpc_write_json(_hpc_phase_manifest_path(phase_dir), manifest)
    _hpc_marker(joinpath(phase_dir, "READY"))
    return phase_dir
end

function hpc_training_model_path(bu::Builder, iter::Int, slot_id::Int)
    slot = _hpc_slot_name(slot_id)
    joinpath(
        bu.state.workdir,
        TRAINING_DIR,
        "$(bu.trainer.prefix)gen-$(iter)-model-slot-$(slot).json",
    )
end

function hpc_claim_training_slot!(bu::Builder, iter::Int, worker_id::AbstractString)
    slots_dir = _hpc_training_slots_dir(bu, iter)
    mkpath(slots_dir)
    now = time()
    for slot_id = 1:bu.trainer.nmodels
        slot = _hpc_slot_name(slot_id)
        done_path = joinpath(slots_dir, "$(slot).done.json")
        failed_path = joinpath(slots_dir, "$(slot).failed.json")
        lock_path = joinpath(slots_dir, "$(slot).lock")
        isfile(done_path) && continue
        if isdir(lock_path)
            age = now - stat(lock_path).mtime
            if age > bu.options.hpc.stale_claim_seconds
                rm(lock_path; recursive=true, force=true)
            else
                continue
            end
        end
        try
            isfile(failed_path) && rm(failed_path; force=true)
            mkdir(lock_path)
            model_path = hpc_training_model_path(bu, iter, slot_id)
            claim = Dict{String,Any}(
                "slot_id" => slot_id,
                "worker_id" => worker_id,
                "host" => get(ENV, "HOSTNAME", ""),
                "pid" => getpid(),
                "model_path" => model_path,
                "claimed_at" => _hpc_timestamp(),
            )
            _hpc_write_json(joinpath(lock_path, "claim.json"), claim)
            return slot_id, lock_path, model_path
        catch err
            err isa IOError || rethrow()
        end
    end
    return nothing
end

function hpc_mark_training_slot_done!(bu::Builder, iter::Int, slot_id::Int, model_path)
    slots_dir = _hpc_training_slots_dir(bu, iter)
    slot = _hpc_slot_name(slot_id)
    payload = Dict{String,Any}(
        "slot_id" => slot_id,
        "model_path" => model_path,
        "done_at" => _hpc_timestamp(),
    )
    _hpc_write_json(joinpath(slots_dir, "$(slot).done.json"), payload)
end

function hpc_mark_training_slot_failed!(bu::Builder, iter::Int, slot_id::Int, message)
    slots_dir = _hpc_training_slots_dir(bu, iter)
    slot = _hpc_slot_name(slot_id)
    lock_path = joinpath(slots_dir, "$(slot).lock")
    payload = Dict{String,Any}(
        "slot_id" => slot_id,
        "message" => message,
        "failed_at" => _hpc_timestamp(),
    )
    _hpc_write_json(joinpath(slots_dir, "$(slot).failed.json"), payload)
    isdir(lock_path) && rm(lock_path; recursive=true, force=true)
end

function _hpc_wait_for_training!(bu::Builder, phase_dir)
    manifest = _hpc_read_json(_hpc_phase_manifest_path(phase_dir))
    manifest === nothing && throw(ArgumentError("Missing training manifest `$phase_dir`."))
    required = Int(manifest["required_models"])
    while true
        nm = num_existing_models(bu)
        if nm >= required
            _hpc_request_phase_stop!(bu, phase_dir)
            return nm
        end
        if should_stop(bu)
            _hpc_abort_phase!(bu, phase_dir, "STOP requested while waiting for training.")
            throw(ErrorException("STOP requested while waiting for training phase."))
        end
        sleep(bu.options.hpc.poll_interval)
    end
end

function hpc_run_training_phase!(
    bu::Builder,
    iter::Int=bu.state.iteration;
    create_ensemble_model::Bool=true,
)
    @info "Preparing dataset..."
    write_dataset(bu)
    phase_dir = hpc_prepare_training_phase!(bu, iter)
    manifest = _hpc_read_json(_hpc_phase_manifest_path(phase_dir))
    control = _hpc_publish_control!(
        bu;
        iteration=iter,
        phase="training",
        phase_dir,
        extra=Dict("training" => manifest),
    )
    _hpc_submit_clients(bu, "training", iter)
    try
        _hpc_wait_for_training!(bu, phase_dir)
    catch
        _hpc_publish_control!(bu; iteration=iter, phase="training", phase_dir, status="stopping")
        rethrow()
    end
    _hpc_wait_for_phase_workers_stopped!(bu, control)
    _hpc_marker(joinpath(phase_dir, "DONE"))
    _hpc_publish_control!(bu; iteration=iter, phase="idle", phase_dir, status="done")
    nm = num_existing_models(bu)
    if nm >= _external_required_models(bu.trainer) && create_ensemble_model
        return create_ensemble(
            bu;
            clean_individual_models=bu.trainer.clean_models,
            save_ensemble_model=true,
        )
    elseif nm >= _external_required_models(bu.trainer)
        return nothing
    end
    throw(ErrorException("Only $nm models are found in the training directory."))
end

function _hpc_client_julia_code()
    "using EDDPotentials; EDDPotentials.hpc_client()"
end

function _hpc_client_base_cmd(bu::Builder)
    project_path = dirname(Base.active_project())
    builder_file = bu.state.builder_file_path
    @assert builder_file != ""
    Cmd([
        Base.julia_cmd()...,
        "--project=$(project_path)",
        "-e",
        _hpc_client_julia_code(),
        "--",
        "--file",
        builder_file,
    ])
end

function _hpc_client_payload(bu::Builder)
    worker_id_line = "WORKER_ID=\"\${SLURM_ARRAY_TASK_ID:-\${SGE_TASK_ID:-1}}\""
    client_cmd = _command_string(_hpc_client_base_cmd(bu)) * " --worker-id \"\$WORKER_ID\""
    worker_id_line * "\n" * client_cmd
end

function _render_hpc_client_job_script(bu::Builder, scheduler::SchedulerConfig, workdir)
    mkpath(workdir)
    template = get_job_script_content(scheduler)
    if !contains(template, HPC_CLIENT_CMD_PLACEHOLDER)
        throw(
            ArgumentError(
                "HPC scheduler script `$(scheduler.script)` must contain the placeholder `$HPC_CLIENT_CMD_PLACEHOLDER`.",
            ),
        )
    end
    rendered = replace(template, HPC_CLIENT_CMD_PLACEHOLDER => _hpc_client_payload(bu))
    script_path = joinpath(workdir, "job.sh")
    _hpc_atomic_write(script_path, rendered)
    return script_path
end

function _hpc_submit_clients(bu::Builder, phase::AbstractString, iter::Int)
    scheduler = bu.options.scheduler
    scheduler === nothing && return ""
    workdir = joinpath(_hpc_root(bu), "submit", "gen$(iter)-$(phase)")
    _render_hpc_client_job_script(bu, scheduler, workdir)
    scheduler.submit_jobs || return ""
    cmd = _training_submit_cmd(scheduler, workdir)
    @info "HPC client submission command: $cmd"
    output = readchomp(cmd)
    !isempty(output) && @info "HPC client submission output: $output"
    return _parse_scheduler_job_id(scheduler, output)
end

function _hpc_stop_requested(bu::Builder, control, phase_dir, worker_id)
    should_stop(bu) && return true
    isfile(joinpath(_hpc_root(bu), "stop")) && return true
    isfile(joinpath(phase_dir, "STOP")) && return true
    isfile(joinpath(phase_dir, "workers", "$(worker_id).STOP")) && return true
    current = _hpc_read_control(bu)
    current === nothing && return false
    get(current, "status", "running") in ("stop", "stopping", "done", "error") && return true
    get(current, "epoch", get(control, "epoch", 0)) != get(control, "epoch", 0) && return true
    return false
end

function _hpc_process_pid(proc)
    try
        return getpid(proc)
    catch
        return ""
    end
end

function _hpc_signal_process(proc, signal)
    try
        kill(proc, signal)
    catch
        try
            kill(proc)
        catch err
            @warn "Failed to signal child process: $err"
        end
    end
end

function _hpc_stop_child!(proc, hpc::HPCWorkflowOption)
    process_running(proc) || return
    _hpc_signal_process(proc, Base.SIGTERM)
    started = time()
    while process_running(proc) && time() - started < hpc.child_grace_seconds
        sleep(1)
    end
    process_running(proc) || return
    _hpc_signal_process(proc, Base.SIGKILL)
    started = time()
    while process_running(proc) && time() - started < hpc.child_kill_seconds
        sleep(1)
    end
end

function _hpc_supervise_child!(
    bu::Builder,
    worker_id,
    control,
    phase_dir,
    cmd::Cmd;
    state,
    claim=nothing,
    model_path=nothing,
)
    proc = run(cmd; wait=false)
    child_pid = _hpc_process_pid(proc)
    last_heartbeat = 0.0
    while process_running(proc)
        if time() - last_heartbeat >= bu.options.hpc.heartbeat_interval
            hpc_write_heartbeat(
                bu,
                worker_id;
                state,
                phase=get(control, "phase", ""),
                iteration=Int(get(control, "iteration", bu.state.iteration)),
                epoch=Int(get(control, "epoch", 0)),
                child_pid,
                claim,
                model_path,
                message="child running",
                run_id=string(get(control, "run_id", "")),
            )
            last_heartbeat = time()
        end
        if _hpc_stop_requested(bu, control, phase_dir, worker_id)
            hpc_write_heartbeat(
                bu,
                worker_id;
                state="STOPPING_CHILD",
                phase=get(control, "phase", ""),
                iteration=Int(get(control, "iteration", bu.state.iteration)),
                epoch=Int(get(control, "epoch", 0)),
                child_pid,
                claim,
                model_path,
                message="stop requested",
                run_id=string(get(control, "run_id", "")),
            )
            _hpc_stop_child!(proc, bu.options.hpc)
            return false
        end
        sleep(max(1.0, bu.options.hpc.poll_interval))
    end
    wait(proc)
    return success(proc)
end

function _hpc_run_dft_client!(bu::Builder, worker_id, control)
    phase_dir = control["phase_dir"]
    manifest = _hpc_read_json(_hpc_phase_manifest_path(phase_dir))
    manifest === nothing && throw(ArgumentError("Missing DFT manifest `$phase_dir`."))
    cmd = Cmd(`crud.pl -mpinp $(Int(manifest["mpinp"]))`, dir=manifest["crud_workdir"])
    ok = _hpc_supervise_child!(bu, worker_id, control, phase_dir, cmd; state="DFT_RUNNING")
    hpc_write_heartbeat(
        bu,
        worker_id;
        state=ok ? "DFT_DONE" : "STOPPED",
        phase="dft",
        iteration=Int(control["iteration"]),
        epoch=Int(control["epoch"]),
        message=ok ? "crud.pl finished" : "crud.pl stopped",
        run_id=string(control["run_id"]),
    )
    _hpc_write_event(phase_dir, worker_id, ok ? "dft-done" : "dft-stopped")
    return ok
end

function _hpc_train_slot_base_cmd(bu::Builder, slot_id::Int, model_path::AbstractString)
    project_path = dirname(Base.active_project())
    builder_file = bu.state.builder_file_path
    Cmd([
        Base.julia_cmd()...,
        "--project=$(project_path)",
        "-e",
        "using EDDPotentials; EDDPotentials.hpc_train_slot()",
        "--",
        "--file",
        builder_file,
        "--iteration",
        string(bu.state.iteration),
        "--slot-id",
        string(slot_id),
        "--model-path",
        model_path,
    ])
end

function _hpc_run_training_client!(bu::Builder, worker_id, control)
    iter = Int(control["iteration"])
    phase_dir = control["phase_dir"]
    claim = hpc_claim_training_slot!(bu, iter, worker_id)
    if claim === nothing
        hpc_write_heartbeat(
            bu,
            worker_id;
            state="EXIT",
            phase="training",
            iteration=iter,
            epoch=Int(control["epoch"]),
            message="no training slot available",
            run_id=string(control["run_id"]),
        )
        return true
    end
    slot_id, claim_path, model_path = claim
    hpc_write_heartbeat(
        bu,
        worker_id;
        state="TRAINING_CLAIM",
        phase="training",
        iteration=iter,
        epoch=Int(control["epoch"]),
        claim=claim_path,
        model_path,
        message="claimed training slot",
        run_id=string(control["run_id"]),
    )
    cmd = _hpc_train_slot_base_cmd(bu, slot_id, model_path)
    ok = _hpc_supervise_child!(
        bu,
        worker_id,
        control,
        phase_dir,
        cmd;
        state="TRAINING_RUNNING",
        claim=claim_path,
        model_path,
    )
    if ok && isfile(model_path)
        hpc_mark_training_slot_done!(bu, iter, slot_id, model_path)
        _hpc_write_event(phase_dir, worker_id, "train-done", Dict("slot_id" => slot_id))
    else
        hpc_mark_training_slot_failed!(bu, iter, slot_id, ok ? "model artifact missing" : "training stopped")
        _hpc_write_event(phase_dir, worker_id, "train-failed", Dict("slot_id" => slot_id))
    end
    hpc_write_heartbeat(
        bu,
        worker_id;
        state=ok ? "TRAINING_DONE" : "STOPPED",
        phase="training",
        iteration=iter,
        epoch=Int(control["epoch"]),
        claim=claim_path,
        model_path,
        message=ok ? "training slot finished" : "training slot stopped",
        run_id=string(control["run_id"]),
    )
    return ok
end

function hpc_client(file::AbstractString; worker_id::AbstractString="")
    bu = Builder(file)
    isempty(worker_id) && (worker_id = get(ENV, "WORKER_ID", string(getpid())))
    hpc_write_heartbeat(bu, worker_id; state="BOOT", message="client boot")
    control = _hpc_read_control(bu)
    if control === nothing
        hpc_write_heartbeat(bu, worker_id; state="EXIT", message="no active control")
        return false
    end
    bu.state.iteration = Int(control["iteration"])
    phase = string(control["phase"])
    hpc_write_heartbeat(
        bu,
        worker_id;
        state="WAIT",
        phase,
        iteration=bu.state.iteration,
        epoch=Int(control["epoch"]),
        message="control loaded",
        run_id=string(control["run_id"]),
    )
    if phase == "dft"
        return _hpc_run_dft_client!(bu, worker_id, control)
    elseif phase == "training"
        return _hpc_run_training_client!(bu, worker_id, control)
    else
        hpc_write_heartbeat(
            bu,
            worker_id;
            state="EXIT",
            phase,
            iteration=bu.state.iteration,
            epoch=Int(get(control, "epoch", 0)),
            message="no runnable phase",
            run_id=string(get(control, "run_id", "")),
        )
        return true
    end
end

function hpc_client()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--file"
        help = "Name of the link TOML/YAML file"
        default = "link.toml"
        arg_type = String
        "--worker-id"
        help = "Worker id"
        default = ""
        arg_type = String
    end
    args = parse_args(s)
    hpc_client(args["file"]; worker_id=args["worker-id"])
end

link_worker(args...; kwargs...) = hpc_client(args...; kwargs...)

function hpc_train_slot()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--file"
        help = "Name of the link TOML/YAML file"
        default = "link.toml"
        arg_type = String
        "--iteration"
        help = "Override the iteration setting of the builder"
        default = -1
        arg_type = Int
        "--slot-id"
        help = "Training slot id"
        required = true
        arg_type = Int
        "--model-path"
        help = "Output model path"
        required = true
        arg_type = String
    end
    args = parse_args(s)
    bu = Builder(args["file"])
    if args["iteration"] >= 0
        bu.state.iteration = args["iteration"]
    end
    run_trainer_slot!(
        bu,
        bu.trainer;
        slot_id=args["slot-id"],
        model_path=args["model-path"],
    )
end

function hpc_master(bu::Builder)
    while bu.state.iteration <= bu.state.max_iterations
        should_stop(bu) && return bu
        iter = bu.state.iteration
        if has_ensemble(bu, iter)
            bu.state.iteration += 1
            continue
        end
        _generate_random_structures(bu, iter)
        should_stop(bu) && return bu
        if !is_training_data_ready(bu, iter)
            hpc_run_dft_phase!(bu, iter)
        end
        if bu.state.run_walk_forward
            walk_forward_tests(bu; print_results=true, iters=[iter - 1])
            should_stop(bu) && return bu
        end
        hpc_run_training_phase!(bu, iter)
        bu.state.iteration += 1
    end
    return bu
end

function hpc_master(file::AbstractString="link.toml")
    bu = Builder(file)
    hpc_master(bu)
end

link_hpc(args...; kwargs...) = hpc_master(args...; kwargs...)
