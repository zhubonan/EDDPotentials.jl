#=
File-backed Client-server master/client workflow for link.

The master owns phase transitions and shared workdir preparation. Clients are
small state-machine supervisors that read `control.json`, launch one child
process, and report liveness through atomic heartbeat files.
=#

using Dates

const CLIENT_SERVER_CMD_PLACEHOLDER = "{{EDDP_CLIENT_SERVER_CMD}}"

_client_server_root(bu::Builder) = joinpath(bu.state.workdir, bu.options.client_server.coordination_dir)
_client_server_control_path(bu::Builder) = joinpath(_client_server_root(bu), "control.json")
_client_server_workers_dir(bu::Builder) = joinpath(_client_server_root(bu), "workers")
_client_server_phase_dir(bu::Builder, iter::Int, phase::AbstractString) =
    joinpath(_client_server_root(bu), "phases", "gen$(iter)-$(phase)")
_client_server_phase_manifest_path(phase_dir) = joinpath(phase_dir, "manifest.json")
_client_server_events_dir(phase_dir) = joinpath(phase_dir, "events")
_client_server_worker_path(bu::Builder, worker_id) =
    joinpath(_client_server_workers_dir(bu), "$(worker_id).json")

_client_server_timestamp() = Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sssZ")

function _client_server_atomic_write(path::AbstractString, text::AbstractString)
    mkpath(dirname(path))
    tmp = path * ".tmp-$(getpid())-$(rand(UInt))"
    open(tmp, "w") do io
        write(io, text)
    end
    mv(tmp, path; force=true)
    return path
end

function _client_server_write_json(path::AbstractString, data)
    io = IOBuffer()
    JSON.print(io, data, 4)
    _client_server_atomic_write(path, String(take!(io)))
end

function _client_server_read_json(path::AbstractString)
    isfile(path) || return nothing
    try
        return JSON.parsefile(path)
    catch err
        @warn "Failed to parse client-server JSON file `$path`: $err"
        return nothing
    end
end

function _client_server_marker(path::AbstractString)
    mkpath(dirname(path))
    touch(path)
    return path
end

function _client_server_write_event(phase_dir, worker_id, kind, data=Dict{String,Any}())
    event_dir = _client_server_events_dir(phase_dir)
    name = "$(replace(_client_server_timestamp(), ':' => ""))-$(worker_id)-$(kind)-$(rand(UInt)).json"
    payload = merge(
        Dict{String,Any}(
            "worker_id" => string(worker_id),
            "kind" => string(kind),
            "created_at" => _client_server_timestamp(),
        ),
        Dict{String,Any}(data),
    )
    _client_server_write_json(joinpath(event_dir, name), payload)
end

function client_server_write_heartbeat(
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
        "started_at" => _client_server_timestamp(),
        "updated_at" => _client_server_timestamp(),
        "message" => message,
    )
    _client_server_write_json(_client_server_worker_path(bu, worker_id), payload)
end

function client_server_worker_states(bu::Builder)
    dir = _client_server_workers_dir(bu)
    isdir(dir) || return Dict{String,Any}[]
    states = Dict{String,Any}[]
    for path in sort(glob_allow_abs(joinpath(dir, "*.json")))
        state = _client_server_read_json(path)
        state === nothing || push!(states, state)
    end
    return states
end

function _client_server_publish_control!(
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
            "updated_at" => _client_server_timestamp(),
        ),
        Dict{String,Any}(extra),
    )
    _client_server_write_json(_client_server_control_path(bu), payload)
    return payload
end

function _client_server_read_control(bu::Builder)
    control = _client_server_read_json(_client_server_control_path(bu))
    control === nothing && return nothing
    get(control, "builder_uuid", "") == builder_uuid(bu) || return nothing
    return control
end

function client_server_request_shutdown!(bu::Builder)
    root = _client_server_root(bu)
    _client_server_marker(joinpath(root, "SHUTDOWN"))
    phase_dir = joinpath(root, "shutdown")
    _client_server_publish_control!(
        bu;
        phase="shutdown",
        phase_dir,
        status="shutdown",
    )
    return true
end

shutdown_link_backend!(::ClientServerLinkBackend, bu::Builder) =
    bu.options.client_server.enabled ? client_server_request_shutdown!(bu) : nothing

function _client_server_clear_shutdown!(bu::Builder)
    path = joinpath(_client_server_root(bu), "SHUTDOWN")
    isfile(path) && rm(path; force=true)
    return true
end

function client_server_prepare_crud_workdir!(bu::Builder, iter::Int=bu.state.iteration)
    state = bu.state
    phase_dir = _client_server_phase_dir(bu, iter, "dft")
    crud_workdir = joinpath(state.workdir, bu.options.client_server.dft_crud_workdir, "gen$(iter)")
    input_dir = input_structure_dir(state.workdir, iter)
    output_dir = output_structure_dir(state.workdir, iter)
    input_files = sort(glob_allow_abs(joinpath(input_dir, "*.res")))
    manifest_path = _client_server_phase_manifest_path(phase_dir)

    if isfile(joinpath(phase_dir, "DONE")) && is_training_data_ready(bu, iter)
        return crud_workdir
    elseif isfile(joinpath(phase_dir, "READY"))
        manifest = _client_server_read_json(manifest_path)
        if manifest !== nothing &&
           Int(get(manifest, "input_count", -1)) == length(input_files) &&
           get(manifest, "input_dir", "") == input_dir &&
           get(manifest, "output_dir", "") == output_dir
            return crud_workdir
        end
    end

    mkpath(phase_dir)
    for marker in ("READY", "DONE", "FAILED", "STOP")
        path = joinpath(phase_dir, marker)
        isfile(path) && rm(path; force=true)
    end
    for dir in ("hopper", "good_castep", "bad_castep")
        path = joinpath(crud_workdir, dir)
        isdir(path) && rm(path; recursive=true)
        mkpath(path)
    end
    mkpath(output_dir)

    seed = splitext(joinpath(state.workdir, state.seedfile_calc))[1]
    cp(seed * ".cell", joinpath(crud_workdir, basename(seed) * ".cell"); force=true)
    cp(seed * ".param", joinpath(crud_workdir, basename(seed) * ".param"); force=true)

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
        "created_at" => _client_server_timestamp(),
    )
    _client_server_write_json(manifest_path, manifest)
    _client_server_marker(joinpath(phase_dir, "READY"))
    return crud_workdir
end

function _client_server_dft_counts(crud_workdir)
    good = length(glob_allow_abs(joinpath(crud_workdir, "good_castep", "*.res")))
    bad = length(glob_allow_abs(joinpath(crud_workdir, "bad_castep", "*.res")))
    return good, bad
end

function client_server_collect_crud_results!(bu::Builder, iter::Int=bu.state.iteration)
    phase_dir = _client_server_phase_dir(bu, iter, "dft")
    manifest = _client_server_read_json(_client_server_phase_manifest_path(phase_dir))
    manifest === nothing && throw(ArgumentError("Missing DFT manifest for generation $iter."))
    crud_workdir = manifest["crud_workdir"]
    output_dir = manifest["output_dir"]
    mkpath(output_dir)
    for file in glob_allow_abs(joinpath(crud_workdir, "good_castep", "*.res"))
        mv(file, joinpath(output_dir, basename(file)); force=true)
    end
    _client_server_marker(joinpath(phase_dir, "DONE"))
    return output_dir
end

function _client_server_required_dft_terminal(manifest)
    n = Int(manifest["input_count"])
    n == 0 && return 0
    return max(1, ceil(Int, n * Float64(manifest["threshold"])))
end

function _client_server_required_dft_success(manifest)
    _client_server_required_dft_terminal(manifest)
end

function _client_server_max_dft_bad(manifest)
    n = Int(manifest["input_count"])
    max(0, n - _client_server_required_dft_success(manifest))
end

function _client_server_request_phase_stop!(bu::Builder, phase_dir::AbstractString)
    _client_server_marker(joinpath(phase_dir, "STOP"))
end

function _client_server_abort_phase!(bu::Builder, phase_dir::AbstractString, message)
    _client_server_marker(joinpath(phase_dir, "STOP"))
    _client_server_write_json(
        joinpath(phase_dir, "ABORTED.json"),
        Dict{String,Any}("message" => string(message), "aborted_at" => _client_server_timestamp()),
    )
end

function _client_server_phase_worker_is_active(state, control)
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

function _client_server_wait_for_phase_workers_stopped!(bu::Builder, control)
    timeout =
        bu.options.client_server.client_child_grace_seconds +
        bu.options.client_server.client_child_kill_seconds +
        bu.options.client_server.client_heartbeat_interval_seconds +
        2 * bu.options.client_server.control_poll_interval_seconds
    started = time()
    while true
        active = filter(state -> _client_server_phase_worker_is_active(state, control), client_server_worker_states(bu))
        isempty(active) && return true
        if time() - started >= timeout
            @warn "Timed out waiting for Client-server clients to stop" active_workers = length(active)
            return false
        end
        sleep(max(1.0, bu.options.client_server.control_poll_interval_seconds))
    end
end

function _client_server_wait_for_dft!(bu::Builder, phase_dir)
    manifest = _client_server_read_json(_client_server_phase_manifest_path(phase_dir))
    manifest === nothing && throw(ArgumentError("Missing DFT manifest `$phase_dir`."))
    crud_workdir = manifest["crud_workdir"]
    input_count = Int(manifest["input_count"])
    required_success = _client_server_required_dft_success(manifest)
    max_bad = _client_server_max_dft_bad(manifest)
    while true
        good, bad = _client_server_dft_counts(crud_workdir)
        if good >= required_success
            _client_server_request_phase_stop!(bu, phase_dir)
            return good, bad
        elseif bad > max_bad || good + bad >= input_count
            _client_server_marker(joinpath(phase_dir, "FAILED"))
            bad_ratio = input_count == 0 ? 0.0 : bad / input_count
            throw(
                ErrorException(
                    "Too many failed DFT calculations for generation $(manifest["iteration"]): $bad bad structures out of $input_count total inputs (bad ratio=$(round(100 * bad_ratio; digits=1))%). Need at least $required_success successful calculations, but only $good are available.",
                ),
            )
        end
        if should_stop(bu)
            _client_server_abort_phase!(bu, phase_dir, "STOP requested while waiting for DFT.")
            return :stopped
        end
        sleep(bu.options.client_server.control_poll_interval_seconds)
    end
end

function client_server_run_dft_phase!(bu::Builder, iter::Int=bu.state.iteration)
    is_training_data_ready(bu, iter) && return :ok
    _client_server_clear_shutdown!(bu)
    phase_dir = _client_server_phase_dir(bu, iter, "dft")
    crud_workdir = client_server_prepare_crud_workdir!(bu, iter)
    manifest = _client_server_read_json(_client_server_phase_manifest_path(phase_dir))
    control = _client_server_publish_control!(
        bu;
        iteration=iter,
        phase="dft",
        phase_dir,
        extra=Dict("dft" => manifest),
    )
    _client_server_submit_clients(bu, "dft", iter)
    result = try
        _client_server_wait_for_dft!(bu, phase_dir)
    catch
        _client_server_publish_control!(bu; iteration=iter, phase="dft", phase_dir, status="stopping")
        rethrow()
    end
    if result == :stopped
        _client_server_publish_control!(bu; iteration=iter, phase="dft", phase_dir, status="stopping")
        _client_server_wait_for_phase_workers_stopped!(bu, control)
        return :stopped
    end
    _client_server_wait_for_phase_workers_stopped!(bu, control)
    client_server_collect_crud_results!(bu, iter)
    _client_server_publish_control!(bu; iteration=iter, phase="idle", phase_dir, status="done")
    return :ok
end

function _client_server_training_phase_dir(bu::Builder, iter=bu.state.iteration)
    _client_server_phase_dir(bu, iter, "train")
end

function _client_server_training_slots_dir(bu::Builder, iter=bu.state.iteration)
    joinpath(_client_server_training_phase_dir(bu, iter), "slots")
end

function _client_server_slot_name(slot_id::Int)
    lpad(string(slot_id), 6, '0')
end

function _client_server_training_manifest_matches(bu::Builder, manifest, iter::Int, dataset_path)
    manifest === nothing && return false
    Int(get(manifest, "iteration", -1)) == iter || return false
    get(manifest, "dataset_path", "") == dataset_path || return false
    Int(get(manifest, "nmodels", -1)) == bu.trainer.nmodels || return false
    Int(get(manifest, "required_models", -1)) == _external_required_models(bu.trainer) || return false
    get(manifest, "prefix", "") == bu.trainer.prefix || return false
    return true
end

function _client_server_clear_training_phase!(bu::Builder, phase_dir, iter::Int)
    for marker in ("READY", "DONE", "FAILED", "STOP", "ABORTED.json")
        path = joinpath(phase_dir, marker)
        isfile(path) && rm(path; force=true)
    end
    for dir in ("slots", "events")
        path = joinpath(phase_dir, dir)
        isdir(path) && rm(path; recursive=true, force=true)
    end
    pattern = joinpath(
        bu.state.workdir,
        TRAINING_DIR,
        "$(bu.trainer.prefix)gen-$(iter)-model-*.json",
    )
    for path in glob_allow_abs(pattern)
        rm(path; force=true)
    end
end

function client_server_prepare_training_phase!(bu::Builder, iter::Int=bu.state.iteration)
    phase_dir = _client_server_training_phase_dir(bu, iter)
    dataset_path = dataset_name(bu, iter)
    manifest_path = _client_server_phase_manifest_path(phase_dir)
    manifest = _client_server_read_json(manifest_path)
    if (isfile(joinpath(phase_dir, "READY")) || isfile(joinpath(phase_dir, "DONE"))) &&
       _client_server_training_manifest_matches(bu, manifest, iter, dataset_path)
        return phase_dir
    end

    mkpath(joinpath(phase_dir, "slots"))
    mkpath(_client_server_events_dir(phase_dir))
    _client_server_clear_training_phase!(bu, phase_dir, iter)
    mkpath(joinpath(phase_dir, "slots"))
    mkpath(_client_server_events_dir(phase_dir))
    manifest = Dict{String,Any}(
        "phase" => "training",
        "iteration" => iter,
        "dataset_path" => dataset_path,
        "training_dir" => joinpath(bu.state.workdir, TRAINING_DIR),
        "nmodels" => bu.trainer.nmodels,
        "required_models" => _external_required_models(bu.trainer),
        "prefix" => bu.trainer.prefix,
        "created_at" => _client_server_timestamp(),
    )
    _client_server_write_json(manifest_path, manifest)
    _client_server_marker(joinpath(phase_dir, "READY"))
    return phase_dir
end

function client_server_training_model_path(bu::Builder, iter::Int, slot_id::Int)
    slot = _client_server_slot_name(slot_id)
    joinpath(
        bu.state.workdir,
        TRAINING_DIR,
        "$(bu.trainer.prefix)gen-$(iter)-model-slot-$(slot).json",
    )
end

function client_server_claim_training_slot!(bu::Builder, iter::Int, worker_id::AbstractString)
    slots_dir = _client_server_training_slots_dir(bu, iter)
    mkpath(slots_dir)
    now = time()
    for slot_id = 1:bu.trainer.nmodels
        slot = _client_server_slot_name(slot_id)
        done_path = joinpath(slots_dir, "$(slot).done.json")
        failed_path = joinpath(slots_dir, "$(slot).failed.json")
        lock_path = joinpath(slots_dir, "$(slot).lock")
        isfile(done_path) && continue
        if isdir(lock_path)
            age = now - stat(lock_path).mtime
            if age > bu.options.client_server.training_slot_stale_seconds
                rm(lock_path; recursive=true, force=true)
            else
                continue
            end
        end
        try
            isfile(failed_path) && rm(failed_path; force=true)
            mkdir(lock_path)
            model_path = client_server_training_model_path(bu, iter, slot_id)
            claim = Dict{String,Any}(
                "slot_id" => slot_id,
                "worker_id" => worker_id,
                "host" => get(ENV, "HOSTNAME", ""),
                "pid" => getpid(),
                "model_path" => model_path,
                "claimed_at" => _client_server_timestamp(),
            )
            _client_server_write_json(joinpath(lock_path, "claim.json"), claim)
            return slot_id, lock_path, model_path
        catch err
            err isa IOError || rethrow()
        end
    end
    return nothing
end

function client_server_mark_training_slot_done!(bu::Builder, iter::Int, slot_id::Int, model_path)
    slots_dir = _client_server_training_slots_dir(bu, iter)
    slot = _client_server_slot_name(slot_id)
    payload = Dict{String,Any}(
        "slot_id" => slot_id,
        "model_path" => model_path,
        "done_at" => _client_server_timestamp(),
    )
    _client_server_write_json(joinpath(slots_dir, "$(slot).done.json"), payload)
end

function client_server_mark_training_slot_failed!(bu::Builder, iter::Int, slot_id::Int, message)
    slots_dir = _client_server_training_slots_dir(bu, iter)
    slot = _client_server_slot_name(slot_id)
    lock_path = joinpath(slots_dir, "$(slot).lock")
    payload = Dict{String,Any}(
        "slot_id" => slot_id,
        "message" => message,
        "failed_at" => _client_server_timestamp(),
    )
    _client_server_write_json(joinpath(slots_dir, "$(slot).failed.json"), payload)
    isdir(lock_path) && rm(lock_path; recursive=true, force=true)
end

function _client_server_wait_for_training!(bu::Builder, phase_dir)
    manifest = _client_server_read_json(_client_server_phase_manifest_path(phase_dir))
    manifest === nothing && throw(ArgumentError("Missing training manifest `$phase_dir`."))
    required = Int(manifest["required_models"])
    while true
        nm = num_existing_models(bu)
        if nm >= required
            _client_server_request_phase_stop!(bu, phase_dir)
            return nm
        end
        if should_stop(bu)
            _client_server_abort_phase!(bu, phase_dir, "STOP requested while waiting for training.")
            return :stopped
        end
        sleep(bu.options.client_server.control_poll_interval_seconds)
    end
end

function client_server_run_training_phase!(bu::Builder, iter::Int=bu.state.iteration)
    _models_ready_for_ensemble(bu) && return :ok
    _client_server_clear_shutdown!(bu)
    phase_dir = client_server_prepare_training_phase!(bu, iter)
    manifest = _client_server_read_json(_client_server_phase_manifest_path(phase_dir))
    control = _client_server_publish_control!(
        bu;
        iteration=iter,
        phase="training",
        phase_dir,
        extra=Dict("training" => manifest),
    )
    _client_server_submit_clients(bu, "training", iter)
    result = try
        _client_server_wait_for_training!(bu, phase_dir)
    catch err
        _client_server_publish_control!(bu; iteration=iter, phase="training", phase_dir, status="stopping")
        rethrow()
    end
    if result == :stopped
        _client_server_publish_control!(bu; iteration=iter, phase="training", phase_dir, status="stopping")
        _client_server_wait_for_phase_workers_stopped!(bu, control)
        return :stopped
    end
    _client_server_wait_for_phase_workers_stopped!(bu, control)
    _client_server_marker(joinpath(phase_dir, "DONE"))
    _client_server_publish_control!(bu; iteration=iter, phase="idle", phase_dir, status="done")
    nm = num_existing_models(bu)
    nm >= _external_required_models(bu.trainer) && return :ok
    throw(ErrorException("Only $nm models are found in the training directory."))
end

function run_dft_phase!(::ClientServerLinkBackend, bu::Builder, iter::Int)
    result = client_server_run_dft_phase!(bu, iter)
    result == :stopped && return LinkPhaseResult(:stopped, :dft, iter)
    return LinkPhaseResult(:ok, :dft, iter)
end

function run_training_phase!(::ClientServerLinkBackend, bu::Builder, iter::Int)
    result = client_server_run_training_phase!(bu, iter)
    result == :stopped && return LinkPhaseResult(:stopped, :train_models, iter)
    return LinkPhaseResult(:ok, :train_models, iter)
end

function _client_server_client_julia_code()
    "using EDDPotentials; EDDPotentials.client_server_client()"
end

function _client_server_client_base_cmd(bu::Builder)
    project_path = dirname(Base.active_project())
    builder_file = _require_builder_file_path!(bu)
    Cmd([
        Base.julia_cmd()...,
        "--project=$(project_path)",
        "-e",
        _client_server_client_julia_code(),
        "--",
        "--file",
        builder_file,
    ])
end

function _client_server_client_payload(bu::Builder)
    worker_id_line = "WORKER_ID=\"\${SLURM_ARRAY_TASK_ID:-\${SGE_TASK_ID:-1}}\""
    client_cmd = _command_string(_client_server_client_base_cmd(bu)) * " --worker-id \"\$WORKER_ID\""
    worker_id_line * "\n" * client_cmd
end

function _render_client_server_client_job_script(bu::Builder, scheduler::SchedulerConfig, workdir)
    mkpath(workdir)
    template = get_job_script_content(scheduler)
    if !contains(template, CLIENT_SERVER_CMD_PLACEHOLDER)
        throw(
            ArgumentError(
                "Client-server scheduler script `$(scheduler.script)` must contain the placeholder `$CLIENT_SERVER_CMD_PLACEHOLDER`.",
            ),
        )
    end
    rendered = replace(template, CLIENT_SERVER_CMD_PLACEHOLDER => _client_server_client_payload(bu))
    script_path = joinpath(workdir, "job.sh")
    _client_server_atomic_write(script_path, rendered)
    return script_path
end

function _client_server_submit_clients(bu::Builder, phase::AbstractString, iter::Int)
    scheduler = bu.options.scheduler
    scheduler === nothing && return ""
    workdir = joinpath(_client_server_root(bu), "submit", "gen$(iter)-$(phase)")
    _render_client_server_client_job_script(bu, scheduler, workdir)
    bu.options.client_server.submit_client_jobs || return ""
    nclients = if phase == "dft"
        bu.options.client_server.dft_client_jobs
    elseif phase == "training"
        bu.options.client_server.training_client_jobs
    else
        scheduler.njobs
    end
    client_server_scheduler = SchedulerConfig(
        scheduler.script,
        max(1, nclients),
        scheduler.type,
        scheduler.clean_workdir,
        scheduler.submit_jobs,
    )
    cmd = _training_submit_cmd(client_server_scheduler, workdir)
    @info "Client-server client submission command: $cmd"
    output = readchomp(cmd)
    !isempty(output) && @info "Client-server client submission output: $output"
    return _parse_scheduler_job_id(scheduler, output)
end

function _client_server_stop_requested(bu::Builder, control, phase_dir, worker_id)
    should_stop(bu) && return true
    isfile(joinpath(_client_server_root(bu), "stop")) && return true
    isfile(joinpath(phase_dir, "STOP")) && return true
    isfile(joinpath(phase_dir, "workers", "$(worker_id).STOP")) && return true
    current = _client_server_read_control(bu)
    current === nothing && return false
    get(current, "status", "running") in ("stop", "stopping", "done", "error") && return true
    get(current, "epoch", get(control, "epoch", 0)) != get(control, "epoch", 0) && return true
    return false
end

function _client_server_process_pid(proc)
    try
        return getpid(proc)
    catch
        return ""
    end
end

function _client_server_signal_process(proc, signal)
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

function _client_server_stop_child!(proc, client_server::ClientServerOption)
    process_running(proc) || return
    _client_server_signal_process(proc, Base.SIGTERM)
    started = time()
    while process_running(proc) && time() - started < client_server.client_child_grace_seconds
        sleep(1)
    end
    process_running(proc) || return
    _client_server_signal_process(proc, Base.SIGKILL)
    started = time()
    while process_running(proc) && time() - started < client_server.client_child_kill_seconds
        sleep(1)
    end
end

function _client_server_supervise_child!(
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
    child_pid = _client_server_process_pid(proc)
    last_heartbeat = 0.0
    while process_running(proc)
        if time() - last_heartbeat >= bu.options.client_server.client_heartbeat_interval_seconds
            client_server_write_heartbeat(
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
        if _client_server_stop_requested(bu, control, phase_dir, worker_id)
            client_server_write_heartbeat(
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
            _client_server_stop_child!(proc, bu.options.client_server)
            return false
        end
        sleep(max(1.0, bu.options.client_server.control_poll_interval_seconds))
    end
    wait(proc)
    return success(proc)
end

function _client_server_run_dft_client!(bu::Builder, worker_id, control)
    phase_dir = control["phase_dir"]
    manifest = _client_server_read_json(_client_server_phase_manifest_path(phase_dir))
    manifest === nothing && throw(ArgumentError("Missing DFT manifest `$phase_dir`."))
    cmd = Cmd(`crud.pl -mpinp $(Int(manifest["mpinp"]))`, dir=manifest["crud_workdir"])
    ok = _client_server_supervise_child!(bu, worker_id, control, phase_dir, cmd; state="DFT_RUNNING")
    client_server_write_heartbeat(
        bu,
        worker_id;
        state=ok ? "DFT_DONE" : "STOPPED",
        phase="dft",
        iteration=Int(control["iteration"]),
        epoch=Int(control["epoch"]),
        message=ok ? "crud.pl finished" : "crud.pl stopped",
        run_id=string(control["run_id"]),
    )
    _client_server_write_event(phase_dir, worker_id, ok ? "dft-done" : "dft-stopped")
    return ok
end

function _client_server_train_slot_base_cmd(bu::Builder, slot_id::Int, model_path::AbstractString)
    project_path = dirname(Base.active_project())
    builder_file = _require_builder_file_path!(bu)
    Cmd([
        Base.julia_cmd()...,
        "--project=$(project_path)",
        "-e",
        "using EDDPotentials; EDDPotentials.client_server_train_slot()",
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

function _client_server_run_training_client!(bu::Builder, worker_id, control)
    iter = Int(control["iteration"])
    phase_dir = control["phase_dir"]
    claim = client_server_claim_training_slot!(bu, iter, worker_id)
    if claim === nothing
        client_server_write_heartbeat(
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
    client_server_write_heartbeat(
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
    cmd = _client_server_train_slot_base_cmd(bu, slot_id, model_path)
    ok = _client_server_supervise_child!(
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
        client_server_mark_training_slot_done!(bu, iter, slot_id, model_path)
        _client_server_write_event(phase_dir, worker_id, "train-done", Dict("slot_id" => slot_id))
    else
        client_server_mark_training_slot_failed!(bu, iter, slot_id, ok ? "model artifact missing" : "training stopped")
        _client_server_write_event(phase_dir, worker_id, "train-failed", Dict("slot_id" => slot_id))
    end
    client_server_write_heartbeat(
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

function _client_server_shutdown_requested(bu::Builder)
    should_stop(bu) && return true
    isfile(joinpath(_client_server_root(bu), "stop")) && return true
    isfile(joinpath(_client_server_root(bu), "SHUTDOWN")) && return true
    control = _client_server_read_control(bu)
    control === nothing && return false
    string(get(control, "phase", "")) == "shutdown" && return true
    string(get(control, "status", "")) == "shutdown" && return true
    return false
end

function _client_server_run_control!(bu::Builder, worker_id, control, handled_dft_epochs::Set{Int})
    bu.state.iteration = Int(control["iteration"])
    phase = string(control["phase"])
    epoch = Int(control["epoch"])
    client_server_write_heartbeat(
        bu,
        worker_id;
        state="WAIT",
        phase,
        iteration=bu.state.iteration,
        epoch,
        message="control loaded",
        run_id=string(control["run_id"]),
    )
    if phase == "dft"
        epoch in handled_dft_epochs && return true
        ok = _client_server_run_dft_client!(bu, worker_id, control)
        push!(handled_dft_epochs, epoch)
        return ok
    elseif phase == "training"
        return _client_server_run_training_client!(bu, worker_id, control)
    elseif phase == "shutdown"
        return :shutdown
    else
        client_server_write_heartbeat(
            bu,
            worker_id;
            state="IDLE",
            phase,
            iteration=bu.state.iteration,
            epoch=Int(get(control, "epoch", 0)),
            message="no runnable phase",
            run_id=string(get(control, "run_id", "")),
        )
        return true
    end
end

function client_server_client(file::AbstractString; worker_id::AbstractString="", once::Bool=false)
    bu = Builder(file)
    isempty(worker_id) && (worker_id = get(ENV, "WORKER_ID", string(getpid())))
    client_server_write_heartbeat(bu, worker_id; state="BOOT", message="client boot")
    handled_dft_epochs = Set{Int}()
    while true
        if _client_server_shutdown_requested(bu)
            client_server_write_heartbeat(bu, worker_id; state="EXIT", message="shutdown requested")
            return true
        end
        control = _client_server_read_control(bu)
        if control === nothing
            client_server_write_heartbeat(bu, worker_id; state="IDLE", message="no active control")
            once && return false
            sleep(max(1.0, bu.options.client_server.client_poll_interval_seconds))
            continue
        end
        result = _client_server_run_control!(bu, worker_id, control, handled_dft_epochs)
        result == :shutdown && return true
        once && return result == true
        sleep(max(1.0, bu.options.client_server.client_poll_interval_seconds))
    end
end

function client_server_client()
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
        "--once"
        help = "Run one control phase and exit instead of polling as a daemon"
        action = :store_true
    end
    args = parse_args(s)
    client_server_client(args["file"]; worker_id=args["worker-id"], once=args["once"])
end

link_worker(args...; kwargs...) = client_server_client(args...; kwargs...)

function client_server_train_slot()
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

client_server_master(bu::Builder) = run_link_workflow!(bu, ClientServerLinkBackend())

function client_server_master(file::AbstractString="link.toml")
    bu = Builder(file)
    client_server_master(bu)
end

link_client_server(args...; kwargs...) = client_server_master(args...; kwargs...)
