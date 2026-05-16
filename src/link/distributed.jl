#=
File-backed Distributed link server/worker workflow for link.

The server owns phase transitions and shared workdir preparation. Workers are
small state-machine supervisors that read `control.json`, launch one child
process, and report liveness through atomic heartbeat files.
=#

using Dates
using Logging

const DISTRIBUTED_WORKER_CMD_PLACEHOLDER = "{{EDDP_DISTRIBUTED_WORKER_CMD}}"
const DISTRIBUTED_LOG_LEVELS = Dict(
    "debug" => Logging.Debug,
    "info" => Logging.Info,
    "warn" => Logging.Warn,
    "error" => Logging.Error,
)
const DISTRIBUTED_LOG_GROUP = :distributed

_coordination_root(bu::Builder) = joinpath(bu.state.workdir, bu.options.distributed.coordination_dir)
_control_path(bu::Builder) = joinpath(_coordination_root(bu), "control.json")
_worker_states_dir(bu::Builder) = joinpath(_coordination_root(bu), "workers")
_phase_dir(bu::Builder, iter::Int, phase::AbstractString) =
    joinpath(_coordination_root(bu), "phases", "gen$(iter)-$(phase)")
_phase_manifest_path(phase_dir) = joinpath(phase_dir, "manifest.json")
_phase_events_dir(phase_dir) = joinpath(phase_dir, "events")
_worker_state_path(bu::Builder, worker_id) =
    joinpath(_worker_states_dir(bu), "$(worker_id).json")

_timestamp() = Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sssZ")

struct DistributedTeeLogger <: Logging.AbstractLogger
    file_io::IO
    min_level::Logging.LogLevel
    role::String
    id::String
end

Logging.min_enabled_level(logger::DistributedTeeLogger) = logger.min_level
Logging.catch_exceptions(::DistributedTeeLogger) = true

Logging.shouldlog(logger::DistributedTeeLogger, level, _module, group, id) =
    level >= logger.min_level

function _distributed_level_name(level::Logging.LogLevel)
    level == Logging.Debug && return "DEBUG"
    level == Logging.Info && return "INFO"
    level == Logging.Warn && return "WARN"
    level == Logging.Error && return "ERROR"
    return string(level)
end

function _log_value_present(value)
    value === nothing && return false
    value isa AbstractString && isempty(value) && return false
    return true
end

function _escape_log_string(value)
    replace(String(value), "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n", "\r" => "\\r")
end

function _format_log_value(value; always_quote::Bool=false)
    if value isa Bool
        return value ? "true" : "false"
    elseif value isa Real
        return string(value)
    end
    text = value isa Symbol ? String(value) : string(value)
    if !always_quote && occursin(r"^[A-Za-z0-9_.:/@%+=,\-]+$", text)
        return text
    end
    return "\"$(_escape_log_string(text))\""
end

function _distributed_kwarg(kwargs, name::Symbol, default=nothing)
    for (key, value) in kwargs
        Symbol(key) == name && return value
    end
    return default
end

function _distributed_log_line(logger::DistributedTeeLogger, level, message; kwargs...)
    role = _distributed_kwarg(kwargs, :role, logger.role)
    log_id = _distributed_kwarg(kwargs, :id, nothing)
    if !_log_value_present(log_id)
        worker_id = _distributed_kwarg(kwargs, :worker_id, nothing)
        pid = _distributed_kwarg(kwargs, :pid, nothing)
        log_id = _log_value_present(worker_id) ? worker_id :
                 _log_value_present(pid) ? pid : logger.id
    end

    parts = String[
        "timestamp=$(_timestamp())",
        "level=$(_distributed_level_name(level))",
        "role=$(_format_log_value(role))",
        "id=$(_format_log_value(log_id))",
        "message=$(_format_log_value(message; always_quote=true))",
    ]
    for (key, value) in kwargs
        key_symbol = Symbol(key)
        key_symbol in (:_group, :role, :id, :worker_id, :pid) && continue
        _log_value_present(value) || continue
        push!(parts, "$(key_symbol)=$(_format_log_value(value))")
    end
    return join(parts, " ")
end

function Logging.handle_message(
    logger::DistributedTeeLogger,
    level,
    message,
    _module,
    group,
    id,
    file,
    line;
    kwargs...,
)
    log_line = _distributed_log_line(logger, level, message; kwargs...)
    println(stderr, log_line)
    flush(stderr)
    file_ok = true
    file_err = nothing
    try
        println(logger.file_io, log_line)
        flush(logger.file_io)
    catch err
        file_ok = false
        file_err = err
    end
    if !file_ok
        try
            println(stderr, "Warning: distributed link diagnostic file logging failed")
            println(stderr, "  error = ", file_err)
            flush(stderr)
        catch
        end
    end
end

struct DistributedDiagnosticLogger <: Logging.AbstractLogger
    parent_logger::Logging.AbstractLogger
    tee_logger::Union{Nothing,DistributedTeeLogger}
end

function Logging.min_enabled_level(logger::DistributedDiagnosticLogger)
    logger.tee_logger === nothing && return Logging.min_enabled_level(logger.parent_logger)
    return min(
        Logging.min_enabled_level(logger.parent_logger),
        Logging.min_enabled_level(logger.tee_logger),
    )
end

Logging.catch_exceptions(::DistributedDiagnosticLogger) = true

function Logging.shouldlog(
    logger::DistributedDiagnosticLogger,
    level,
    _module,
    group,
    id,
)
    if group == DISTRIBUTED_LOG_GROUP
        logger.tee_logger === nothing && return false
        return Logging.shouldlog(logger.tee_logger, level, _module, group, id)
    end
    return Logging.shouldlog(logger.parent_logger, level, _module, group, id)
end

function Logging.handle_message(
    logger::DistributedDiagnosticLogger,
    level,
    message,
    _module,
    group,
    id,
    file,
    line;
    kwargs...,
)
    if group == DISTRIBUTED_LOG_GROUP
        logger.tee_logger === nothing && return nothing
        return Logging.handle_message(logger.tee_logger, level, message, _module, group, id, file, line; kwargs...)
    end
    Logging.handle_message(
        logger.parent_logger,
        level,
        message,
        _module,
        group,
        id,
        file,
        line;
        kwargs...,
    )
end

function _distributed_log_level(level::AbstractString)
    key = lowercase(strip(level))
    haskey(DISTRIBUTED_LOG_LEVELS, key) ||
        throw(ArgumentError("Invalid distributed link log_level `$level`; expected one of debug, info, warn, error."))
    DISTRIBUTED_LOG_LEVELS[key]
end

function _distributed_log_path(bu::Builder; role::AbstractString, id::AbstractString)
    safe_role = replace(String(role), r"[^A-Za-z0-9_.-]" => "_")
    safe_id = replace(String(id), r"[^A-Za-z0-9_.-]" => "_")
    joinpath(
        _coordination_root(bu),
        bu.options.distributed.log_dir,
        "$safe_role-$safe_id.log",
    )
end

function _with_distributed_logger(f, bu::Builder; role::AbstractString, id::AbstractString)
    current_logger() isa DistributedDiagnosticLogger && return f()
    parent_logger = current_logger()
    if !bu.options.distributed.log_enabled
        logger = DistributedDiagnosticLogger(parent_logger, nothing)
        return with_logger(logger) do
            f()
        end
    end
    path = _distributed_log_path(bu; role, id)
    mkpath(dirname(path))
    open(path, "a") do io
        min_level = _distributed_log_level(bu.options.distributed.log_level)
        tee_logger = DistributedTeeLogger(
            io,
            min_level,
            String(role),
            String(id),
        )
        logger = DistributedDiagnosticLogger(parent_logger, tee_logger)
        with_logger(logger) do
            f()
        end
    end
end

_server_id() = string(getpid())

function _log_interval_due(last_emit::Float64, bu::Builder)
    return time() - last_emit >= bu.options.distributed.log_interval_seconds
end

function _atomic_write(path::AbstractString, text::AbstractString)
    mkpath(dirname(path))
    tmp = path * ".tmp-$(getpid())-$(rand(UInt))"
    open(tmp, "w") do io
        write(io, text)
    end
    mv(tmp, path; force=true)
    return path
end

function _write_json(path::AbstractString, data)
    io = IOBuffer()
    JSON.print(io, data, 4)
    _atomic_write(path, String(take!(io)))
end

function _read_json(path::AbstractString)
    isfile(path) || return nothing
    try
        return JSON.parsefile(path)
    catch err
        @warn "Failed to parse distributed link JSON file `$path`: $err"
        return nothing
    end
end

function _touch_marker(path::AbstractString)
    mkpath(dirname(path))
    touch(path)
    return path
end

function _write_phase_event(phase_dir, worker_id, kind, data=Dict{String,Any}())
    event_dir = _phase_events_dir(phase_dir)
    name = "$(replace(_timestamp(), ':' => ""))-$(worker_id)-$(kind)-$(rand(UInt)).json"
    payload = merge(
        Dict{String,Any}(
            "worker_id" => string(worker_id),
            "kind" => string(kind),
            "created_at" => _timestamp(),
        ),
        Dict{String,Any}(data),
    )
    _write_json(joinpath(event_dir, name), payload)
end

mutable struct WorkerState
    worker_id::String
    state::String
    phase::String
    iteration::Int
    epoch::Int
    run_id::String
    child_pid::Any
    claim::Any
    model_path::Any
    message::String
end

function WorkerState(
    worker_id::AbstractString;
    state::AbstractString="BOOT",
    phase::AbstractString="",
    iteration::Int=0,
    epoch::Int=0,
    run_id::AbstractString="",
    child_pid=nothing,
    claim=nothing,
    model_path=nothing,
    message::AbstractString="",
)
    return WorkerState(
        String(worker_id),
        String(state),
        String(phase),
        iteration,
        epoch,
        String(run_id),
        child_pid,
        claim,
        model_path,
        String(message),
    )
end

function _update_worker_state!(
    worker::WorkerState;
    state::AbstractString=worker.state,
    phase::AbstractString=worker.phase,
    iteration::Int=worker.iteration,
    epoch::Int=worker.epoch,
    run_id::AbstractString=worker.run_id,
    child_pid=worker.child_pid,
    claim=worker.claim,
    model_path=worker.model_path,
    message::AbstractString=worker.message,
)
    worker.state = String(state)
    worker.phase = String(phase)
    worker.iteration = iteration
    worker.epoch = epoch
    worker.run_id = String(run_id)
    worker.child_pid = child_pid
    worker.claim = claim
    worker.model_path = model_path
    worker.message = String(message)
    return worker
end

function _worker_log(
    level::Logging.LogLevel,
    message::AbstractString,
    worker::WorkerState;
    waiting::Bool=false,
    next_action::AbstractString="",
    kwargs...,
)
    @logmsg level message _group = DISTRIBUTED_LOG_GROUP role = "worker" worker_id = worker.worker_id state = worker.state phase = worker.phase iteration = worker.iteration epoch = worker.epoch waiting next_action kwargs...
end

function write_worker_heartbeat!(
    bu::Builder,
    worker::WorkerState;
    kwargs...,
)
    _update_worker_state!(worker; kwargs...)
    updated_at = _timestamp()
    payload = Dict{String,Any}(
        "worker_id" => worker.worker_id,
        "run_id" => worker.run_id,
        "builder_uuid" => builder_uuid(bu),
        "host" => get(ENV, "HOSTNAME", get(ENV, "COMPUTERNAME", "")),
        "pid" => getpid(),
        "state" => worker.state,
        "phase" => worker.phase,
        "iteration" => worker.iteration,
        "epoch" => worker.epoch,
        "child_pid" => worker.child_pid === nothing ? "" : string(worker.child_pid),
        "claim" => worker.claim === nothing ? "" : string(worker.claim),
        "model_path" => worker.model_path === nothing ? "" : string(worker.model_path),
        "started_at" => updated_at,
        "updated_at" => updated_at,
        "message" => worker.message,
    )
    _write_json(_worker_state_path(bu, worker.worker_id), payload)
end

function write_worker_heartbeat!(
    bu::Builder,
    worker_id::AbstractString;
    kwargs...,
)
    worker = WorkerState(worker_id; iteration=bu.state.iteration)
    write_worker_heartbeat!(bu, worker; kwargs...)
end

function read_worker_states(bu::Builder)
    dir = _worker_states_dir(bu)
    isdir(dir) || return Dict{String,Any}[]
    states = Dict{String,Any}[]
    for path in sort(glob_allow_abs(joinpath(dir, "*.json")))
        state = _read_json(path)
        state === nothing || push!(states, state)
    end
    return states
end

function _server_publish_control!(
    bu::Builder;
    iteration::Int=bu.state.iteration,
    phase::AbstractString,
    phase_dir::AbstractString,
    status::AbstractString="running",
    epoch::Int=Int(time_ns()),
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
            "updated_at" => _timestamp(),
        ),
        Dict{String,Any}(extra),
    )
    _write_json(_control_path(bu), payload)
    @info "distributed link control published" _group = DISTRIBUTED_LOG_GROUP role = "server" pid = getpid() state = status phase iteration epoch waiting = false next_action = phase == "shutdown" ? "workers_exit" : "workers_poll_control"

    return payload
end

function _read_control(bu::Builder)
    control = _read_json(_control_path(bu))
    control === nothing && return nothing
    get(control, "builder_uuid", "") == builder_uuid(bu) || return nothing
    return control
end

function _server_request_shutdown_impl!(bu::Builder)
    root = _coordination_root(bu)
    _touch_marker(joinpath(root, "SHUTDOWN"))
    phase_dir = joinpath(root, "shutdown")
    @warn "distributed link shutdown requested" _group = DISTRIBUTED_LOG_GROUP role = "server" pid = getpid() state = "SHUTDOWN" phase = "shutdown" iteration = bu.state.iteration waiting = false next_action = "publish_shutdown_control"

    _server_publish_control!(
        bu;
        phase="shutdown",
        phase_dir,
        status="shutdown",
    )
    return true
end

function server_request_shutdown!(bu::Builder)
    _with_distributed_logger(bu; role="server", id=_server_id()) do
        _server_request_shutdown_impl!(bu)
    end
end

shutdown_link_backend!(::DistributedLinkBackend, bu::Builder) =
    bu.options.distributed.enabled ? server_request_shutdown!(bu) : nothing

function _server_clear_shutdown!(bu::Builder)
    path = joinpath(_coordination_root(bu), "SHUTDOWN")
    isfile(path) && rm(path; force=true)
    return true
end

function server_prepare_dft_phase!(bu::Builder, iter::Int=bu.state.iteration)
    state = bu.state
    phase_dir = _phase_dir(bu, iter, "dft")
    crud_workdir = joinpath(state.workdir, bu.options.distributed.dft_crud_workdir, "gen$(iter)")
    input_dir = input_structure_dir(state.workdir, iter)
    output_dir = output_structure_dir(state.workdir, iter)
    input_files = sort(glob_allow_abs(joinpath(input_dir, "*.res")))
    manifest_path = _phase_manifest_path(phase_dir)

    if isfile(joinpath(phase_dir, "DONE")) && is_training_data_ready(bu, iter)
        return crud_workdir
    elseif isfile(joinpath(phase_dir, "READY"))
        manifest = _read_json(manifest_path)
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
        "created_at" => _timestamp(),
    )
    _write_json(manifest_path, manifest)
    _touch_marker(joinpath(phase_dir, "READY"))
    return crud_workdir
end

function _dft_result_counts(crud_workdir)
    good = length(glob_allow_abs(joinpath(crud_workdir, "good_castep", "*.res")))
    bad = length(glob_allow_abs(joinpath(crud_workdir, "bad_castep", "*.res")))
    return good, bad
end

function server_collect_dft_results!(bu::Builder, iter::Int=bu.state.iteration)
    phase_dir = _phase_dir(bu, iter, "dft")
    manifest = _read_json(_phase_manifest_path(phase_dir))
    manifest === nothing && throw(ArgumentError("Missing DFT manifest for generation $iter."))
    crud_workdir = manifest["crud_workdir"]
    output_dir = manifest["output_dir"]
    mkpath(output_dir)
    for file in glob_allow_abs(joinpath(crud_workdir, "good_castep", "*.res"))
        mv(file, joinpath(output_dir, basename(file)); force=true)
    end
    _touch_marker(joinpath(phase_dir, "DONE"))
    return output_dir
end

function _required_dft_terminal(manifest)
    n = Int(manifest["input_count"])
    n == 0 && return 0
    return max(1, ceil(Int, n * Float64(manifest["threshold"])))
end

function _required_dft_success(manifest)
    _required_dft_terminal(manifest)
end

function _max_dft_bad(manifest)
    n = Int(manifest["input_count"])
    max(0, n - _required_dft_success(manifest))
end

function _server_request_phase_stop!(bu::Builder, phase_dir::AbstractString)
    _touch_marker(joinpath(phase_dir, "STOP"))
end

function _server_abort_phase!(bu::Builder, phase_dir::AbstractString, message)
    _touch_marker(joinpath(phase_dir, "STOP"))
    _write_json(
        joinpath(phase_dir, "ABORTED.json"),
        Dict{String,Any}("message" => string(message), "aborted_at" => _timestamp()),
    )
end

function _worker_active_for_control(state, control)
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

function _server_wait_for_workers_stopped!(bu::Builder, control)
    timeout =
        bu.options.distributed.worker_child_grace_seconds +
        bu.options.distributed.worker_child_kill_seconds +
        bu.options.distributed.worker_heartbeat_interval_seconds +
        2 * bu.options.distributed.control_poll_interval_seconds
    started = time()
    while true
        active = filter(state -> _worker_active_for_control(state, control), read_worker_states(bu))
        isempty(active) && return true
        if time() - started >= timeout
            @warn "timed out waiting for distributed link workers to stop" _group = DISTRIBUTED_LOG_GROUP role = "server" pid = getpid() state = "WAIT_WORKERS_STOPPED" phase = string(get(control, "phase", "")) iteration = Int(get(control, "iteration", bu.state.iteration)) epoch = Int(get(control, "epoch", 0)) waiting = true next_action = "continue_after_timeout" active_workers = length(active)

            return false
        end
        sleep(max(1.0, bu.options.distributed.control_poll_interval_seconds))
    end
end

function _server_wait_for_dft!(bu::Builder, phase_dir)
    manifest = _read_json(_phase_manifest_path(phase_dir))
    manifest === nothing && throw(ArgumentError("Missing DFT manifest `$phase_dir`."))
    crud_workdir = manifest["crud_workdir"]
    input_count = Int(manifest["input_count"])
    required_success = _required_dft_success(manifest)
    max_bad = _max_dft_bad(manifest)
    last_diag = 0.0
    while true
        good, bad = _dft_result_counts(crud_workdir)
        if _log_interval_due(last_diag, bu)
            @info "distributed link waiting for DFT results" _group = DISTRIBUTED_LOG_GROUP role = "server" pid = getpid() state = "WAIT_DFT" phase = "dft" iteration = Int(manifest["iteration"]) waiting = true next_action = "wait_for_dft_results" good bad required_success input_count
            last_diag = time()
        end
        if good >= required_success
            @info "distributed link DFT threshold reached" _group = DISTRIBUTED_LOG_GROUP role = "server" pid = getpid() state = "DFT_READY" phase = "dft" iteration = Int(manifest["iteration"]) waiting = false next_action = "request_phase_stop" good bad required_success input_count

            _server_request_phase_stop!(bu, phase_dir)
            return good, bad
        elseif bad > max_bad || good + bad >= input_count
            _touch_marker(joinpath(phase_dir, "FAILED"))
            bad_ratio = input_count == 0 ? 0.0 : bad / input_count
            @error "distributed link DFT failed" _group = DISTRIBUTED_LOG_GROUP role = "server" pid = getpid() state = "DFT_FAILED" phase = "dft" iteration = Int(manifest["iteration"]) waiting = false next_action = "throw" good bad required_success input_count bad_ratio

            throw(
                ErrorException(
                    "Too many failed DFT calculations for generation $(manifest["iteration"]): $bad bad structures out of $input_count total inputs (bad ratio=$(round(100 * bad_ratio; digits=1))%). Need at least $required_success successful calculations, but only $good are available.",
                ),
            )
        end
        if should_stop(bu)
            @warn "STOP requested while waiting for DFT" _group = DISTRIBUTED_LOG_GROUP role = "server" pid = getpid() state = "STOPPING" phase = "dft" iteration = Int(manifest["iteration"]) waiting = false next_action = "abort_phase" good bad required_success input_count

            _server_abort_phase!(bu, phase_dir, "STOP requested while waiting for DFT.")
            return :stopped
        end
        sleep(bu.options.distributed.control_poll_interval_seconds)
    end
end

function _server_run_dft_phase_impl!(bu::Builder, iter::Int=bu.state.iteration)
    if is_training_data_ready(bu, iter)
        @info "distributed link DFT phase skipped" _group = DISTRIBUTED_LOG_GROUP role = "server" pid = getpid() state = "DFT_READY" phase = "dft" iteration = iter waiting = false next_action = "advance_workflow"

        return :ok
    end
    _server_clear_shutdown!(bu)
    phase_dir = _phase_dir(bu, iter, "dft")
    crud_workdir = server_prepare_dft_phase!(bu, iter)
    manifest = _read_json(_phase_manifest_path(phase_dir))
    @info "distributed link DFT phase starting" _group = DISTRIBUTED_LOG_GROUP role = "server" pid = getpid() state = "START_DFT" phase = "dft" iteration = iter waiting = false next_action = "publish_dft_control" crud_workdir input_count = Int(get(manifest, "input_count", 0))

    control = _server_publish_control!(
        bu;
        iteration=iter,
        phase="dft",
        phase_dir,
        extra=Dict("dft" => manifest),
    )
    _server_submit_workers(bu, "dft", iter)
    @info "distributed link DFT phase waiting" _group = DISTRIBUTED_LOG_GROUP role = "server" pid = getpid() state = "WAIT_DFT" phase = "dft" iteration = iter epoch = Int(control["epoch"]) waiting = true next_action = "wait_for_dft_results"

    result = try
        _server_wait_for_dft!(bu, phase_dir)
    catch
        _server_publish_control!(bu; iteration=iter, phase="dft", phase_dir, status="stopping")
        rethrow()
    end
    if result == :stopped
        _server_publish_control!(bu; iteration=iter, phase="dft", phase_dir, status="stopping")
        _server_wait_for_workers_stopped!(bu, control)
        return :stopped
    end
    _server_wait_for_workers_stopped!(bu, control)
    server_collect_dft_results!(bu, iter)
    _server_publish_control!(bu; iteration=iter, phase="idle", phase_dir, status="done")
    @info "distributed link DFT phase complete" _group = DISTRIBUTED_LOG_GROUP role = "server" pid = getpid() state = "DFT_DONE" phase = "dft" iteration = iter epoch = Int(control["epoch"]) waiting = false next_action = "advance_workflow"

    return :ok
end

function server_run_dft_phase!(bu::Builder, iter::Int=bu.state.iteration)
    _with_distributed_logger(bu; role="server", id=_server_id()) do
        _server_run_dft_phase_impl!(bu, iter)
    end
end

function _training_phase_dir(bu::Builder, iter=bu.state.iteration)
    _phase_dir(bu, iter, "train")
end

function _training_slots_dir(bu::Builder, iter=bu.state.iteration)
    joinpath(_training_phase_dir(bu, iter), "slots")
end

function _training_slot_name(slot_id::Int)
    lpad(string(slot_id), 6, '0')
end

function _training_manifest_matches(bu::Builder, manifest, iter::Int, dataset_path)
    manifest === nothing && return false
    Int(get(manifest, "iteration", -1)) == iter || return false
    get(manifest, "dataset_path", "") == dataset_path || return false
    Int(get(manifest, "nmodels", -1)) == bu.trainer.nmodels || return false
    Int(get(manifest, "required_models", -1)) == _external_required_models(bu.trainer) || return false
    get(manifest, "prefix", "") == bu.trainer.prefix || return false
    return true
end

function _server_clear_training_phase!(bu::Builder, phase_dir, iter::Int)
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

function server_prepare_training_phase!(bu::Builder, iter::Int=bu.state.iteration)
    phase_dir = _training_phase_dir(bu, iter)
    dataset_path = dataset_name(bu, iter)
    manifest_path = _phase_manifest_path(phase_dir)
    manifest = _read_json(manifest_path)
    if (isfile(joinpath(phase_dir, "READY")) || isfile(joinpath(phase_dir, "DONE"))) &&
       _training_manifest_matches(bu, manifest, iter, dataset_path)
        return phase_dir
    end

    mkpath(joinpath(phase_dir, "slots"))
    mkpath(_phase_events_dir(phase_dir))
    _server_clear_training_phase!(bu, phase_dir, iter)
    mkpath(joinpath(phase_dir, "slots"))
    mkpath(_phase_events_dir(phase_dir))
    manifest = Dict{String,Any}(
        "phase" => "training",
        "iteration" => iter,
        "dataset_path" => dataset_path,
        "training_dir" => joinpath(bu.state.workdir, TRAINING_DIR),
        "nmodels" => bu.trainer.nmodels,
        "required_models" => _external_required_models(bu.trainer),
        "prefix" => bu.trainer.prefix,
        "created_at" => _timestamp(),
    )
    _write_json(manifest_path, manifest)
    _touch_marker(joinpath(phase_dir, "READY"))
    return phase_dir
end

function training_slot_model_path(bu::Builder, iter::Int, slot_id::Int)
    slot = _training_slot_name(slot_id)
    joinpath(
        bu.state.workdir,
        TRAINING_DIR,
        "$(bu.trainer.prefix)gen-$(iter)-model-slot-$(slot).json",
    )
end

function worker_claim_training_slot!(bu::Builder, iter::Int, worker_id::AbstractString)
    slots_dir = _training_slots_dir(bu, iter)
    mkpath(slots_dir)
    now = time()
    for slot_id = 1:bu.trainer.nmodels
        slot = _training_slot_name(slot_id)
        done_path = joinpath(slots_dir, "$(slot).done.json")
        failed_path = joinpath(slots_dir, "$(slot).failed.json")
        lock_path = joinpath(slots_dir, "$(slot).lock")
        isfile(done_path) && continue
        if isdir(lock_path)
            age = now - stat(lock_path).mtime
            if age > bu.options.distributed.training_slot_stale_seconds
                rm(lock_path; recursive=true, force=true)
            else
                continue
            end
        end
        try
            isfile(failed_path) && rm(failed_path; force=true)
            mkdir(lock_path)
            model_path = training_slot_model_path(bu, iter, slot_id)
            claim = Dict{String,Any}(
                "slot_id" => slot_id,
                "worker_id" => worker_id,
                "host" => get(ENV, "HOSTNAME", ""),
                "pid" => getpid(),
                "model_path" => model_path,
                "claimed_at" => _timestamp(),
            )
            _write_json(joinpath(lock_path, "claim.json"), claim)
            return slot_id, lock_path, model_path
        catch err
            (err isa Base.IOError && err.code == Base.UV_EEXIST) || rethrow()
        end
    end
    return nothing
end

function worker_mark_training_slot_done!(bu::Builder, iter::Int, slot_id::Int, model_path)
    slots_dir = _training_slots_dir(bu, iter)
    slot = _training_slot_name(slot_id)
    payload = Dict{String,Any}(
        "slot_id" => slot_id,
        "model_path" => model_path,
        "done_at" => _timestamp(),
    )
    _write_json(joinpath(slots_dir, "$(slot).done.json"), payload)
end

function worker_mark_training_slot_failed!(bu::Builder, iter::Int, slot_id::Int, message)
    slots_dir = _training_slots_dir(bu, iter)
    slot = _training_slot_name(slot_id)
    lock_path = joinpath(slots_dir, "$(slot).lock")
    payload = Dict{String,Any}(
        "slot_id" => slot_id,
        "message" => message,
        "failed_at" => _timestamp(),
    )
    _write_json(joinpath(slots_dir, "$(slot).failed.json"), payload)
    isdir(lock_path) && rm(lock_path; recursive=true, force=true)
end

function _server_wait_for_training!(bu::Builder, phase_dir)
    manifest = _read_json(_phase_manifest_path(phase_dir))
    manifest === nothing && throw(ArgumentError("Missing training manifest `$phase_dir`."))
    required = Int(manifest["required_models"])
    last_diag = 0.0
    while true
        nm = num_existing_models(bu)
        if _log_interval_due(last_diag, bu)
            @info "distributed link waiting for training models" _group = DISTRIBUTED_LOG_GROUP role = "server" pid = getpid() state = "WAIT_TRAINING" phase = "training" iteration = Int(manifest["iteration"]) waiting = true next_action = "wait_for_models" models_found = nm required_models = required
            last_diag = time()
        end
        if nm >= required
            @info "distributed link training threshold reached" _group = DISTRIBUTED_LOG_GROUP role = "server" pid = getpid() state = "TRAINING_READY" phase = "training" iteration = Int(manifest["iteration"]) waiting = false next_action = "request_phase_stop" models_found = nm required_models = required

            _server_request_phase_stop!(bu, phase_dir)
            return nm
        end
        if should_stop(bu)
            @warn "STOP requested while waiting for training" _group = DISTRIBUTED_LOG_GROUP role = "server" pid = getpid() state = "STOPPING" phase = "training" iteration = Int(manifest["iteration"]) waiting = false next_action = "abort_phase" models_found = nm required_models = required

            _server_abort_phase!(bu, phase_dir, "STOP requested while waiting for training.")
            return :stopped
        end
        sleep(bu.options.distributed.control_poll_interval_seconds)
    end
end

function _server_run_training_phase_impl!(bu::Builder, iter::Int=bu.state.iteration)
    if _models_ready_for_ensemble(bu)
        @info "distributed link training phase skipped" _group = DISTRIBUTED_LOG_GROUP role = "server" pid = getpid() state = "TRAINING_READY" phase = "training" iteration = iter waiting = false next_action = "create_ensemble" models_found = num_existing_models(bu) required_models = _external_required_models(bu.trainer)

        return :ok
    end
    _server_clear_shutdown!(bu)
    phase_dir = server_prepare_training_phase!(bu, iter)
    manifest = _read_json(_phase_manifest_path(phase_dir))
    @info "distributed link training phase starting" _group = DISTRIBUTED_LOG_GROUP role = "server" pid = getpid() state = "START_TRAINING" phase = "training" iteration = iter waiting = false next_action = "publish_training_control" required_models = Int(get(manifest, "required_models", 0))

    control = _server_publish_control!(
        bu;
        iteration=iter,
        phase="training",
        phase_dir,
        extra=Dict("training" => manifest),
    )
    _server_submit_workers(bu, "training", iter)
    @info "distributed link training phase waiting" _group = DISTRIBUTED_LOG_GROUP role = "server" pid = getpid() state = "WAIT_TRAINING" phase = "training" iteration = iter epoch = Int(control["epoch"]) waiting = true next_action = "wait_for_models"

    result = try
        _server_wait_for_training!(bu, phase_dir)
    catch err
        _server_publish_control!(bu; iteration=iter, phase="training", phase_dir, status="stopping")
        rethrow()
    end
    if result == :stopped
        _server_publish_control!(bu; iteration=iter, phase="training", phase_dir, status="stopping")
        _server_wait_for_workers_stopped!(bu, control)
        return :stopped
    end
    _server_wait_for_workers_stopped!(bu, control)
    _touch_marker(joinpath(phase_dir, "DONE"))
    _server_publish_control!(bu; iteration=iter, phase="idle", phase_dir, status="done")
    nm = num_existing_models(bu)
    @info "distributed link training phase complete" _group = DISTRIBUTED_LOG_GROUP role = "server" pid = getpid() state = "TRAINING_DONE" phase = "training" iteration = iter epoch = Int(control["epoch"]) waiting = false next_action = "create_ensemble" models_found = nm required_models = _external_required_models(bu.trainer)

    nm >= _external_required_models(bu.trainer) && return :ok
    @error "distributed link training phase failed" _group = DISTRIBUTED_LOG_GROUP role = "server" pid = getpid() state = "TRAINING_FAILED" phase = "training" iteration = iter waiting = false next_action = "throw" models_found = nm required_models = _external_required_models(bu.trainer)

    throw(ErrorException("Only $nm models are found in the training directory."))
end

function server_run_training_phase!(bu::Builder, iter::Int=bu.state.iteration)
    _with_distributed_logger(bu; role="server", id=_server_id()) do
        _server_run_training_phase_impl!(bu, iter)
    end
end

function run_dft_phase!(::DistributedLinkBackend, bu::Builder, iter::Int)
    result = server_run_dft_phase!(bu, iter)
    result == :stopped && return LinkPhaseResult(:stopped, :dft, iter)
    return LinkPhaseResult(:ok, :dft, iter)
end

function run_training_phase!(::DistributedLinkBackend, bu::Builder, iter::Int)
    result = server_run_training_phase!(bu, iter)
    result == :stopped && return LinkPhaseResult(:stopped, :train_models, iter)
    return LinkPhaseResult(:ok, :train_models, iter)
end

function run_link_workflow!(builder::Builder, backend::DistributedLinkBackend)
    _with_distributed_logger(builder; role="server", id=_server_id()) do
        invoke(run_link_workflow!, Tuple{Builder,AbstractLinkBackend}, builder, backend)
    end
end

function _worker_julia_code()
    "using EDDPotentials; EDDPotentials.distributed_worker()"
end

function _worker_base_cmd(bu::Builder)
    project_path = dirname(Base.active_project())
    builder_file = _require_builder_file_path!(bu)
    Cmd([
        Base.julia_cmd()...,
        "--project=$(project_path)",
        "-e",
        _worker_julia_code(),
        "--",
        "--file",
        builder_file,
    ])
end

function _worker_job_payload(bu::Builder)
    worker_id_line = "WORKER_ID=\"\${SLURM_ARRAY_TASK_ID:-\${SGE_TASK_ID:-1}}\""
    worker_cmd = _command_string(_worker_base_cmd(bu)) * " --worker-id \"\$WORKER_ID\""
    worker_id_line * "\n" * worker_cmd
end

function _render_distributed_worker_job_script(bu::Builder, scheduler::SchedulerConfig, workdir)
    mkpath(workdir)
    template = get_job_script_content(scheduler)
    if !contains(template, DISTRIBUTED_WORKER_CMD_PLACEHOLDER)
        throw(
            ArgumentError(
                "Distributed link scheduler script `$(scheduler.script)` must contain the placeholder `$DISTRIBUTED_WORKER_CMD_PLACEHOLDER`.",
            ),
        )
    end
    rendered = replace(template, DISTRIBUTED_WORKER_CMD_PLACEHOLDER => _worker_job_payload(bu))
    script_path = joinpath(workdir, "job.sh")
    _atomic_write(script_path, rendered)
    return script_path
end

function _server_submit_workers(bu::Builder, phase::AbstractString, iter::Int)
    scheduler = bu.options.scheduler
    scheduler === nothing && return ""
    workdir = joinpath(_coordination_root(bu), "submit", "gen$(iter)-$(phase)")
    _render_distributed_worker_job_script(bu, scheduler, workdir)
    bu.options.distributed.manage_worker_jobs || return ""
    scheduler.submit_jobs || return ""
    nworkers = if phase == "dft"
        bu.options.distributed.dft_worker_jobs
    elseif phase == "training"
        bu.options.distributed.training_worker_jobs
    else
        scheduler.njobs
    end
    distributed_scheduler = SchedulerConfig(
        scheduler.script,
        max(1, nworkers),
        scheduler.type,
        scheduler.clean_workdir,
        scheduler.submit_jobs,
    )
    cmd = _training_submit_cmd(distributed_scheduler, workdir)
    @info "Distributed link worker submission command: $cmd"
    output = readchomp(cmd)
    !isempty(output) && @info "Distributed link worker submission output: $output"
    return _parse_scheduler_job_id(scheduler, output)
end

function _worker_stop_requested(bu::Builder, control, phase_dir, worker_id)
    should_stop(bu) && return true
    isfile(joinpath(_coordination_root(bu), "stop")) && return true
    isfile(joinpath(phase_dir, "STOP")) && return true
    isfile(joinpath(phase_dir, "workers", "$(worker_id).STOP")) && return true
    current = _read_control(bu)
    current === nothing && return false
    get(current, "status", "running") in ("stop", "stopping", "done", "error") && return true
    get(current, "epoch", get(control, "epoch", 0)) != get(control, "epoch", 0) && return true
    return false
end

function _process_pid(proc)
    try
        return getpid(proc)
    catch
        return ""
    end
end

function _signal_process(proc, signal)
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

function _worker_stop_child!(proc, distributed::DistributedLinkOption)
    process_running(proc) || return
    _signal_process(proc, Base.SIGTERM)
    started = time()
    while process_running(proc) && time() - started < distributed.worker_child_grace_seconds
        sleep(1)
    end
    process_running(proc) || return
    _signal_process(proc, Base.SIGKILL)
    started = time()
    while process_running(proc) && time() - started < distributed.worker_child_kill_seconds
        sleep(1)
    end
end

function _worker_supervise_child!(
    bu::Builder,
    worker::WorkerState,
    control,
    phase_dir,
    cmd::Cmd;
    state,
    claim=nothing,
    model_path=nothing,
    slot_id=nothing,
)
    started = time()
    proc = run(cmd; wait=false)
    child_pid = _process_pid(proc)
    phase = string(get(control, "phase", ""))
    iteration = Int(get(control, "iteration", bu.state.iteration))
    epoch = Int(get(control, "epoch", 0))
    run_id = string(get(control, "run_id", ""))
    _update_worker_state!(
        worker;
        state,
        phase,
        iteration,
        epoch,
        run_id,
        child_pid,
        claim,
        model_path,
    )
    _worker_log(
        Logging.Info,
        "distributed link child launched",
        worker;
        next_action="supervise_child",
        child_pid,
        command=string(cmd),
        slot_id,
        model_path,
    )

    last_heartbeat = 0.0
    while process_running(proc)
        if time() - last_heartbeat >= bu.options.distributed.worker_heartbeat_interval_seconds
            write_worker_heartbeat!(bu, worker; message="child running")
            runtime_seconds = round(time() - started; digits=3)
            _worker_log(
                Logging.Debug,
                "distributed link child still running",
                worker;
                waiting=true,
                next_action="supervise_child",
                child_pid,
                runtime_seconds,
                slot_id,
                model_path,
            )

            last_heartbeat = time()
        end
        if _worker_stop_requested(bu, control, phase_dir, worker.worker_id)
            write_worker_heartbeat!(bu, worker; state="STOPPING_CHILD", message="stop requested")
            runtime_seconds = round(time() - started; digits=3)
            _worker_log(
                Logging.Warn,
                "distributed link child stop requested",
                worker;
                next_action="terminate_child",
                child_pid,
                runtime_seconds,
                reason="stop_requested",
                slot_id,
                model_path,
            )

            _worker_stop_child!(proc, bu.options.distributed)
            return false
        end
        sleep(max(1.0, bu.options.distributed.control_poll_interval_seconds))
    end
    wait(proc)
    ok = success(proc)
    runtime_seconds = round(time() - started; digits=3)
    if ok
        _worker_log(
            Logging.Info,
            "distributed link child finished",
            worker;
            next_action="mark_child_result",
            child_pid,
            runtime_seconds,
            success=true,
            slot_id,
            model_path,
        )
    else
        _worker_log(
            Logging.Error,
            "distributed link child failed",
            worker;
            next_action="mark_child_failed",
            child_pid,
            runtime_seconds,
            success=false,
            reason="process_failed",
            slot_id,
            model_path,
        )
    end
    return ok
end

function _worker_run_dft!(bu::Builder, worker::WorkerState, control)
    phase_dir = control["phase_dir"]
    manifest = _read_json(_phase_manifest_path(phase_dir))
    if manifest === nothing
        _update_worker_state!(
            worker;
            state="DFT_FAILED",
            phase="dft",
            iteration=Int(control["iteration"]),
            epoch=Int(control["epoch"]),
            run_id=string(control["run_id"]),
            child_pid=nothing,
            claim=nothing,
            model_path=nothing,
        )
        _worker_log(Logging.Error, "missing DFT manifest", worker; next_action="throw", phase_dir)

        throw(ArgumentError("Missing DFT manifest `$phase_dir`."))
    end
    cmd = Cmd(`crud.pl -mpinp $(Int(manifest["mpinp"]))`, dir=manifest["crud_workdir"])
    _update_worker_state!(
        worker;
        state="DFT_RUNNING",
        phase="dft",
        iteration=Int(control["iteration"]),
        epoch=Int(control["epoch"]),
        run_id=string(control["run_id"]),
        child_pid=nothing,
        claim=nothing,
        model_path=nothing,
    )
    _worker_log(Logging.Info, "distributed link DFT child starting", worker; next_action="launch_crud", crud_workdir=manifest["crud_workdir"])

    ok = _worker_supervise_child!(bu, worker, control, phase_dir, cmd; state="DFT_RUNNING")
    write_worker_heartbeat!(
        bu,
        worker;
        state=ok ? "DFT_DONE" : "STOPPED",
        phase="dft",
        iteration=Int(control["iteration"]),
        epoch=Int(control["epoch"]),
        message=ok ? "crud.pl finished" : "crud.pl stopped",
        run_id=string(control["run_id"]),
        child_pid=nothing,
        claim=nothing,
        model_path=nothing,
    )
    _write_phase_event(phase_dir, worker.worker_id, ok ? "dft-done" : "dft-stopped")
    _worker_log(Logging.Info, "distributed link DFT child result", worker; next_action="poll_control", success=ok)

    return ok
end

function _train_slot_base_cmd(bu::Builder, slot_id::Int, model_path::AbstractString)
    project_path = dirname(Base.active_project())
    builder_file = _require_builder_file_path!(bu)
    Cmd([
        Base.julia_cmd()...,
        "--project=$(project_path)",
        "-e",
        "using EDDPotentials; EDDPotentials.distributed_train_slot()",
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

function _worker_run_training!(bu::Builder, worker::WorkerState, control)
    iter = Int(control["iteration"])
    phase_dir = control["phase_dir"]
    claim = worker_claim_training_slot!(bu, iter, worker.worker_id)
    if claim === nothing
        write_worker_heartbeat!(
            bu,
            worker;
            state="EXIT",
            phase="training",
            iteration=iter,
            epoch=Int(control["epoch"]),
            message="no training slot available",
            run_id=string(control["run_id"]),
            child_pid=nothing,
            claim=nothing,
            model_path=nothing,
        )
        _worker_log(Logging.Info, "distributed link no training slot available", worker; next_action="poll_control")
        return true
    end
    slot_id, claim_path, model_path = claim
    write_worker_heartbeat!(
        bu,
        worker;
        state="TRAINING_CLAIM",
        phase="training",
        iteration=iter,
        epoch=Int(control["epoch"]),
        claim=claim_path,
        model_path,
        message="claimed training slot",
        run_id=string(control["run_id"]),
        child_pid=nothing,
    )
    _worker_log(Logging.Info, "distributed link training slot claimed", worker; next_action="launch_training_slot", slot_id, claim_path, model_path)
    cmd = _train_slot_base_cmd(bu, slot_id, model_path)
    _update_worker_state!(worker; state="TRAINING_RUNNING", message="")
    _worker_log(Logging.Info, "distributed link training child starting", worker; next_action="supervise_child", slot_id, model_path)

    ok = _worker_supervise_child!(
        bu,
        worker,
        control,
        phase_dir,
        cmd;
        state="TRAINING_RUNNING",
        claim=claim_path,
        model_path,
        slot_id,
    )
    if ok && isfile(model_path)
        worker_mark_training_slot_done!(bu, iter, slot_id, model_path)
        _write_phase_event(phase_dir, worker.worker_id, "train-done", Dict("slot_id" => slot_id))
        _update_worker_state!(worker; state="TRAINING_DONE", message="")
        _worker_log(Logging.Info, "distributed link training slot done", worker; next_action="poll_control", slot_id, model_path)
    else
        worker_mark_training_slot_failed!(bu, iter, slot_id, ok ? "model artifact missing" : "training stopped")
        _write_phase_event(phase_dir, worker.worker_id, "train-failed", Dict("slot_id" => slot_id))
        _update_worker_state!(worker; state="TRAINING_FAILED", message="")
        _worker_log(Logging.Error, "distributed link training slot failed", worker; next_action="mark_slot_failed", slot_id, model_path, success=ok)
    end
    write_worker_heartbeat!(
        bu,
        worker;
        state=ok ? "TRAINING_DONE" : "STOPPED",
        phase="training",
        iteration=iter,
        epoch=Int(control["epoch"]),
        claim=claim_path,
        model_path,
        message=ok ? "training slot finished" : "training slot stopped",
        run_id=string(control["run_id"]),
        child_pid=nothing,
    )
    return ok
end

function _worker_shutdown_requested(bu::Builder)
    should_stop(bu) && return true
    isfile(joinpath(_coordination_root(bu), "stop")) && return true
    isfile(joinpath(_coordination_root(bu), "SHUTDOWN")) && return true
    control = _read_control(bu)
    control === nothing && return false
    string(get(control, "phase", "")) == "shutdown" && return true
    string(get(control, "status", "")) == "shutdown" && return true
    return false
end

function _worker_handle_control!(
    bu::Builder,
    worker::WorkerState,
    control,
    handled_dft_controls::Set{Tuple{Int,Int}},
)
    bu.state.iteration = Int(control["iteration"])
    phase = string(control["phase"])
    epoch = Int(control["epoch"])
    control_key = (bu.state.iteration, epoch)
    write_worker_heartbeat!(
        bu,
        worker;
        state="WAIT",
        phase,
        iteration=bu.state.iteration,
        epoch,
        message="control loaded",
        run_id=string(control["run_id"]),
        child_pid=nothing,
        claim=nothing,
        model_path=nothing,
    )
    _worker_log(
        Logging.Info,
        "distributed link control loaded",
        worker;
        next_action=phase == "dft" ? "run_dft_worker" : phase == "training" ? "claim_training_slot" : phase == "shutdown" ? "exit" : "idle",
    )
    if phase == "dft"
        if control_key in handled_dft_controls
            write_worker_heartbeat!(bu, worker; state="IDLE", message="DFT control already handled")
            _worker_log(Logging.Debug, "distributed link DFT control already handled", worker; waiting=true, next_action="poll_control")
        end
        control_key in handled_dft_controls && return true
        ok = _worker_run_dft!(bu, worker, control)
        push!(handled_dft_controls, control_key)
        return ok
    elseif phase == "training"
        return _worker_run_training!(bu, worker, control)
    elseif phase == "shutdown"
        write_worker_heartbeat!(bu, worker; state="EXIT", message="shutdown control received")
        _worker_log(Logging.Info, "distributed link shutdown control received", worker; next_action="exit")

        return :shutdown
    else
        write_worker_heartbeat!(
            bu,
            worker;
            state="IDLE",
            phase,
            iteration=bu.state.iteration,
            epoch=Int(get(control, "epoch", 0)),
            message="no runnable phase",
            run_id=string(get(control, "run_id", "")),
        )
        _worker_log(Logging.Warn, "distributed link no runnable phase", worker; waiting=true, next_action="poll_control")
        return true
    end
end

function distributed_worker(file::AbstractString; worker_id::AbstractString="", once::Bool=false)
    bu = Builder(file)
    isempty(worker_id) && (worker_id = get(ENV, "WORKER_ID", string(getpid())))
    _with_distributed_logger(bu; role="worker", id=worker_id) do
        worker = WorkerState(worker_id; state="BOOT", iteration=bu.state.iteration)
        _worker_log(Logging.Info, "distributed link worker boot", worker; next_action="poll_control")

        write_worker_heartbeat!(bu, worker; message="worker boot")
        handled_dft_controls = Set{Tuple{Int,Int}}()
        last_idle_diag = 0.0
        while true
            if _worker_shutdown_requested(bu)
                write_worker_heartbeat!(
                    bu,
                    worker;
                    state="EXIT",
                    phase="shutdown",
                    iteration=bu.state.iteration,
                    message="shutdown requested",
                    child_pid=nothing,
                    claim=nothing,
                    model_path=nothing,
                )
                _worker_log(Logging.Info, "distributed link worker shutdown exit", worker; next_action="exit")

                return true
            end
            control = _read_control(bu)
            if control === nothing
                write_worker_heartbeat!(
                    bu,
                    worker;
                    state="IDLE",
                    phase="",
                    iteration=bu.state.iteration,
                    epoch=0,
                    message="no active control",
                    run_id="",
                    child_pid=nothing,
                    claim=nothing,
                    model_path=nothing,
                )
                if once || _log_interval_due(last_idle_diag, bu)
                    _worker_log(Logging.Info, "distributed link worker idle", worker; waiting=true, next_action="poll_control")
                    last_idle_diag = time()
                end
                once && return false
                sleep(max(1.0, bu.options.distributed.worker_poll_interval_seconds))
                continue
            end
            result = _worker_handle_control!(bu, worker, control, handled_dft_controls)
            result == :shutdown && return true
            once && return result == true
            sleep(max(1.0, bu.options.distributed.worker_poll_interval_seconds))
        end
    end
end

function distributed_worker()
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
    distributed_worker(args["file"]; worker_id=args["worker-id"], once=args["once"])
end

function distributed_train_slot()
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

function link_distributed(bu::Builder)
    _with_distributed_logger(bu; role="server", id=_server_id()) do
        @info "distributed link server starting" _group = DISTRIBUTED_LOG_GROUP role = "server" pid = getpid() state = "START" phase = "" iteration = bu.state.iteration waiting = false next_action = "run_link_workflow"

        run_link_workflow!(bu, DistributedLinkBackend())
    end
end

function link_distributed(file::AbstractString="link.toml")
    bu = Builder(file)
    link_distributed(bu)
end
