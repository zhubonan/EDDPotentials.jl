using EDDPotentials
using Test

function _distributed_builder(
    temp;
    state=EDDPotentials.BuilderState(
        seedfile="seed.cell",
        seedfile_calc="calc.cell",
        workdir=temp,
    ),
    cf=EDDPotentials.CellFeatureConfig(elements=["H"]),
    rss=EDDPotentials.RssSetting(),
    trainer=EDDPotentials.TrainingOption(),
    scheduler=nothing,
    distributed=EDDPotentials.DistributedLinkOption(),
)
    distributed.enabled = true
    Builder(EDDPotentials.BuilderOption(state, cf, nothing, rss, trainer, scheduler, distributed))
end

function _write_minimal_castep_inputs(temp)
    open(joinpath(temp, "calc.cell"), "w") do io
        println(io, "%BLOCK LATTICE_CART")
        println(io, "1 0 0")
        println(io, "0 1 0")
        println(io, "0 0 1")
        println(io, "%ENDBLOCK LATTICE_CART")
    end
    write(joinpath(temp, "calc.param"), "task : SinglePoint\n")
end

function _write_generated_res(temp, names; iter=0)
    gen_dir = EDDPotentials.input_structure_dir(temp, iter)
    mkpath(gen_dir)
    for name in names
        write(joinpath(gen_dir, name), "TITL test\nEND\n")
    end
    gen_dir
end

@testset "Distributed link" begin

    @testset "File protocol helpers" begin
        mktempdir() do temp
            state = EDDPotentials.BuilderState(
                seedfile="seed.cell",
                seedfile_calc="calc.cell",
                workdir=temp,
            )
            state.mpinp = 8
            state.per_generation_threshold = 0.5
            trainer = EDDPotentials.TrainingOption()
            trainer.nmodels = 3
            distributed = EDDPotentials.DistributedLinkOption()
            distributed.worker_heartbeat_interval_seconds = 0.01
            distributed.dft_env = Dict{String,Any}("OMP_NUM_THREADS" => "1")
            distributed.training_env = Dict{String,Any}("OMP_NUM_THREADS" => 16)
            builder = _distributed_builder(temp; state, trainer, distributed)
            builder.state.builder_file_path = joinpath(temp, "link.toml")
            write(builder.state.builder_file_path, "")

            _write_minimal_castep_inputs(temp)
            _write_generated_res(temp, ["calc-001.res", "calc-002.res"])

            crud_workdir = EDDPotentials.server_prepare_dft_phase!(builder, 0)
            dft_cmd = EDDPotentials._distributed_dft_cmd(
                builder,
                Dict("mpinp" => 8, "crud_workdir" => crud_workdir),
            )
            @test "OMP_NUM_THREADS=1" in getfield(dft_cmd, :env)
            @test isdir(joinpath(crud_workdir, "hopper"))
            @test isdir(joinpath(crud_workdir, "good_castep"))
            @test isdir(joinpath(crud_workdir, "bad_castep"))
            @test isfile(joinpath(crud_workdir, "calc.cell"))
            @test isfile(joinpath(crud_workdir, "calc.param"))
            @test count(endswith(".res"), readdir(joinpath(crud_workdir, "hopper"))) == 2

            snapshot_worker = EDDPotentials.WorkerState(
                "worker-1";
                state="DFT_RUNNING",
                phase="dft",
                iteration=0,
                epoch=1,
                child_pid=123,
                message="testing",
            )
            EDDPotentials.write_worker_heartbeat!(builder, snapshot_worker)
            states = EDDPotentials.read_worker_states(builder)
            @test length(states) == 1
            @test states[1]["worker_id"] == "worker-1"
            @test states[1]["state"] == "DFT_RUNNING"
            @test_throws ArgumentError EDDPotentials.WorkerState(
                "worker-bad-state";
                state="NOT_A_STATE",
                phase="dft",
            )
            @test_throws ArgumentError EDDPotentials.WorkerState(
                "worker-bad-phase";
                state="DFT_RUNNING",
                phase="training",
            )

            transition_worker = EDDPotentials.WorkerState("worker-transition"; iteration=0)
            EDDPotentials.advance_worker!(
                builder,
                transition_worker,
                EDDPotentials.WORKER_WAIT;
                phase="dft",
                iteration=0,
                epoch=2,
                run_id="transition-test",
            )
            EDDPotentials.advance_worker!(
                builder,
                transition_worker,
                EDDPotentials.WORKER_DFT_RUNNING;
                phase="dft",
            )
            EDDPotentials.advance_worker!(
                builder,
                transition_worker,
                EDDPotentials.WORKER_DFT_DONE;
                phase="dft",
            )
            @test transition_worker.state == "DFT_DONE"
            EDDPotentials.write_worker_heartbeat!(
                builder,
                transition_worker;
                message="refresh only",
            )
            @test transition_worker.state == "DFT_DONE"
            @test transition_worker.message == "refresh only"
            @test_throws MethodError EDDPotentials.write_worker_heartbeat!(
                builder,
                transition_worker;
                state="IDLE",
            )
            @test_throws ArgumentError EDDPotentials.advance_worker!(
                builder,
                transition_worker,
                EDDPotentials.WORKER_DFT_RUNNING;
                phase="dft",
            )

            training_phase = EDDPotentials.server_prepare_training_phase!(builder, 0)
            @test isfile(joinpath(training_phase, "READY"))
            @test isfile(joinpath(training_phase, "manifest.json"))
            @test isdir(joinpath(training_phase, "events"))
            @test !isdir(joinpath(training_phase, "slots"))
            touch(joinpath(training_phase, "STOP"))
            mkpath(joinpath(training_phase, "slots"))
            @test EDDPotentials.server_prepare_training_phase!(builder, 0) == training_phase
            @test !isdir(joinpath(training_phase, "slots"))
            @test !isfile(joinpath(training_phase, "STOP"))
            trainer.num_threads_per_worker = 3
            trainer.cuda_visible_devices = "0,1"
            train_cmd = EDDPotentials._distributed_training_cmd(
                builder,
                0,
                "worker-1",
                training_phase,
            )
            train_cmd_text = join(string.(collect(train_cmd)), " ")
            @test contains(train_cmd_text, "EDDPotentials.run_trainer()")
            @test contains(train_cmd_text, "--stop-file")
            @test !contains(train_cmd_text, "--slot-id")
            train_cmd_env = getfield(train_cmd, :env)
            @test "JULIA_NUM_THREADS=3" in train_cmd_env
            @test "OMP_NUM_THREADS=16" in train_cmd_env
            @test "EDDP_CUDA_DEVICE=0" in train_cmd_env
            distributed.training_env = Dict{String,Any}()
            train_cmd_without_omp = EDDPotentials._distributed_training_cmd(
                builder,
                0,
                "worker-1",
                training_phase,
            )
            train_cmd_without_omp_env = getfield(train_cmd_without_omp, :env)
            @test "JULIA_NUM_THREADS=3" in train_cmd_without_omp_env
            @test !any(startswith("OMP_NUM_THREADS="), train_cmd_without_omp_env)
            model_path = joinpath(temp, EDDPotentials.TRAINING_DIR, "gen-0-model-abcd1234.json")
            mkpath(dirname(model_path))
            write(model_path, "{}")
            trainer.nmodels = 4
            EDDPotentials.server_prepare_training_phase!(builder, 0)
            @test !isfile(model_path)
        end
    end

    @testset "Logging" begin
        mktempdir() do temp
            distributed = EDDPotentials.DistributedLinkOption()
            distributed.log_enabled = true
            distributed.log_level = "info"
            distributed.log_interval_seconds = 0.0
            builder = _distributed_builder(temp; distributed)

            log_path = EDDPotentials._distributed_log_path(builder; role="worker", id="worker-info")
            err_text = mktemp() do _, err_io
                redirect_stderr(err_io) do
                    EDDPotentials._with_distributed_logger(builder; role="worker", id="worker-info") do
                        @debug "hidden debug log" _group=EDDPotentials.DISTRIBUTED_LOG_GROUP role="worker"
                        @info "visible info log" _group=EDDPotentials.DISTRIBUTED_LOG_GROUP role="worker" state="IDLE" next_action="poll_control"
                    end
                end
                seekstart(err_io)
                read(err_io, String)
            end
            log_text = read(log_path, String)
            @test contains(log_text, "visible info log")
            @test contains(err_text, "visible info log")
            @test contains(err_text, "timestamp=")
            @test contains(err_text, "level=INFO")
            @test !contains(log_text, "hidden debug log")
            @test contains(log_text, "timestamp=")
            @test contains(log_text, "level=INFO")
            @test contains(log_text, "role=worker")
            @test contains(log_text, "id=worker-info")
            @test contains(log_text, "message=\"visible info log\"")
            @test contains(log_text, "state=IDLE")
            @test contains(log_text, "next_action=poll_control")

            builder.options.distributed.log_level = "warn"
            warn_path = EDDPotentials._distributed_log_path(builder; role="worker", id="worker-warn")
            redirect_stderr(devnull) do
                EDDPotentials._with_distributed_logger(builder; role="worker", id="worker-warn") do
                    @info "suppressed info log" _group=EDDPotentials.DISTRIBUTED_LOG_GROUP role="worker"
                    @warn "visible warn log" _group=EDDPotentials.DISTRIBUTED_LOG_GROUP role="worker" state="STOPPING_CHILD"
                end
            end
            warn_text = read(warn_path, String)
            @test contains(warn_text, "visible warn log")
            @test contains(warn_text, "level=WARN")
            @test !contains(warn_text, "suppressed info log")

            builder.options.distributed.log_level = "debug"
            debug_path = EDDPotentials._distributed_log_path(builder; role="worker", id="worker-debug")
            redirect_stderr(devnull) do
                EDDPotentials._with_distributed_logger(builder; role="worker", id="worker-debug") do
                    @debug "visible debug log" _group=EDDPotentials.DISTRIBUTED_LOG_GROUP role="worker" state="IDLE"
                end
            end
            debug_text = read(debug_path, String)
            @test contains(debug_text, "visible debug log")
            @test contains(debug_text, "level=DEBUG")

            builder.options.distributed.log_enabled = false
            disabled_path = EDDPotentials._distributed_log_path(builder; role="worker", id="worker-disabled")
            disabled_err = mktemp() do _, err_io
                redirect_stderr(err_io) do
                    EDDPotentials._with_distributed_logger(builder; role="worker", id="worker-disabled") do
                        @info "should not be written" _group=EDDPotentials.DISTRIBUTED_LOG_GROUP
                    end
                end
                seekstart(err_io)
                read(err_io, String)
            end
            @test !isfile(disabled_path)
            @test !contains(disabled_err, "should not be written")

            builder.options.distributed.log_enabled = true
            builder.options.distributed.log_level = "verbose"
            @test_throws ArgumentError EDDPotentials.validate_builder_for_link!(
                builder,
                EDDPotentials.DistributedLinkBackend(),
            )

            builder.options.distributed.log_level = "info"
            builder.options.distributed.log_interval_seconds = 0.0
            @test_throws ArgumentError EDDPotentials.validate_builder_for_link!(
                builder,
                EDDPotentials.DistributedLinkBackend(),
            )
        end
    end

    @testset "Worker idle logging" begin
        mktempdir() do temp
            state = EDDPotentials.BuilderState(
                seedfile="seed.cell",
                seedfile_calc="calc.cell",
                workdir=".",
            )
            distributed = EDDPotentials.DistributedLinkOption()
            distributed.log_enabled = true
            distributed.log_level = "debug"
            builder = _distributed_builder(temp; state, distributed)
            link_path = joinpath(temp, "link.toml")
            EDDPotentials.save_builder(link_path, builder)

            result, err_text = mktemp() do _, err_io
                result = redirect_stderr(err_io) do
                    EDDPotentials.distributed_worker(link_path; worker_id="idle-worker", once=true)
                end
                seekstart(err_io)
                result, read(err_io, String)
            end
            @test result == false
            log_path = joinpath(temp, ".eddp-distributed", "logs", "trainer-idle-worker.log")
            text = read(log_path, String)
            @test contains(text, "distributed link worker idle")
            @test contains(text, "timestamp=")
            @test contains(text, "level=INFO")
            @test contains(text, "role=worker")
            @test contains(text, "id=idle-worker")
            @test contains(text, "state=IDLE")
            @test contains(text, "waiting=true")
            @test contains(text, "next_action=poll_control")
            @test contains(err_text, "distributed link worker idle")

            withenv("SLURM_JOB_ID" => "17", "WORKER_ID" => nothing) do
                redirect_stderr(devnull) do
                    @test EDDPotentials.distributed_worker(link_path; once=true) == false
                end
            end
            log_dir = joinpath(temp, ".eddp-distributed", "logs")
            trainer_logs = filter(
                name -> startswith(name, "trainer-17-") && endswith(name, ".log"),
                readdir(log_dir),
            )
            @test length(trainer_logs) == 1
            @test contains(read(joinpath(log_dir, only(trainer_logs)), String), "id=17-")

            withenv("SLURM_JOB_ID" => nothing, "SGE_TASK_ID" => nothing, "WORKER_ID" => nothing) do
                redirect_stderr(devnull) do
                    @test EDDPotentials.distributed_worker(link_path; once=true) == false
                end
            end
            na_logs = filter(
                name -> startswith(name, "trainer-NA-") && endswith(name, ".log"),
                readdir(log_dir),
            )
            @test length(na_logs) == 1
        end
    end

    @testset "Child supervision logging" begin
        mktempdir() do temp
            distributed = EDDPotentials.DistributedLinkOption()
            distributed.log_enabled = true
            distributed.log_level = "debug"
            distributed.worker_heartbeat_interval_seconds = 0.0
            distributed.control_poll_interval_seconds = 0.01
            builder = _distributed_builder(temp; distributed)
            worker = EDDPotentials.WorkerState("child-worker"; iteration=0)
            phase_dir = joinpath(temp, ".eddp-distributed", "phases", "gen0-train")
            mkpath(phase_dir)
            control = Dict{String,Any}(
                "phase" => "training",
                "iteration" => 0,
                "epoch" => 1,
                "run_id" => "child-test",
            )
            EDDPotentials.advance_worker!(
                builder,
                worker,
                EDDPotentials.WORKER_WAIT;
                phase="training",
                iteration=0,
                epoch=1,
                run_id="child-test",
            )
            EDDPotentials.advance_worker!(
                builder,
                worker,
                EDDPotentials.WORKER_TRAINING_RUNNING;
                phase="training",
                iteration=0,
                epoch=1,
                run_id="child-test",
                model_path="model.json",
            )
            cmd = Cmd([Base.julia_cmd()..., "--startup-file=no", "-e", "sleep(0.2)"])
            redirect_stderr(devnull) do
                EDDPotentials._with_distributed_logger(builder; role="worker", id="child-worker") do
                    @test EDDPotentials._worker_supervise_child!(
                        builder,
                        worker,
                        control,
                        phase_dir,
                        cmd;
                        state="TRAINING_RUNNING",
                        model_path="model.json",
                    )
                end
            end

            child_log = read(joinpath(temp, ".eddp-distributed", "logs", "worker-child-worker.log"), String)
            @test contains(child_log, "distributed link child launched")
            @test contains(child_log, "child_pid=")
            @test contains(child_log, "command=")
            @test contains(child_log, "model_path=model.json")
            @test contains(child_log, "distributed link child still running")
            @test contains(child_log, "runtime_seconds=")
            @test contains(child_log, "distributed link child finished")
            @test contains(child_log, "success=true")
        end
    end

    @testset "DFT control identity" begin
        mktempdir() do temp
            builder = _distributed_builder(temp)
            worker = EDDPotentials.WorkerState("worker-1"; iteration=0)
            control = Dict{String,Any}(
                "phase" => "dft",
                "iteration" => 1,
                "epoch" => 1,
                "run_id" => "test-1",
                "phase_dir" => joinpath(temp, ".eddp-distributed", "phases", "gen1-dft"),
            )

            handled = Set{Tuple{Int,Int}}([(0, 1)])
            @test_throws ArgumentError EDDPotentials._worker_handle_control!(
                builder,
                worker,
                control,
                handled,
            )

            handled = Set{Tuple{Int,Int}}([(1, 1)])
            @test EDDPotentials._worker_handle_control!(
                builder,
                worker,
                control,
                handled,
            ) == true
            @test worker.state == "IDLE"
        end
    end

    @testset "DFT waits until pass or impossible failure" begin
        mktempdir() do temp
            state = EDDPotentials.BuilderState(
                seedfile="seed.cell",
                seedfile_calc="calc.cell",
                workdir=temp,
            )
            state.per_generation_threshold = 0.8
            distributed = EDDPotentials.DistributedLinkOption()
            distributed.control_poll_interval_seconds = 0.01
            builder = _distributed_builder(temp; state, distributed)

            _write_minimal_castep_inputs(temp)
            _write_generated_res(
                temp,
                ["calc-$(lpad(string(i), 3, '0')).res" for i = 1:10],
            )

            crud_workdir = EDDPotentials.server_prepare_dft_phase!(builder, 0)
            good_dir = joinpath(crud_workdir, "good_castep")
            bad_dir = joinpath(crud_workdir, "bad_castep")
            hopper_dir = joinpath(crud_workdir, "hopper")
            for i = 1:7
                write(joinpath(good_dir, "good-$i.res"), "TITL test\nEND\n")
            end
            write(joinpath(bad_dir, "bad-1.res"), "TITL test\nEND\n")

            phase_dir = joinpath(temp, ".eddp-distributed", "phases", "gen0-dft")
            task = @async EDDPotentials._server_wait_for_dft!(builder, phase_dir)
            sleep(0.05)
            @test !istaskdone(task)
            write(joinpath(good_dir, "good-8.res"), "TITL test\nEND\n")
            sleep(0.05)
            @test !istaskdone(task)

            for file in filter(endswith(".res"), readdir(hopper_dir; join=true))
                rm(file)
            end
            write(joinpath(crud_workdir, "calc-running.res"), "TITL test\nEND\n")
            sleep(0.05)
            @test !istaskdone(task)

            write(joinpath(crud_workdir, "calc-running.castep"), "running\n")
            sleep(0.05)
            @test !istaskdone(task)

            distributed.dft_stale_seconds = 0.0
            @test fetch(task) == (8, 1)
            @test isfile(joinpath(phase_dir, "STOP"))
        end
    end

    @testset "DFT terminal failure waits for drained queue" begin
        mktempdir() do temp
            state = EDDPotentials.BuilderState(
                seedfile="seed.cell",
                seedfile_calc="calc.cell",
                workdir=temp,
            )
            state.per_generation_threshold = 0.8
            distributed = EDDPotentials.DistributedLinkOption()
            distributed.control_poll_interval_seconds = 0.01
            builder = _distributed_builder(temp; state, distributed)

            _write_minimal_castep_inputs(temp)
            _write_generated_res(
                temp,
                ["calc-$(lpad(string(i), 3, '0')).res" for i = 1:10],
            )

            crud_workdir = EDDPotentials.server_prepare_dft_phase!(builder, 0)
            good_dir = joinpath(crud_workdir, "good_castep")
            bad_dir = joinpath(crud_workdir, "bad_castep")
            hopper_dir = joinpath(crud_workdir, "hopper")
            for i = 1:7
                write(joinpath(good_dir, "good-$i.res"), "TITL test\nEND\n")
            end
            for i = 1:3
                write(joinpath(bad_dir, "bad-$i.res"), "TITL test\nEND\n")
            end

            phase_dir = joinpath(temp, ".eddp-distributed", "phases", "gen0-dft")
            task = @async try
                EDDPotentials._server_wait_for_dft!(builder, phase_dir)
            catch err
                err
            end
            sleep(0.05)
            @test !istaskdone(task)

            for file in filter(endswith(".res"), readdir(hopper_dir; join=true))
                rm(file)
            end
            err = fetch(task)
            @test err isa ErrorException
            @test contains(err.msg, "Too many failed DFT calculations")
            @test isfile(joinpath(phase_dir, "FAILED"))
        end
    end

    @testset "Server wait logging" begin
        mktempdir() do temp
            state = EDDPotentials.BuilderState(
                seedfile="seed.cell",
                seedfile_calc="calc.cell",
                workdir=temp,
            )
            state.per_generation_threshold = 1.0
            trainer = EDDPotentials.TrainingOption()
            trainer.nmodels = 1
            distributed = EDDPotentials.DistributedLinkOption()
            distributed.log_enabled = true
            distributed.log_level = "debug"
            distributed.log_interval_seconds = 0.0
            builder = _distributed_builder(temp; state, trainer, distributed)

            _write_minimal_castep_inputs(temp)
            gen0 = _write_generated_res(temp, ["calc-001.res"])
            crud_workdir = EDDPotentials.server_prepare_dft_phase!(builder, 0)
            cp(
                joinpath(gen0, "calc-001.res"),
                joinpath(crud_workdir, "good_castep", "calc-001.res"),
            )
            rm(joinpath(crud_workdir, "hopper", "calc-001.res"))

            phase_dir = joinpath(temp, ".eddp-distributed", "phases", "gen0-dft")
            redirect_stderr(devnull) do
                EDDPotentials._with_distributed_logger(builder; role="server", id="wait-test") do
                    @test EDDPotentials._server_wait_for_dft!(builder, phase_dir) == (1, 0)
                end
            end
            dft_log = read(joinpath(temp, ".eddp-distributed", "logs", "server-wait-test.log"), String)
            @test contains(dft_log, "distributed link waiting for DFT results")
            @test contains(dft_log, "timestamp=")
            @test contains(dft_log, "level=INFO")
            @test contains(dft_log, "role=server")
            @test contains(dft_log, "id=")
            @test contains(dft_log, "good=")
            @test contains(dft_log, "bad=")
            @test contains(dft_log, "queued=")
            @test contains(dft_log, "active=")
            @test contains(dft_log, "required_success=")
            @test contains(dft_log, "input_count=")
            @test !contains(dft_log, "distributed link DFT poll")

            training_phase = EDDPotentials.server_prepare_training_phase!(builder, 0)
            training_dir = joinpath(temp, EDDPotentials.TRAINING_DIR)
            mkpath(training_dir)
            write(joinpath(training_dir, "gen-0-model-1.json"), "{}")
            redirect_stderr(devnull) do
                EDDPotentials._with_distributed_logger(builder; role="server", id="train-wait-test") do
                    @test EDDPotentials._server_wait_for_training!(builder, training_phase) == 1
                end
            end
            train_log = read(joinpath(temp, ".eddp-distributed", "logs", "server-train-wait-test.log"), String)
            @test contains(train_log, "distributed link waiting for training models")
            @test contains(train_log, "timestamp=")
            @test contains(train_log, "level=INFO")
            @test contains(train_log, "models_found=")
            @test contains(train_log, "required_models=")
            @test !contains(train_log, "distributed link training poll")
        end
    end

    @testset "Stop and retry behavior" begin
        mktempdir() do temp
            state = EDDPotentials.BuilderState(
                seedfile="seed.cell",
                seedfile_calc="calc.cell",
                workdir=temp,
            )
            state.per_generation_threshold = 1.0
            cf = EDDPotentials.CellFeatureConfig(elements=["H"])
            rss = EDDPotentials.RssSetting()
            trainer = EDDPotentials.TrainingOption()
            trainer.nmodels = 1
            distributed = EDDPotentials.DistributedLinkOption()
            distributed.control_poll_interval_seconds = 0.01
            builder = _distributed_builder(temp; state, cf, rss, trainer, distributed)

            _write_minimal_castep_inputs(temp)
            _write_generated_res(temp, ["calc-001.res"])

            EDDPotentials.server_prepare_dft_phase!(builder, 0)
            touch(joinpath(temp, "STOP"))
            phase_dir = joinpath(temp, ".eddp-distributed", "phases", "gen0-dft")
            @test EDDPotentials._server_wait_for_dft!(builder, phase_dir) == :stopped
            @test isfile(joinpath(phase_dir, "ABORTED.json"))
            rm(joinpath(temp, "STOP"))

            EDDPotentials.server_prepare_dft_phase!(builder, 0)
            crud_workdir = joinpath(temp, "crud-work", "gen0")
            for file in filter(endswith(".res"), readdir(joinpath(crud_workdir, "hopper"); join=true))
                rm(file)
            end
            write(joinpath(crud_workdir, "stale.res"), "TITL test\nEND\n")
            write(joinpath(crud_workdir, "stale.castep"), "running\n")
            distributed.dft_stale_seconds = 0.0
            touch(joinpath(temp, "STOP"))
            @test EDDPotentials._server_wait_for_dft!(builder, phase_dir) == :stopped
            @test isfile(joinpath(phase_dir, "ABORTED.json"))
            @test !isfile(joinpath(phase_dir, "FAILED"))
            rm(joinpath(temp, "STOP"))

            training_phase = EDDPotentials.server_prepare_training_phase!(builder, 0)
            touch(joinpath(training_phase, "STOP"))
            worker = EDDPotentials.WorkerState("worker-training-stop"; iteration=0)
            control = Dict{String,Any}(
                "phase" => "training",
                "phase_dir" => training_phase,
                "iteration" => 0,
                "epoch" => 1,
                "run_id" => "stop-test",
                "status" => "running",
            )
            EDDPotentials.advance_worker!(
                builder,
                worker,
                EDDPotentials.WORKER_WAIT;
                phase="training",
                iteration=0,
                epoch=1,
                run_id="stop-test",
            )
            @test EDDPotentials._worker_run_training!(builder, worker, control)
            states = EDDPotentials.read_worker_states(builder)
            stopped = filter(state -> state["worker_id"] == "worker-training-stop", states)
            @test length(stopped) == 1
            @test stopped[1]["state"] == "STOPPED"
            @test stopped[1]["message"] == "training phase stop requested"
        end
    end

    @testset "Enabled overrides backend selectors" begin
        mktempdir() do temp
            state = EDDPotentials.BuilderState(
                seedfile="seed.cell",
                seedfile_calc="calc.cell",
                workdir=temp,
            )
            state.dft_mode = "mock"
            cf = EDDPotentials.CellFeatureConfig(elements=["H"])
            rss = EDDPotentials.RssSetting()
            trainer = EDDPotentials.TrainingOption()
            trainer.external = true
            trainer.external_mode = "local"
            distributed = EDDPotentials.DistributedLinkOption()
            distributed.manage_worker_jobs = false
            builder = _distributed_builder(temp; state, cf, rss, trainer, distributed)

            gen0 = _write_generated_res(temp, ["calc-001.res"])
            _write_minimal_castep_inputs(temp)

            EDDPotentials.server_prepare_dft_phase!(builder, 0)
            phase_dir = joinpath(temp, ".eddp-distributed", "phases", "gen0-dft")
            cp(
                joinpath(gen0, "calc-001.res"),
                joinpath(temp, "crud-work", "gen0", "good_castep", "calc-001.res"),
            )
            rm(joinpath(temp, "crud-work", "gen0", "hopper", "calc-001.res"))
            @test EDDPotentials.server_run_dft_phase!(builder, 0) == :ok
            @test isfile(joinpath(EDDPotentials.output_structure_dir(temp, 0), "calc-001.res"))
            @test isfile(joinpath(phase_dir, "DONE"))
        end
    end

    @testset "Scheduler script rendering" begin
        mktempdir() do temp
            script = joinpath(temp, "distributed_job.sh")
            open(script, "w") do io
                println(io, "#!/bin/bash")
                println(io, "{{EDDP_DISTRIBUTED_WORKER_CMD}}")
            end

            state = EDDPotentials.BuilderState(
                seedfile="seed.cell",
                seedfile_calc="calc.cell",
                workdir=temp,
            )
            cf = EDDPotentials.CellFeatureConfig(elements=["H"])
            rss = EDDPotentials.RssSetting()
            trainer = EDDPotentials.TrainingOption()
            distributed = EDDPotentials.DistributedLinkOption()
            distributed.manage_worker_jobs = false
            scheduler = EDDPotentials.SchedulerConfig(
                script=script,
                njobs=2,
                type="SLURM",
                submit_jobs=true,
            )
            opt = EDDPotentials.BuilderOption(state, cf, nothing, rss, trainer, scheduler, distributed)
            builder = Builder(opt)
            builder.state.builder_file_path = joinpath(temp, "link.toml")
            EDDPotentials.save_builder(builder.state.builder_file_path, builder)

            rendered = EDDPotentials._render_distributed_worker_job_script(
                builder,
                scheduler,
                joinpath(temp, "submit"),
            )
            text = read(rendered, String)
            @test !contains(text, "{{EDDP_DISTRIBUTED_WORKER_CMD}}")
            @test contains(text, "EDDPotentials.distributed_worker()")
            @test contains(text, "--worker-id \"\$WORKER_ID\"")
            @test contains(text, "SLURM_JOB_ID")
            @test EDDPotentials._server_submit_workers(builder, 0) == ""

            distributed.manage_worker_jobs = true
            scheduler = EDDPotentials.SchedulerConfig(
                script=script,
                njobs=2,
                type="SLURM",
                submit_jobs=false,
            )
            builder = Builder(EDDPotentials.BuilderOption(state, cf, nothing, rss, trainer, scheduler, distributed))
            builder.state.builder_file_path = joinpath(temp, "link.toml")
            EDDPotentials.save_builder(builder.state.builder_file_path, builder)
            @test "crud.pl" in EDDPotentials._required_commands(
                EDDPotentials.DistributedLinkBackend(),
                builder,
            )
            @test !("sbatch" in EDDPotentials._required_commands(
                EDDPotentials.DistributedLinkBackend(),
                builder,
            ))
            @test EDDPotentials._server_submit_workers(builder, 0) == ""
        end
    end
end
