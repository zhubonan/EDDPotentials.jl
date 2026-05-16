using EDDPotentials
using Test
using TOML
using Configurations

isdefined(Main, :_h2_cell) || include("utils.jl")

check_equal(a, b) = a == b
check_equal(a::AbstractArray, b::AbstractArray) = all(a .== b)

function _builder_rattled_family_store()
    base = _h2o_cell()
    labels = [
        "builder-parent-a",
        "builder-parent-a-shake-1",
        "builder-parent-a-force-shake-1",
        "builder-parent-b",
        "builder-parent-b-shake-1",
        "builder-parent-b-shake-2",
        "builder-parent-c",
        "builder-parent-c-shake-1",
        "builder-parent-c-force-shake-1",
    ]
    cells = typeof(base)[deepcopy(base) for _ in labels]
    for (i, cell) in enumerate(cells)
        cell.metadata[:label] = labels[i]
        set_positions!(cell, get_positions(cell) .+ 0.01 * i)
    end
    sc = EDDPotentials.StructureContainer(cells, -12.5 .+ 0.01 .* collect(eachindex(cells)), labels)
    cf = EDDPotentials.CellFeature(unique(species(base)); p2=[2, 3], p3=[2], q3=[1], rcut2=4.0, rcut3=3.5)
    EDDPotentials.FeatureStore(sc, cf; show_progress=false)
end

function _builder_parent_sets(train, val, test)
    parent(label) = begin
        result = match(r"^(.+?)(?:-force)?-shake-\d+$", label)
        isnothing(result) ? label : result.captures[1]
    end
    map(fc -> Set(parent.(fc.labels)), (train, val, test))
end

@testset "Builder" begin

    @testset "Serialization" begin


        "Test for round-trip toml conversion"
        function test_to_from_toml(obj::T) where {T}
            obj_reconstructed = mktempdir() do temp
                open(joinpath(temp, "test.toml"), "w") do f
                    to_toml(f, obj)
                end
                from_toml(T, joinpath(temp, "test.toml"))
            end
            for name in fieldnames(T)
                @test check_equal(
                    getproperty(obj, name),
                    getproperty(obj_reconstructed, name),
                )
            end
        end

        "Test for round-trip toml conversion"
        function test_to_from_toml_builder(obj)

            options = mktempdir() do temp
                EDDPotentials.save_builder(joinpath(temp, "test.toml"), builder)
                o2 = from_toml(EDDPotentials.BuilderOption, joinpath(temp, "test.toml"))
                open(joinpath(temp, "showout"), "w") do fh
                    Base.show(fh, o2)
                end
                o2
            end
            obj_reconstructed = Builder(options)
            for name in fieldnames(Builder)
                @test check_equal(
                    getproperty(obj, name),
                    getproperty(obj_reconstructed, name),
                )
            end
        end



        state = EDDPotentials.BuilderState(seedfile="seed.cell", seedfile_calc="calc.cell")
        state.rss_prune.enabled = true
        state.rss_prune.pool_size = 25
        state.rss_prune.keep_fraction = 0.2

        cf = EDDPotentials.CellFeatureConfig(elements=["H"])

        rss = EDDPotentials.RssSetting()
        rss.prune.enabled = true
        rss.prune.pool_size = 50
        rss.prune.keep_fraction = 0.3

        opt = EDDPotentials.BuilderOption(state, cf, nothing, rss, EDDPotentials.TrainingOption(), nothing)
        test_to_from_toml(opt)

        builder = Builder(opt)
        test_to_from_toml_builder(builder)

        distributed = EDDPotentials.DistributedLinkOption()
        distributed.enabled = true
        distributed.coordination_dir = ".custom-distributed"
        distributed.log_enabled = true
        distributed.log_level = "debug"
        distributed.log_dir = "diag-logs"
        distributed.log_interval_seconds = 1.5
        opt_distributed = EDDPotentials.BuilderOption(
            state,
            cf,
            nothing,
            rss,
            EDDPotentials.TrainingOption(),
            nothing,
            distributed,
        )
        test_to_from_toml(opt_distributed)
    end

    @testset "Training scheduler script rendering" begin
        mktempdir() do temp
            script = joinpath(temp, "train_job.sh")
            open(script, "w") do io
                println(io, "#!/bin/bash")
                println(io, "#SBATCH --job-name=eddp-train")
                println(io, "{{EDDP_TRAIN_CMD}}")
            end

            state = EDDPotentials.BuilderState(
                seedfile="seed.cell",
                seedfile_calc="calc.cell",
                workdir=temp,
            )
            cf = EDDPotentials.CellFeatureConfig(elements=["H"])
            rss = EDDPotentials.RssSetting()
            trainer = EDDPotentials.TrainingOption()
            trainer.external_mode = "submit"
            scheduler = EDDPotentials.SchedulerConfig(
                script=script,
                njobs=4,
                type="SLURM",
                submit_jobs=false,
            )
            opt = EDDPotentials.BuilderOption(state, cf, nothing, rss, trainer, scheduler)
            builder = Builder(opt)
            builder.state.builder_file_path = joinpath(temp, "link.toml")
            EDDPotentials.save_builder(builder.state.builder_file_path, builder)

            workdir = joinpath(temp, "training-work", "gen0")
            rendered_script =
                EDDPotentials._render_training_job_script(builder, scheduler, workdir)
            rendered = read(rendered_script, String)

            @test !contains(rendered, "{{EDDP_TRAIN_CMD}}")
            @test contains(rendered, "EDDPotentials.run_trainer()")
            @test contains(rendered, "--id \"\$WORKER_ID\"")
            @test contains(rendered, "SLURM_ARRAY_TASK_ID")

            slurm_cmd = EDDPotentials._training_submit_cmd(scheduler, workdir)
            @test contains(string(slurm_cmd), "sbatch")
            @test contains(string(slurm_cmd), "--array=1-4")
            @test contains(string(slurm_cmd), "job.sh")

            sge_scheduler = EDDPotentials.SchedulerConfig(
                script=script,
                njobs=4,
                type="SGE",
                submit_jobs=false,
            )
            sge_cmd = EDDPotentials._training_submit_cmd(sge_scheduler, workdir)
            @test contains(string(sge_cmd), "qsub")
            @test contains(string(sge_cmd), "1-4")
            @test contains(string(sge_cmd), "-cwd")

            bad_script = joinpath(temp, "bad_train_job.sh")
            write(bad_script, "#!/bin/bash\n")
            bad_scheduler = EDDPotentials.SchedulerConfig(script=bad_script, type="SLURM")
            @test_throws ArgumentError EDDPotentials._render_training_job_script(
                builder,
                bad_scheduler,
                workdir,
            )
        end
    end

    @testset "DFT readiness uses actual generated inputs" begin
        mktempdir() do temp
            state = EDDPotentials.BuilderState(
                seedfile="seed.cell",
                seedfile_calc="calc.cell",
                workdir=temp,
            )
            state.per_generation_threshold = 1.0
            cf = EDDPotentials.CellFeatureConfig(elements=["H"])
            builder = Builder(
                EDDPotentials.BuilderOption(
                    state,
                    cf,
                    nothing,
                    EDDPotentials.RssSetting(),
                    EDDPotentials.TrainingOption(),
                    nothing,
                ),
            )

            gen0 = EDDPotentials.input_structure_dir(temp, 0)
            dft0 = EDDPotentials.output_structure_dir(temp, 0)
            mkpath(gen0)
            mkpath(dft0)
            for i = 1:3
                write(joinpath(gen0, "s$i.res"), "TITL test\nEND\n")
            end
            for i = 1:2
                write(joinpath(dft0, "s$i.res"), "TITL test\nEND\n")
            end

            @test EDDPotentials.generated_input_count(builder, 0) == 3
            @test EDDPotentials.required_success_count(3, 1.0) == 3
            @test !EDDPotentials.is_training_data_ready(builder, 0)
            @test_throws ErrorException EDDPotentials.validate_dft_outputs!(builder, 0)

            write(joinpath(dft0, "s3.res"), "TITL test\nEND\n")
            @test EDDPotentials.is_training_data_ready(builder, 0)
            @test EDDPotentials.validate_dft_outputs!(builder, 0)
        end

        @test !(:hyper_ndim in fieldnames(EDDPotentials.BuilderState))
        @test EDDPotentials.BuilderState(seedfile="seed.cell", seedfile_calc="calc.cell").per_generation_threshold == 0.7

        trainer = EDDPotentials.TrainingOption()
        trainer.nmodels = 10
        trainer.external = true
        trainer.external_completion_ratio = 0.5
            @test EDDPotentials._required_models_for_ensemble(trainer) == 5
    end

    @testset "Workflow artifact readiness skips completed phases" begin
        mktempdir() do temp
            state = EDDPotentials.BuilderState(
                seedfile="seed.cell",
                seedfile_calc="calc.cell",
                workdir=temp,
            )
            state.n_initial = 3
            state.per_generation_threshold = 1.0
            cf = EDDPotentials.CellFeatureConfig(elements=["H"])
            trainer = EDDPotentials.TrainingOption()
            trainer.nmodels = 2
            trainer.external = false
            builder = Builder(
                EDDPotentials.BuilderOption(
                    state,
                    cf,
                    nothing,
                    EDDPotentials.RssSetting(),
                    trainer,
                    nothing,
                ),
            )

            gen0 = EDDPotentials.input_structure_dir(temp, 0)
            dft0 = EDDPotentials.output_structure_dir(temp, 0)
            mkpath(gen0)
            mkpath(dft0)
            for i = 1:3
                write(joinpath(gen0, "s$i.res"), "TITL test\nEND\n")
                write(joinpath(dft0, "s$i.res"), "TITL test\nEND\n")
            end

            @test EDDPotentials.is_generation_data_ready(builder, 0)
            @test EDDPotentials._generate_random_structures(builder, 0) == :ok
            @test EDDPotentials.validate_generated_structures!(builder, 0)
            @test length(readdir(gen0)) == 3

            dft_result = EDDPotentials.run_dft_phase!(EDDPotentials.LocalLinkBackend(), builder, 0)
            @test dft_result.status == :ok
            @test dft_result.message == "DFT outputs already ready"

            training_dir = joinpath(temp, EDDPotentials.TRAINING_DIR)
            mkpath(training_dir)
            for i = 1:2
                write(joinpath(training_dir, "gen-0-model-$i.json"), "{}")
            end
            dataset_path = EDDPotentials.dataset_name(builder, 0)
            mkpath(dirname(dataset_path))
            write(dataset_path, "placeholder")
            @test EDDPotentials.is_training_dataset_ready(builder, 0)
            sleep(1.1)
            write(joinpath(dft0, "s3.res"), "TITL updated\nEND\n")
            @test !EDDPotentials.is_training_dataset_ready(builder, 0)
            sleep(1.1)
            write(dataset_path, "placeholder")
            @test EDDPotentials.is_training_dataset_ready(builder, 0)
            @test EDDPotentials._models_ready_for_ensemble(builder)
            train_result = EDDPotentials.run_training_phase!(EDDPotentials.LocalLinkBackend(), builder, 0)
            @test train_result.status == :ok
            @test train_result.message == "models already ready"
        end
    end

    @testset "Distributed link file protocol helpers" begin
        mktempdir() do temp
            state = EDDPotentials.BuilderState(
                seedfile="seed.cell",
                seedfile_calc="calc.cell",
                workdir=temp,
            )
            state.mpinp = 8
            state.per_generation_threshold = 0.5
            cf = EDDPotentials.CellFeatureConfig(elements=["H"])
            rss = EDDPotentials.RssSetting()
            trainer = EDDPotentials.TrainingOption()
            trainer.nmodels = 3
            distributed = EDDPotentials.DistributedLinkOption()
            distributed.enabled = true
            distributed.worker_heartbeat_interval_seconds = 0.01
            opt = EDDPotentials.BuilderOption(state, cf, nothing, rss, trainer, nothing, distributed)
            builder = Builder(opt)
            builder.state.builder_file_path = joinpath(temp, "link.toml")
            write(builder.state.builder_file_path, "")

            open(joinpath(temp, "calc.cell"), "w") do io
                println(io, "%BLOCK LATTICE_CART")
                println(io, "1 0 0")
                println(io, "0 1 0")
                println(io, "0 0 1")
                println(io, "%ENDBLOCK LATTICE_CART")
            end
            write(joinpath(temp, "calc.param"), "task : SinglePoint\n")
            gen0 = EDDPotentials.input_structure_dir(temp, 0)
            mkpath(gen0)
            write(joinpath(gen0, "calc-001.res"), "TITL test\nEND\n")
            write(joinpath(gen0, "calc-002.res"), "TITL test\nEND\n")

            crud_workdir = EDDPotentials.server_prepare_dft_phase!(builder, 0)
            @test isdir(joinpath(crud_workdir, "hopper"))
            @test isdir(joinpath(crud_workdir, "good_castep"))
            @test isdir(joinpath(crud_workdir, "bad_castep"))
            @test isfile(joinpath(crud_workdir, "calc.cell"))
            @test isfile(joinpath(crud_workdir, "calc.param"))
            @test count(endswith(".res"), readdir(joinpath(crud_workdir, "hopper"))) == 2

            EDDPotentials.write_worker_heartbeat!(
                builder,
                "worker-1";
                state="DFT_RUNNING",
                phase="dft",
                iteration=0,
                epoch=1,
                child_pid=123,
                message="testing",
            )
            states = EDDPotentials.read_worker_states(builder)
            @test length(states) == 1
            @test states[1]["worker_id"] == "worker-1"
            @test states[1]["state"] == "DFT_RUNNING"

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
            @test "OMP_NUM_THREADS=1" in train_cmd_env
            @test "EDDP_CUDA_DEVICE=0" in train_cmd_env
            @test_throws ArgumentError EDDPotentials.training_slot_model_path(builder, 0, 1)
            @test_throws ArgumentError EDDPotentials.worker_claim_training_slot!(builder, 0, "worker-1")
            model_path = joinpath(temp, EDDPotentials.TRAINING_DIR, "gen-0-model-abcd1234.json")
            mkpath(dirname(model_path))
            write(model_path, "{}")
            trainer.nmodels = 4
            EDDPotentials.server_prepare_training_phase!(builder, 0)
            @test !isfile(model_path)
        end
    end

    @testset "Distributed link logging" begin
        mktempdir() do temp
            state = EDDPotentials.BuilderState(
                seedfile="seed.cell",
                seedfile_calc="calc.cell",
                workdir=temp,
            )
            distributed = EDDPotentials.DistributedLinkOption()
            distributed.enabled = true
            distributed.log_enabled = true
            distributed.log_level = "info"
            distributed.log_interval_seconds = 0.0
            builder = Builder(
                EDDPotentials.BuilderOption(
                    state,
                    EDDPotentials.CellFeatureConfig(elements=["H"]),
                    nothing,
                    EDDPotentials.RssSetting(),
                    EDDPotentials.TrainingOption(),
                    nothing,
                    distributed,
                ),
            )

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

    @testset "Distributed link worker idle logging" begin
        mktempdir() do temp
            state = EDDPotentials.BuilderState(
                seedfile="seed.cell",
                seedfile_calc="calc.cell",
                workdir=".",
            )
            distributed = EDDPotentials.DistributedLinkOption()
            distributed.enabled = true
            distributed.log_enabled = true
            distributed.log_level = "debug"
            opt = EDDPotentials.BuilderOption(
                state,
                EDDPotentials.CellFeatureConfig(elements=["H"]),
                nothing,
                EDDPotentials.RssSetting(),
                EDDPotentials.TrainingOption(),
                nothing,
                distributed,
            )
            builder = Builder(opt)
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
            log_path = joinpath(temp, ".eddp-distributed", "logs", "worker-idle-worker.log")
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
        end
    end

    @testset "Distributed link child supervision logging" begin
        mktempdir() do temp
            state = EDDPotentials.BuilderState(
                seedfile="seed.cell",
                seedfile_calc="calc.cell",
                workdir=temp,
            )
            distributed = EDDPotentials.DistributedLinkOption()
            distributed.enabled = true
            distributed.log_enabled = true
            distributed.log_level = "debug"
            distributed.worker_heartbeat_interval_seconds = 0.0
            distributed.control_poll_interval_seconds = 0.01
            builder = Builder(
                EDDPotentials.BuilderOption(
                    state,
                    EDDPotentials.CellFeatureConfig(elements=["H"]),
                    nothing,
                    EDDPotentials.RssSetting(),
                    EDDPotentials.TrainingOption(),
                    nothing,
                    distributed,
                ),
            )
            worker = EDDPotentials.WorkerState("child-worker"; iteration=0)
            phase_dir = joinpath(temp, ".eddp-distributed", "phases", "gen0-train")
            mkpath(phase_dir)
            control = Dict{String,Any}(
                "phase" => "training",
                "iteration" => 0,
                "epoch" => 1,
                "run_id" => "child-test",
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

    @testset "Distributed link DFT control identity" begin
        mktempdir() do temp
            state = EDDPotentials.BuilderState(
                seedfile="seed.cell",
                seedfile_calc="calc.cell",
                workdir=temp,
            )
            distributed = EDDPotentials.DistributedLinkOption()
            distributed.enabled = true
            builder = Builder(
                EDDPotentials.BuilderOption(
                    state,
                    EDDPotentials.CellFeatureConfig(elements=["H"]),
                    nothing,
                    EDDPotentials.RssSetting(),
                    EDDPotentials.TrainingOption(),
                    nothing,
                    distributed,
                ),
            )
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

    @testset "Distributed link DFT waits until pass or impossible failure" begin
        mktempdir() do temp
            state = EDDPotentials.BuilderState(
                seedfile="seed.cell",
                seedfile_calc="calc.cell",
                workdir=temp,
            )
            state.per_generation_threshold = 0.8
            distributed = EDDPotentials.DistributedLinkOption()
            distributed.enabled = true
            distributed.control_poll_interval_seconds = 0.01
            builder = Builder(
                EDDPotentials.BuilderOption(
                    state,
                    EDDPotentials.CellFeatureConfig(elements=["H"]),
                    nothing,
                    EDDPotentials.RssSetting(),
                    EDDPotentials.TrainingOption(),
                    nothing,
                    distributed,
                ),
            )

            open(joinpath(temp, "calc.cell"), "w") do io
                println(io, "%BLOCK LATTICE_CART")
                println(io, "1 0 0")
                println(io, "0 1 0")
                println(io, "0 0 1")
                println(io, "%ENDBLOCK LATTICE_CART")
            end
            write(joinpath(temp, "calc.param"), "task : SinglePoint\n")
            gen0 = EDDPotentials.input_structure_dir(temp, 0)
            mkpath(gen0)
            for i = 1:10
                write(joinpath(gen0, "calc-$(lpad(string(i), 3, '0')).res"), "TITL test\nEND\n")
            end

            crud_workdir = EDDPotentials.server_prepare_dft_phase!(builder, 0)
            good_dir = joinpath(crud_workdir, "good_castep")
            bad_dir = joinpath(crud_workdir, "bad_castep")
            for i = 1:7
                write(joinpath(good_dir, "good-$i.res"), "TITL test\nEND\n")
            end
            write(joinpath(bad_dir, "bad-1.res"), "TITL test\nEND\n")

            phase_dir = joinpath(temp, ".eddp-distributed", "phases", "gen0-dft")
            task = @async EDDPotentials._server_wait_for_dft!(builder, phase_dir)
            sleep(0.05)
            @test !istaskdone(task)
            write(joinpath(good_dir, "good-8.res"), "TITL test\nEND\n")
            @test fetch(task) == (8, 1)
        end
    end

    @testset "Distributed link server wait logging" begin
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
            distributed.enabled = true
            distributed.log_enabled = true
            distributed.log_level = "debug"
            distributed.log_interval_seconds = 0.0
            builder = Builder(
                EDDPotentials.BuilderOption(
                    state,
                    EDDPotentials.CellFeatureConfig(elements=["H"]),
                    nothing,
                    EDDPotentials.RssSetting(),
                    trainer,
                    nothing,
                    distributed,
                ),
            )

            open(joinpath(temp, "calc.cell"), "w") do io
                println(io, "%BLOCK LATTICE_CART")
                println(io, "1 0 0")
                println(io, "0 1 0")
                println(io, "0 0 1")
                println(io, "%ENDBLOCK LATTICE_CART")
            end
            write(joinpath(temp, "calc.param"), "task : SinglePoint\n")
            gen0 = EDDPotentials.input_structure_dir(temp, 0)
            mkpath(gen0)
            write(joinpath(gen0, "calc-001.res"), "TITL test\nEND\n")
            crud_workdir = EDDPotentials.server_prepare_dft_phase!(builder, 0)
            cp(
                joinpath(gen0, "calc-001.res"),
                joinpath(crud_workdir, "good_castep", "calc-001.res"),
            )

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

    @testset "Distributed link stop and retry behavior" begin
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
            distributed.enabled = true
            distributed.control_poll_interval_seconds = 0.01
            opt = EDDPotentials.BuilderOption(state, cf, nothing, rss, trainer, nothing, distributed)
            builder = Builder(opt)

            open(joinpath(temp, "calc.cell"), "w") do io
                println(io, "%BLOCK LATTICE_CART")
                println(io, "1 0 0")
                println(io, "0 1 0")
                println(io, "0 0 1")
                println(io, "%ENDBLOCK LATTICE_CART")
            end
            write(joinpath(temp, "calc.param"), "task : SinglePoint\n")
            gen0 = EDDPotentials.input_structure_dir(temp, 0)
            mkpath(gen0)
            write(joinpath(gen0, "calc-001.res"), "TITL test\nEND\n")

            EDDPotentials.server_prepare_dft_phase!(builder, 0)
            touch(joinpath(temp, "STOP"))
            phase_dir = joinpath(temp, ".eddp-distributed", "phases", "gen0-dft")
            @test EDDPotentials._server_wait_for_dft!(builder, phase_dir) == :stopped
            @test isfile(joinpath(phase_dir, "ABORTED.json"))
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
            @test EDDPotentials._worker_run_training!(builder, worker, control)
            states = EDDPotentials.read_worker_states(builder)
            stopped = filter(state -> state["worker_id"] == "worker-training-stop", states)
            @test length(stopped) == 1
            @test stopped[1]["state"] == "STOPPED"
            @test stopped[1]["message"] == "training phase stop requested"
        end
    end

    @testset "Distributed link enabled overrides backend selectors" begin
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
            distributed.enabled = true
            distributed.manage_worker_jobs = false
            opt = EDDPotentials.BuilderOption(state, cf, nothing, rss, trainer, nothing, distributed)
            builder = Builder(opt)

            open(joinpath(temp, "calc.cell"), "w") do io
                println(io, "%BLOCK LATTICE_CART")
                println(io, "1 0 0")
                println(io, "0 1 0")
                println(io, "0 0 1")
                println(io, "%ENDBLOCK LATTICE_CART")
            end
            write(joinpath(temp, "calc.param"), "task : SinglePoint\n")
            gen0 = EDDPotentials.input_structure_dir(temp, 0)
            mkpath(gen0)
            write(joinpath(gen0, "calc-001.res"), "TITL test\nEND\n")

            EDDPotentials.server_prepare_dft_phase!(builder, 0)
            phase_dir = joinpath(temp, ".eddp-distributed", "phases", "gen0-dft")
            cp(
                joinpath(gen0, "calc-001.res"),
                joinpath(temp, "crud-work", "gen0", "good_castep", "calc-001.res"),
            )
            @test EDDPotentials.server_run_dft_phase!(builder, 0) == :ok
            @test isfile(joinpath(EDDPotentials.output_structure_dir(temp, 0), "calc-001.res"))
            @test isfile(joinpath(phase_dir, "DONE"))
        end
    end

    @testset "Distributed link scheduler script rendering" begin
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
            @test contains(text, "SLURM_ARRAY_TASK_ID")
            @test EDDPotentials._server_submit_workers(builder, "dft", 0) == ""

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
            @test EDDPotentials._server_submit_workers(builder, "dft", 0) == ""
        end
    end

    @testset "Training dataset respects rattled groups when configured" begin
        mktempdir() do temp
            state = EDDPotentials.BuilderState(
                seedfile="seed.cell",
                seedfile_calc="calc.cell",
                workdir=temp,
            )
            cf = EDDPotentials.CellFeatureConfig(elements=["H", "O"])
            rss = EDDPotentials.RssSetting()
            trainer = EDDPotentials.TrainingOption()
            trainer.train_split = [1.0, 1.0, 1.0]
            trainer.respect_shakes = true
            opt = EDDPotentials.BuilderOption(state, cf, nothing, rss, trainer, nothing)
            builder = Builder(opt)

            fs = _builder_rattled_family_store()
            dataset_path = joinpath(temp, "training", "grouped-dataset.jld2")
            EDDPotentials.write_dataset(builder, fs; name=dataset_path)
            train, val, test = EDDPotentials.load_training_data(dataset_path)
            train_parents, val_parents, test_parents = _builder_parent_sets(train, val, test)

            @test isempty(intersect(train_parents, val_parents))
            @test isempty(intersect(train_parents, test_parents))
            @test isempty(intersect(val_parents, test_parents))
            @test isfile(splitext(dataset_path)[1] * ".yaml")
        end
    end
end
