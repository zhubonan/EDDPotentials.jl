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

        hpc = EDDPotentials.HPCWorkflowOption()
        hpc.enabled = true
        hpc.root = ".custom-hpc"
        opt_hpc = EDDPotentials.BuilderOption(
            state,
            cf,
            nothing,
            rss,
            EDDPotentials.TrainingOption(),
            nothing,
            hpc,
        )
        test_to_from_toml(opt_hpc)
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

    @testset "HPC file protocol helpers" begin
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
            hpc = EDDPotentials.HPCWorkflowOption()
            hpc.enabled = true
            hpc.heartbeat_interval = 0.01
            opt = EDDPotentials.BuilderOption(state, cf, nothing, rss, trainer, nothing, hpc)
            builder = Builder(opt)
            builder.state.builder_file_path = joinpath(temp, "link.toml")

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

            crud_workdir = EDDPotentials.hpc_prepare_crud_workdir!(builder, 0)
            @test isdir(joinpath(crud_workdir, "hopper"))
            @test isdir(joinpath(crud_workdir, "good_castep"))
            @test isdir(joinpath(crud_workdir, "bad_castep"))
            @test isfile(joinpath(crud_workdir, "calc.cell"))
            @test isfile(joinpath(crud_workdir, "calc.param"))
            @test count(endswith(".res"), readdir(joinpath(crud_workdir, "hopper"))) == 2

            EDDPotentials.hpc_write_heartbeat(
                builder,
                "worker-1";
                state="DFT_RUNNING",
                phase="dft",
                iteration=0,
                epoch=1,
                child_pid=123,
                message="testing",
            )
            states = EDDPotentials.hpc_worker_states(builder)
            @test length(states) == 1
            @test states[1]["worker_id"] == "worker-1"
            @test states[1]["state"] == "DFT_RUNNING"

            EDDPotentials.hpc_prepare_training_phase!(builder, 0)
            claim1 = EDDPotentials.hpc_claim_training_slot!(builder, 0, "worker-1")
            claim2 = EDDPotentials.hpc_claim_training_slot!(builder, 0, "worker-2")
            @test claim1 !== nothing
            @test claim2 !== nothing
            @test claim1[1] != claim2[1]
            model_path = EDDPotentials.hpc_training_model_path(builder, 0, claim1[1])
            @test endswith(model_path, "gen-0-model-slot-$(lpad(string(claim1[1]), 6, '0')).json")
            EDDPotentials.hpc_mark_training_slot_done!(builder, 0, claim1[1], model_path)
            @test isfile(joinpath(temp, ".eddp-hpc", "phases", "gen0-train", "slots", "$(lpad(string(claim1[1]), 6, '0')).done.json"))
        end
    end

    @testset "HPC stop and retry behavior" begin
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
            hpc = EDDPotentials.HPCWorkflowOption()
            hpc.enabled = true
            hpc.poll_interval = 0.01
            opt = EDDPotentials.BuilderOption(state, cf, nothing, rss, trainer, nothing, hpc)
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

            EDDPotentials.hpc_prepare_crud_workdir!(builder, 0)
            touch(joinpath(temp, "STOP"))
            phase_dir = joinpath(temp, ".eddp-hpc", "phases", "gen0-dft")
            @test_throws ErrorException EDDPotentials._hpc_wait_for_dft!(builder, phase_dir)
            @test isfile(joinpath(phase_dir, "ABORTED.json"))
            rm(joinpath(temp, "STOP"))

            EDDPotentials.hpc_prepare_training_phase!(builder, 0)
            claim1 = EDDPotentials.hpc_claim_training_slot!(builder, 0, "worker-1")
            @test claim1 !== nothing
            EDDPotentials.hpc_mark_training_slot_failed!(
                builder,
                0,
                claim1[1],
                "simulated failure",
            )
            claim2 = EDDPotentials.hpc_claim_training_slot!(builder, 0, "worker-2")
            @test claim2 !== nothing
            @test claim2[1] == claim1[1]
        end
    end

    @testset "HPC scheduler script rendering" begin
        mktempdir() do temp
            script = joinpath(temp, "hpc_job.sh")
            open(script, "w") do io
                println(io, "#!/bin/bash")
                println(io, "{{EDDP_HPC_CLIENT_CMD}}")
            end

            state = EDDPotentials.BuilderState(
                seedfile="seed.cell",
                seedfile_calc="calc.cell",
                workdir=temp,
            )
            cf = EDDPotentials.CellFeatureConfig(elements=["H"])
            rss = EDDPotentials.RssSetting()
            trainer = EDDPotentials.TrainingOption()
            hpc = EDDPotentials.HPCWorkflowOption()
            scheduler = EDDPotentials.SchedulerConfig(
                script=script,
                njobs=2,
                type="SLURM",
                submit_jobs=false,
            )
            opt = EDDPotentials.BuilderOption(state, cf, nothing, rss, trainer, scheduler, hpc)
            builder = Builder(opt)
            builder.state.builder_file_path = joinpath(temp, "link.toml")

            rendered = EDDPotentials._render_hpc_client_job_script(
                builder,
                scheduler,
                joinpath(temp, "submit"),
            )
            text = read(rendered, String)
            @test !contains(text, "{{EDDP_HPC_CLIENT_CMD}}")
            @test contains(text, "EDDPotentials.hpc_client()")
            @test contains(text, "--worker-id \"\$WORKER_ID\"")
            @test contains(text, "SLURM_ARRAY_TASK_ID")
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
