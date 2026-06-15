using EDDPotentials
using EDDPotentials.Workflow:
    TRAINING_DIR,
    LocalLinkBackend,
    generated_input_count,
    required_success_count,
    validate_dft_outputs!,
    validate_generated_structures!,
    is_generation_data_ready,
    is_training_dataset_ready,
    run_dft_phase!,
    run_training_phase!,
    load_training_data,
    _render_training_job_script,
    _training_submit_cmd,
    _required_models_for_ensemble,
    _generate_random_structures,
    _models_ready_for_ensemble
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
        state.rss_require_converged = true

        cf = EDDPotentials.CellFeatureConfig(elements=["H"])

        rss = EDDPotentials.RssSetting()
        rss.prune.enabled = true
        rss.prune.pool_size = 50
        rss.prune.keep_fraction = 0.3
        rss.require_converged = true

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
        distributed.dft_env = Dict{String,Any}("OMP_NUM_THREADS" => "1")
        distributed.training_env = Dict{String,Any}(
            "OMP_NUM_THREADS" => 16,
            "EDDP_USE_GPU" => false,
        )
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
                _render_training_job_script(builder, scheduler, workdir)
            rendered = read(rendered_script, String)

            @test !contains(rendered, "{{EDDP_TRAIN_CMD}}")
            @test contains(rendered, "EDDPotentials.run_trainer()")
            @test contains(rendered, "--id \"\$WORKER_ID\"")
            @test contains(rendered, "SLURM_ARRAY_TASK_ID")

            slurm_cmd = _training_submit_cmd(scheduler, workdir)
            @test contains(string(slurm_cmd), "sbatch")
            @test contains(string(slurm_cmd), "--array=1-4")
            @test contains(string(slurm_cmd), "job.sh")

            sge_scheduler = EDDPotentials.SchedulerConfig(
                script=script,
                njobs=4,
                type="SGE",
                submit_jobs=false,
            )
            sge_cmd = _training_submit_cmd(sge_scheduler, workdir)
            @test contains(string(sge_cmd), "qsub")
            @test contains(string(sge_cmd), "1-4")
            @test contains(string(sge_cmd), "-cwd")

            bad_script = joinpath(temp, "bad_train_job.sh")
            write(bad_script, "#!/bin/bash\n")
            bad_scheduler = EDDPotentials.SchedulerConfig(script=bad_script, type="SLURM")
            @test_throws ArgumentError _render_training_job_script(
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

            @test generated_input_count(builder, 0) == 3
            @test required_success_count(3, 1.0) == 3
            @test !EDDPotentials.is_training_data_ready(builder, 0)
            @test_throws ErrorException validate_dft_outputs!(builder, 0)

            write(joinpath(dft0, "s3.res"), "TITL test\nEND\n")
            @test EDDPotentials.is_training_data_ready(builder, 0)
            @test validate_dft_outputs!(builder, 0)
        end

        @test !(:hyper_ndim in fieldnames(EDDPotentials.BuilderState))
        @test EDDPotentials.BuilderState(seedfile="seed.cell", seedfile_calc="calc.cell").per_generation_threshold == 0.7

        trainer = EDDPotentials.TrainingOption()
        trainer.nmodels = 10
        trainer.external = true
        trainer.external_completion_ratio = 0.5
        @test _required_models_for_ensemble(trainer) == 5

        distributed = EDDPotentials.DistributedLinkOption()
        distributed.enabled = true
        trainer.external = false
        builder = Builder(
            EDDPotentials.BuilderOption(
                EDDPotentials.BuilderState(seedfile="seed.cell", seedfile_calc="calc.cell"),
                EDDPotentials.CellFeatureConfig(elements=["H"]),
                nothing,
                EDDPotentials.RssSetting(),
                trainer,
                nothing,
                distributed,
            ),
        )
        @test _required_models_for_ensemble(builder) == 5
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

            @test is_generation_data_ready(builder, 0)
            @test _generate_random_structures(builder, 0) == :ok
            @test validate_generated_structures!(builder, 0)
            @test length(readdir(gen0)) == 3

            dft_result = run_dft_phase!(LocalLinkBackend(), builder, 0)
            @test dft_result.status == :ok
            @test dft_result.message == "DFT outputs already ready"

            training_dir = joinpath(temp, TRAINING_DIR)
            mkpath(training_dir)
            for i = 1:2
                write(joinpath(training_dir, "gen-0-model-$i.json"), "{}")
            end
            dataset_path = EDDPotentials.dataset_name(builder, 0)
            mkpath(dirname(dataset_path))
            write(dataset_path, "placeholder")
            @test is_training_dataset_ready(builder, 0)
            sleep(1.1)
            write(joinpath(dft0, "s3.res"), "TITL updated\nEND\n")
            @test !is_training_dataset_ready(builder, 0)
            sleep(1.1)
            write(dataset_path, "placeholder")
            @test is_training_dataset_ready(builder, 0)
            @test _models_ready_for_ensemble(builder)
            train_result = run_training_phase!(LocalLinkBackend(), builder, 0)
            @test train_result.status == :ok
            @test train_result.message == "models already ready"
        end
    end

    @testset "acrud dispatch uses airsspy CLI" begin
        mktempdir() do temp
            fake_ap = joinpath(temp, "ap")
            args_file = joinpath(temp, "ap-args.txt")
            open(fake_ap, "w") do io
                println(io, "#!/bin/sh")
                println(io, "printf '%s\\n' \"\$@\" > '$(args_file)'")
                println(io, "workdir=''")
                println(io, "prev=''")
                println(io, "for arg in \"\$@\"; do")
                println(io, "  if [ \"\$prev\" = '--workdir' ]; then workdir=\"\$arg\"; fi")
                println(io, "  prev=\"\$arg\"")
                println(io, "done")
                println(io, "mkdir -p \"\$workdir/good_castep\"")
                println(io, "for file in \"\$workdir\"/hopper/*.res; do")
                println(io, "  [ -e \"\$file\" ] || continue")
                println(io, "  cp \"\$file\" \"\$workdir/good_castep/\$(basename \"\$file\")\"")
                println(io, "done")
            end
            chmod(fake_ap, 0o755)

            write(joinpath(temp, "calc.cell"), "%BLOCK LATTICE_CART\n1 0 0\n0 1 0\n0 0 1\n%ENDBLOCK LATTICE_CART\n")
            write(joinpath(temp, "calc.param"), "task : SinglePoint\n")
            indir = joinpath(temp, "input")
            outdir = joinpath(temp, "output")
            mkpath(indir)
            write(joinpath(indir, "calc-001.res"), "TITL test\nEND\n")

            state = EDDPotentials.BuilderState(
                seedfile="seed.cell",
                seedfile_calc="calc.cell",
                workdir=temp,
            )
            state.dft_mode = "acrud"
            state.dft_kwargs = Dict{String,Any}(
                "ap_cmd" => fake_ap,
                "code" => "castep",
                "mpinp" => 2,
                "singlepoint" => true,
            )
            builder = Builder(
                EDDPotentials.BuilderOption(
                    state,
                    EDDPotentials.CellFeatureConfig(elements=["H"]),
                    nothing,
                    EDDPotentials.RssSetting(),
                    EDDPotentials.TrainingOption(),
                    nothing,
                ),
            )

            @test EDDPotentials.get_external_calculator(builder)(indir, outdir)
            args = read(args_file, String)
            @test contains(args, "run\ncrud")
            @test contains(args, "--code\ncastep")
            @test contains(args, "--mpinp\n2")
            @test contains(args, "--singlepoint")
            @test isfile(joinpath(temp, "acrud-work", "calc.cell"))
            @test isfile(joinpath(temp, "acrud-work", "calc.param"))
            @test isfile(joinpath(outdir, "calc-001.res"))
        end
    end

    @testset "airsspy acrud result normalization removes stress-only RES tokens" begin
        mktempdir() do temp
            res = joinpath(temp, "airsspy.res")
            write(
                res,
                """
                TITL C-001  64.533 14.702 -10.112719  0  0  2 (P-1) n - 1
                CELL 1.0 2.083940 2.382460 3.275690 74.203670 77.147910 71.863990 -0.700829 -0.298584 1.304074
                LATT -1 0.700829 0.298584 -1.304073
                SFAC C
                C 1 0.802278 0.520067 0.075267 1.0
                C 1 0.128022 0.881055 0.109820 1.0
                END
                """,
            )
            EDDPotentials.normalize_airsspy_res_file!(res)
            text = read(res, String)
            @test occursin("CELL 1.0 2.083940 2.382460 3.275690 74.203670 77.147910 71.863990\n", text)
            @test occursin("LATT -1\n", text)
            @test EDDPotentials.read_res(res).metadata[:enthalpy] ≈ -10.112719
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
            train, val, test = load_training_data(dataset_path)
            train_parents, val_parents, test_parents = _builder_parent_sets(train, val, test)

            @test isempty(intersect(train_parents, val_parents))
            @test isempty(intersect(train_parents, test_parents))
            @test isempty(intersect(val_parents, test_parents))
            @test isfile(splitext(dataset_path)[1] * ".yaml")
        end
    end

    @testset "Builder train_split accepts ratio-like weights" begin
        mktempdir() do temp
            dft_dir = joinpath(temp, "gen0-dft")
            mkpath(dft_dir)
            for i = 1:20
                write(
                    joinpath(dft_dir, "s$(i).res"),
                    """
                    TITL s$(i)  0.000 15.000 $(Float64(i))  0  0  2 (P-1) n - 1
                    CELL 1.0 3.000000 3.000000 3.000000 90.000000 90.000000 90.000000
                    LATT -1
                    SFAC H
                    H 1 0.000000 0.000000 0.000000 1.0
                    H 1 0.500000 0.500000 0.500000 1.0
                    END
                    """,
                )
            end
            state = EDDPotentials.BuilderState(
                seedfile="seed.cell",
                seedfile_calc="calc.cell",
                workdir=temp,
            )
            cf = EDDPotentials.CellFeatureConfig(elements=["H"])
            rss = EDDPotentials.RssSetting()
            trainer = EDDPotentials.TrainingOption()
            trainer.train_split = [8.0, 1.0, 1.0]
            opt = EDDPotentials.BuilderOption(state, cf, nothing, rss, trainer, nothing)
            builder = Builder(opt)
            dataset_path = joinpath(temp, "training", "ratio-dataset.jld2")

            EDDPotentials.write_dataset(builder; name=dataset_path)
            train, val, test = load_training_data(dataset_path)

            @test length(train) == 16
            @test length(val) == 2
            @test length(test) == 2
        end
    end
end
