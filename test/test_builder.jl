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
