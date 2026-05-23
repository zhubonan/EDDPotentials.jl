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



        state = EDDPotentials.BuilderState(
            seedfile_calc="calc.cell",
            rss=EDDPotentials.RssSetting(
                seedfile="seed.cell",
                pressure_gpa=1.0,
                nprocs=4,
                ensemble_std_max=0.2,
                formula_sampling_enabled=true,
                formula_list=["H2"],
            ),
        )
        state.rss.prune.enabled = true
        state.rss.prune.pool_size = 25
        state.rss.prune.keep_fraction = 0.2

        cf = EDDPotentials.CellFeatureConfig(elements=["H"])

        rss = EDDPotentials.RssSetting(seedfile="prod.cell")
        rss.prune.enabled = true
        rss.prune.pool_size = 50
        rss.prune.keep_fraction = 0.3

        opt = EDDPotentials.BuilderOption(state, cf, nothing, rss, EDDPotentials.TrainingOption(), nothing)
        test_to_from_toml(opt)

        builder = Builder(opt)
        @test builder.state.rss.ensemble_std_min == -1.0
        @test EDDPotentials.RssSetting().packed == false
        @test EDDPotentials.RssSetting().pressure_gpa == 0.01
        @test EDDPotentials.RssSetting().ensemble_std_min == -1.0
        @test EDDPotentials.RssSetting().ensemble_std_max == -1.0
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

    @testset "Legacy flat RSS settings migrate to state.rss" begin
        state = EDDPotentials.BuilderState(
            seedfile="legacy.cell",
            seedfile_calc="calc.cell",
            rss_pressure_gpa=2.0,
            rss_external=false,
            rss_nprocs=3,
            rss_num_threads=2,
            core_size=1.2,
            ensemble_std_min=0.01,
            ensemble_std_max=0.2,
            formula_sampling_enabled=true,
            formula_list=["H2"],
            relax_mode="hot",
            hot_n_rounds=4,
        )
        cf = EDDPotentials.CellFeatureConfig(elements=["H"])
        opt = EDDPotentials.BuilderOption(
            state,
            cf,
            nothing,
            EDDPotentials.RssSetting(),
            EDDPotentials.TrainingOption(),
            nothing,
        )
        builder = @test_logs (
            :warn,
            r"Deprecated flat BuilderState RSS settings were used",
        ) match_mode=:any Builder(opt)
        @test builder.state.rss.seedfile == "legacy.cell"
        @test builder.state.rss.pressure_gpa == 2.0
        @test builder.state.rss.external == false
        @test builder.state.rss.nprocs == 3
        @test builder.state.rss.num_threads == 2
        @test builder.state.rss.core_size == 1.2
        @test builder.state.rss.ensemble_std_min == 0.01
        @test builder.state.rss.ensemble_std_max == 0.2
        @test builder.state.rss.formula_sampling_enabled
        @test builder.state.rss.formula_list == ["H2"]
        @test builder.state.rss.relax_mode == "hot"
        @test builder.state.rss.hot_n_rounds == 4
        @test builder.rss.seedfile == "legacy.cell"
        @test builder.rss.core_size == 0.5
        @test !builder.rss.formula_sampling_enabled
        @test isempty(builder.rss.formula_list)
        @test builder.rss.relax_mode == "standard"
    end

    @testset "Nested state.rss preserves explicit values from TOML" begin
        mktempdir() do temp
            path = joinpath(temp, "link.toml")
            write(
                path,
                """
                [state]
                seedfile_calc = "calc.cell"

                [state.rss]
                seedfile = "seed.cell"
                pressure_gpa = 0.01
                ensemble_std_min = -1.0
                ensemble_std_max = 0.2

                [cf]
                elements = ["H"]

                [rss]
                seedfile = "prod.cell"

                [trainer]
                """,
            )
            builder = Builder(path)
            @test builder.state.rss.pressure_gpa == 0.01
            @test builder.state.rss.ensemble_std_min == -1.0
            @test builder.state.rss.ensemble_std_max == 0.2
        end
    end

    @testset "Default-valued flat RSS keys warn from TOML" begin
        mktempdir() do temp
            path = joinpath(temp, "link.toml")
            write(
                path,
                """
                [state]
                seedfile_calc = "calc.cell"
                rss_nprocs = 2

                [state.rss]
                seedfile = "seed.cell"

                [cf]
                elements = ["H"]

                [rss]
                seedfile = "prod.cell"

                [trainer]
                """,
            )
            @test_logs (
                :warn,
                r"Deprecated flat BuilderState RSS settings were used",
            ) match_mode=:any Builder(path)
        end
    end

    @testset "Local backend rejects obsolete execution modes" begin
        mktempdir() do temp
            state = EDDPotentials.BuilderState(
                seedfile="seed.cell",
                seedfile_calc="calc.cell",
                workdir=temp,
            )
            cf = EDDPotentials.CellFeatureConfig(elements=["H"])
            rss = EDDPotentials.RssSetting()
            trainer = EDDPotentials.TrainingOption()
            opt = EDDPotentials.BuilderOption(state, cf, nothing, rss, trainer, nothing)

            trainer.external = false
            builder = Builder(opt)
            @test_throws ArgumentError EDDPotentials.validate_builder_for_link!(
                builder,
                LocalLinkBackend(),
            )

            trainer.external = true
            trainer.external_mode = "submit"
            builder = Builder(opt)
            @test_throws ArgumentError EDDPotentials.validate_builder_for_link!(
                builder,
                LocalLinkBackend(),
            )

            trainer.external_mode = "command"
            builder = Builder(opt)
            @test_throws ArgumentError EDDPotentials.validate_builder_for_link!(
                builder,
                LocalLinkBackend(),
            )

            trainer.external_mode = "local"
            state.dft_mode = "crud-queue"
            builder = Builder(opt)
            @test_throws ArgumentError EDDPotentials.validate_builder_for_link!(
                builder,
                LocalLinkBackend(),
            )

            state.dft_mode = "disp-castep"
            builder = Builder(opt)
            @test_throws ArgumentError EDDPotentials.validate_builder_for_link!(
                builder,
                LocalLinkBackend(),
            )

            state.dft_mode = "crud"
            trainer.external = false
            builder = Builder(opt)
            @test_throws ArgumentError run_training_phase!(
                LocalLinkBackend(),
                builder,
                0,
            )

            trainer.external = true
            trainer.external_mode = "submit"
            builder = Builder(opt)
            @test_throws ArgumentError run_training_phase!(
                LocalLinkBackend(),
                builder,
                0,
            )

            trainer.external_mode = "local"
            state.dft_mode = "crud-queue"
            builder = Builder(opt)
            @test_throws ArgumentError run_dft_phase!(
                LocalLinkBackend(),
                builder,
                0,
            )

            state.dft_mode = "disp-castep"
            builder = Builder(opt)
            @test_throws ArgumentError run_dft_phase!(
                LocalLinkBackend(),
                builder,
                0,
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
        trainer.external = false
        @test _required_models_for_ensemble(trainer) == 5
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
end
