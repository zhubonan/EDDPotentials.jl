using EDDPotentials
using EDDPotentials:
    FeatureContainer,
    RandomEmbeddedCellFeature,
    StructureContainer,
    TrainingResults,
    ginit,
    mae_per_atom,
    nfeatures,
    rmse_per_atom,
    train!
using EDDPotentials.EDDPotentialsCore: ManualFluxBackPropInterface, nparams
using CellBase: species

const DEFAULT_DATASET = joinpath(
    @__DIR__,
    "mp_finetuning-MACE_matpes_ft_run-42_combined.xyz",
)

function _parse_bool(value::AbstractString)
    lowercase(value) in ("1", "true", "yes", "y", "on")
end

function _parse_range(value::AbstractString)
    parts = split(value, ":")
    if length(parts) == 1
        n = parse(Int, parts[1])
        return n:n
    elseif length(parts) == 2
        return parse(Int, parts[1]):parse(Int, parts[2])
    end
    error("Expected range in `start:end` form, got `$value`.")
end

function _parse_args(args)
    config = Dict{String,Any}(
        "dataset" => DEFAULT_DATASET,
        "max_structures" => 10000,
        "embedding_size" => 4,
        "hidden_size" => 8,
        "p2" => [2,4,6,8,10],
        "p3" => [2,4,6,8,10],
        "q3" => [3,5,7,9],
        "test_ratio" => 0.1,
        "train_method" => "lm_batched",
        "batch_size" => 1024,
        "max_iter" => 20,
        "earlystop" => 5,
        "seed" => 42,
        "show_progress" => true,
        "threshold" => 10.0,
    )

    for arg in args
        occursin("=", arg) || error("Expected key=value arguments, got `$arg`.")
        key, value = split(arg, "="; limit=2)
        if key in ("dataset", "data")
            config["dataset"] = value
        elseif key == "max_structures"
            config["max_structures"] = parse(Int, value)
        elseif key == "embedding_size"
            config["embedding_size"] = parse(Int, value)
        elseif key == "hidden_size"
            config["hidden_size"] = parse(Int, value)
        elseif key == "p2"
            config["p2"] = _parse_range(value)
        elseif key == "p3"
            config["p3"] = _parse_range(value)
        elseif key == "q3"
            config["q3"] = _parse_range(value)
        elseif key == "test_ratio"
            config["test_ratio"] = parse(Float64, value)
        elseif key == "train_method"
            config["train_method"] = value
        elseif key == "batch_size"
            config["batch_size"] = parse(Int, value)
        elseif key in ("max_iter", "maxIter")
            config["max_iter"] = parse(Int, value)
        elseif key == "earlystop"
            config["earlystop"] = parse(Int, value)
        elseif key == "seed"
            config["seed"] = parse(Int, value)
        elseif key == "show_progress"
            config["show_progress"] = _parse_bool(value)
        elseif key == "threshold"
            config["threshold"] = parse(Float64, value)
        else
            error("Unknown argument `$key`.")
        end
    end

    config
end

function _unique_symbols(structures)
    symbols = Set{Symbol}()
    for structure in structures
        union!(symbols, Set(species(structure)))
    end
    sort!(collect(symbols))
end

function main(args=ARGS)
    cfg = _parse_args(args)
    dataset = cfg["dataset"]

    println("Loading dataset: $dataset")
    println(
        "Config: max_structures=$(cfg["max_structures"]) embedding_size=$(cfg["embedding_size"]) " *
        "hidden_size=$(cfg["hidden_size"]) p2=$(cfg["p2"]) p3=$(cfg["p3"]) q3=$(cfg["q3"]) " *
        "train_method=$(cfg["train_method"]) batch_size=$(cfg["batch_size"]) test_ratio=$(cfg["test_ratio"]) " *
        "max_iter=$(cfg["max_iter"]) earlystop=$(cfg["earlystop"]) " *
        "show_progress=$(cfg["show_progress"])",
    )
    max_structures = cfg["max_structures"]
    dataset_path = if max_structures > 0 && endswith(lowercase(dataset), ".xyz") || endswith(lowercase(dataset), ".extxyz")
        "$(dataset)@:$(max_structures)"
    else
        dataset
    end
    sc = StructureContainer(dataset_path; threshold=cfg["threshold"])
    println("Loaded $(length(sc)) structures after threshold filtering.")

    symbols = _unique_symbols(sc.structures)
    embedded = RandomEmbeddedCellFeature(
        symbols;
        nembed=cfg["embedding_size"],
        seed=cfg["seed"],
        p2=cfg["p2"],
        p3=cfg["p3"],
        q3=cfg["q3"],
    )
    println("Embedding feature size: $(nfeatures(embedded))")

    fc = FeatureContainer(sc, embedded; show_progress=true)
    ntest = max(1, Int(floor(length(fc) * cfg["test_ratio"])))
    ntrain = length(fc) - ntest
    ntrain > 0 || error("Not enough structures after filtering to create a train/test split.")
    fc_train, fc_test = split(
        fc,
        ntrain,
        ntest;
        shuffle=true,
        seed=cfg["seed"],
        standardize=true,
        apply_transform=true,
    )

    nh = cfg["hidden_size"]
    model = ManualFluxBackPropInterface(
        embedded,
        nh;
        init=ginit,
        σ=tanh,
        xt=fc_train.xt,
        yt=fc_train.yt,
        apply_xt=false,
    )
    println("Model parameter count: $(nparams(model))")

    println("Training with $(length(fc_train)) train / $(length(fc_test)) test structures.")
    train_kwargs = (
        train_method=cfg["train_method"],
        maxIter=cfg["max_iter"],
        show_progress=cfg["show_progress"],
        earlystop=cfg["earlystop"],
    )
    if cfg["train_method"] == "lm_batched"
        train!(
            model,
            fc_train,
            fc_test;
            train_kwargs...,
            batch_size=cfg["batch_size"],
        )
    else
        train!(model, fc_train, fc_test; train_kwargs...)
    end

    train_results = TrainingResults(model, fc_train)
    test_results = TrainingResults(model, fc_test)

    println()
    println("Train RMSE/atom: $(rmse_per_atom(train_results)) eV")
    println("Train MAE/atom : $(mae_per_atom(train_results)) eV")
    println("Test RMSE/atom : $(rmse_per_atom(test_results)) eV")
    println("Test MAE/atom  : $(mae_per_atom(test_results)) eV")
end

main()
