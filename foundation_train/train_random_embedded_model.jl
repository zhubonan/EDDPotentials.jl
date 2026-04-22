using EDDPotentials
using EDDPotentials:
    FeatureContainer,
    FeatureStore,
    RandomEmbeddedCellFeature,
    MultiRandomEmbeddedCellFeature,
    StructureContainer,
    TrainingResults,
    ginit,
    mae_per_atom,
    nfeatures,
    rmse_per_atom,
    save_feature_store,
    save_feature_store_streaming,
    load_feature_store,
    train!
using EDDPotentials: _prescan_extxyz
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
        "n_multi" => 1,
        "hidden_size" => [10, 10, 10],
        "p2" => [2,4,6,8,10],
        "p3" => [2,4,6,8,10],
        "q3" => [3,5,7,9],
        "val_ratio" => 0.1,
        "train_method" => "lm_batched",
        "batch_size" => 1024,
        "max_iter" => 20,
        "earlystop" => 20,
        "seed" => 42,
        "show_progress" => true,
        "threshold" => 10.0,
        "feature_store" => "",
        "threaded" => true,
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
        elseif key == "n_multi"
            config["n_multi"] = parse(Int, value)
        elseif key == "hidden_size"
            config["hidden_size"] = parse(Int, value)
        elseif key == "p2"
            config["p2"] = _parse_range(value)
        elseif key == "p3"
            config["p3"] = _parse_range(value)
        elseif key == "q3"
            config["q3"] = _parse_range(value)
        elseif key in ("val_ratio", "test_ratio")
            config["val_ratio"] = parse(Float64, value)
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
        elseif key in ("feature_store", "store", "save_features")
            config["feature_store"] = value
        elseif key in ("threaded", "threads")
            config["threaded"] = _parse_bool(value)
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
        "n_multi=$(cfg["n_multi"]) " *
        "hidden_size=$(cfg["hidden_size"]) p2=$(cfg["p2"]) p3=$(cfg["p3"]) q3=$(cfg["q3"]) " *
        "train_method=$(cfg["train_method"]) batch_size=$(cfg["batch_size"]) val_ratio=$(cfg["val_ratio"]) " *
        "max_iter=$(cfg["max_iter"]) earlystop=$(cfg["earlystop"]) threaded=$(cfg["threaded"]) " *
        "show_progress=$(cfg["show_progress"]) " *
        "feature_store=$(isempty(cfg["feature_store"]) ? "(in-memory)" : cfg["feature_store"])",
    )
    max_structures = cfg["max_structures"]
    dataset_path = if max_structures > 0 && endswith(lowercase(dataset), ".xyz") || endswith(lowercase(dataset), ".extxyz")
        "$(dataset)@:$(max_structures)"
    else
        dataset
    end

    store_path = cfg["feature_store"]
    use_store = !isempty(store_path)

    if use_store
        # --- Streaming path: no full FeatureContainer in memory ---
        # Get species from lightweight prescan (no Cell construction)
        _, _, symbols = _prescan_extxyz(dataset_path; threshold=cfg["threshold"],
                                        max_structures=cfg["max_structures"])
    else
        sc = StructureContainer(dataset_path; threshold=cfg["threshold"])
        println("Loaded $(length(sc)) structures after threshold filtering.")
        symbols = _unique_symbols(sc.structures)
    end

    n_multi = cfg["n_multi"]
    embedded = if n_multi > 1
        MultiRandomEmbeddedCellFeature(
            symbols;
            nembed=cfg["embedding_size"],
            n_multi=n_multi,
            seeds=collect(0:n_multi-1) .+ cfg["seed"],
            p2=cfg["p2"],
            p3=cfg["p3"],
            q3=cfg["q3"],
        )
    else
        RandomEmbeddedCellFeature(
            symbols;
            nembed=cfg["embedding_size"],
            seed=cfg["seed"],
            p2=cfg["p2"],
            p3=cfg["p3"],
            q3=cfg["q3"],
        )
    end
    println("Embedding feature size: $(nfeatures(embedded))")

    if use_store
        # Streaming: compute features chunk-by-chunk, write to disk
        println("Streaming features to FeatureStore: $store_path")
        train_data, fc_val = save_feature_store_streaming(
            dataset_path, embedded, store_path;
            batch_size=cfg["batch_size"],
            val_ratio=cfg["val_ratio"],
            seed=cfg["seed"],
            threshold=cfg["threshold"],
            max_structures=cfg["max_structures"],
            show_progress=cfg["show_progress"],
        )
        xt, yt = train_data.xt, train_data.yt
    else
        # In-memory path
        fc = FeatureContainer(sc, embedded; show_progress=true)
        nval = max(1, Int(floor(length(fc) * cfg["val_ratio"])))
        ntrain = length(fc) - nval
        ntrain > 0 || error("Not enough structures after filtering to create a train/val split.")
        fc_train, fc_val = split(
            fc,
            ntrain,
            nval;
            shuffle=true,
            seed=cfg["seed"],
            standardize=true,
        )
        train_data = fc_train
        xt, yt = fc_train.xt, fc_train.yt
    end

    nh = cfg["hidden_size"]
    model = ManualFluxBackPropInterface(
        embedded,
        nh...;
        init=ginit,
        σ=tanh,
        xt=xt,
        yt=yt,
    )
    println("Model parameter count: $(nparams(model))")

    println("Training with $(length(train_data)) train / $(length(fc_val)) val structures.")
    train_kwargs = (
        train_method=cfg["train_method"],
        max_epochs=cfg["max_iter"],
        show_progress=cfg["show_progress"],
        earlystop=cfg["earlystop"],
    )
    if cfg["train_method"] == "lm_batched"
        train!(
            model,
            train_data,
            fc_val;
            train_kwargs...,
            batch_size=cfg["batch_size"],
            threaded=cfg["threaded"],
        )
    else
        train!(model, train_data, fc_val; train_kwargs...)
    end

    val_results = TrainingResults(model, fc_val)

    println()
    if !use_store
        train_results = TrainingResults(model, fc_train)
        println("Train RMSE/atom: $(rmse_per_atom(train_results)) eV")
        println("Train MAE/atom : $(mae_per_atom(train_results)) eV")
    end
    println("Val RMSE/atom  : $(rmse_per_atom(val_results)) eV")
    println("Val MAE/atom   : $(mae_per_atom(val_results)) eV")
end

main()
