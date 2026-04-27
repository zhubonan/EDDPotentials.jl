#=
Train an EDDP model using SGD/Adam optimizer.

Usage:
  julia --project=.. foundation_train/train_sgd_model.jl [key=value ...]

Examples:
  # Adam with default settings
  julia --project=.. foundation_train/train_sgd_model.jl

  # SGD with momentum and cosine annealing
  julia --project=.. foundation_train/train_sgd_model.jl optimizer=sgd_momentum lr=0.01 momentum=0.9 lr_schedule=cosine

  # Adam with weight decay and custom schedule
  julia --project=.. foundation_train/train_sgd_model.jl optimizer=adam lr=0.001 weight_decay=1e-4 lr_schedule=step

  # Use a specific dataset
  julia --project=.. foundation_train/train_sgd_model.jl dataset=my_data.xyz max_structures=5000
=#

using EDDPotentials
using EDDPotentials:
    FeatureStore,
    RandomEmbeddedCellFeature,
    MultiRandomEmbeddedCellFeature,
    StructureContainer,
    TrainingResults,
    ginit,
    get_elemental_energy,
    mae_per_atom,
    nfeatures,
    rmse_per_atom,
    save_feature_store_streaming,
    split_and_featurize,
    train!,
    train_sgd!,
    constant_lr,
    step_decay_lr,
    cosine_annealing_lr
using EDDPotentials: _prescan_extxyz
using EDDPotentials.EDDPotentialsCore: ManualFluxBackPropInterface, nparams
using CellBase: species
using JSON

const DEFAULT_DATASET = joinpath(
    @__DIR__,
    "matpes-r2scan-replay-data.extxyz"
)
const DEFAULT_E0S = joinpath(@__DIR__, "e0s.json")

"""
    load_e0s([path]) -> Dict{Symbol,Float64}

Load elemental reference energies from a JSON file.
Each key is an element symbol, each value is the reference energy in eV.
Returns an empty Dict if `path` is `""` or `"none"`.
"""
function load_e0s(path::AbstractString=DEFAULT_E0S)
    (path == "" || lowercase(path) == "none") && return Dict{Symbol,Float64}()
    isfile(path) || error("Elemental energies file not found: `$path`")
    data = JSON.parsefile(path)
    Dict{Symbol,Float64}(Symbol(k) => Float64(v) for (k, v) in data)
end

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
        # Data
        "dataset" => DEFAULT_DATASET,
        "max_structures" => 10000,
        "val_ratio" => 0.1,
        "threshold" => 10.0,
        "seed" => 42,
        "feature_store" => "",
        "e0s" => DEFAULT_E0S,
        # Features
        "embedding_size" => 4,
        "n_multi" => 1,
        "hidden_size" => [128, 30, 30],
        "p2" => [2, 4, 6, 8, 10],
        "p3" => [2, 4, 6, 8, 10],
        "q3" => [3, 5, 7, 9],
        # Optimizer
        "optimizer" => :adam,
        "lr" => 1e-3,
        "momentum" => 0.9,
        "weight_decay" => 0.0,
        "grad_clip" => 1.0,
        "beta1" => 0.9,
        "beta2" => 0.999,
        # Schedule
        "lr_schedule" => "cosine",   # constant, step, cosine
        "lr_min" => 1e-5,              # for cosine
        "lr_step_size" => 100,         # for step
        "lr_step_factor" => 0.5,       # for step
        # Training
        "batch_size" => 32,
        "max_epochs" => 300,
        "earlystop" => 50,
        "show_progress" => true,
        "checkpoint_every" => 0,
        "checkpoint_prefix" => "",
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
        elseif key == "threshold"
            config["threshold"] = parse(Float64, value)
        elseif key == "seed"
            config["seed"] = parse(Int, value)
        elseif key in ("feature_store", "store", "save_features")
            config["feature_store"] = value
        elseif key in ("e0s", "elemental_energies", "e0")
            config["e0s"] = value
        elseif key == "optimizer"
            config["optimizer"] = Symbol(value)
        elseif key == "lr"
            config["lr"] = parse(Float64, value)
        elseif key == "momentum"
            config["momentum"] = parse(Float64, value)
        elseif key == "weight_decay"
            config["weight_decay"] = parse(Float64, value)
        elseif key == "grad_clip"
            config["grad_clip"] = parse(Float64, value)
        elseif key == "beta1"
            config["beta1"] = parse(Float64, value)
        elseif key == "beta2"
            config["beta2"] = parse(Float64, value)
        elseif key == "lr_schedule"
            config["lr_schedule"] = value
        elseif key == "lr_min"
            config["lr_min"] = parse(Float64, value)
        elseif key == "lr_step_size"
            config["lr_step_size"] = parse(Int, value)
        elseif key == "lr_step_factor"
            config["lr_step_factor"] = parse(Float64, value)
        elseif key == "batch_size"
            config["batch_size"] = parse(Int, value)
        elseif key in ("max_epochs", "max_iter")
            config["max_epochs"] = parse(Int, value)
        elseif key == "earlystop"
            config["earlystop"] = parse(Int, value)
        elseif key == "show_progress"
            config["show_progress"] = _parse_bool(value)
        elseif key == "checkpoint_every"
            config["checkpoint_every"] = parse(Int, value)
        elseif key == "checkpoint_prefix"
            config["checkpoint_prefix"] = value
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

function _build_schedule(cfg)
    sched = cfg["lr_schedule"]
    if sched == "constant"
        return nothing  # train_sgd! defaults to constant_lr(cfg["lr"])
    elseif sched == "step"
        return step_decay_lr(
            cfg["lr"];
            step_size=cfg["lr_step_size"],
            factor=cfg["lr_step_factor"],
        )
    elseif sched == "cosine"
        return cosine_annealing_lr(cfg["lr"], cfg["lr_min"], cfg["max_epochs"])
    else
        error("Unknown lr_schedule: `$sched`. Use: constant, step, cosine")
    end
end

function main(args=ARGS)
    cfg = _parse_args(args)
    dataset = cfg["dataset"]

    println("Loading dataset: $dataset")
    println(
        "Config: max_structures=$(cfg["max_structures"]) embedding_size=$(cfg["embedding_size"]) " *
        "n_multi=$(cfg["n_multi"]) " *
        "hidden_size=$(cfg["hidden_size"]) p2=$(cfg["p2"]) p3=$(cfg["p3"]) q3=$(cfg["q3"]) " *
        "optimizer=$(cfg["optimizer"]) lr=$(cfg["lr"]) lr_schedule=$(cfg["lr_schedule"]) " *
        "batch_size=$(cfg["batch_size"]) val_ratio=$(cfg["val_ratio"]) " *
        "max_epochs=$(cfg["max_epochs"]) earlystop=$(cfg["earlystop"]) " *
        "weight_decay=$(cfg["weight_decay"]) " *
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

    e0s = load_e0s(cfg["e0s"])
    if !isempty(e0s)
        println("Elemental reference energies: $(length(e0s)) elements loaded from $(cfg["e0s"])")
    end

    if use_store
        println("Streaming features to FeatureStore: $store_path")
        train_data, fc_val = save_feature_store_streaming(
            dataset_path, embedded, store_path;
            batch_size=cfg["batch_size"],
            val_ratio=cfg["val_ratio"],
            seed=cfg["seed"],
            threshold=cfg["threshold"],
            max_structures=cfg["max_structures"],
            show_progress=cfg["show_progress"],
            elemental_energies=e0s,
        )
        xt, yt = train_data.xt, train_data.yt
    else
        # In-memory path: split structures first, then featurize each split
        nval = max(1, Int(floor(length(sc) * cfg["val_ratio"])))
        ntrain = length(sc) - nval
        ntrain > 0 || error("Not enough structures after filtering to create a train/val split.")
        fc_train, fc_val = split_and_featurize(
            sc,
            embedded,
            ntrain,
            nval;
            shuffle=true,
            seed=cfg["seed"],
            show_progress=true,
            elemental_energies=e0s,
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

    # Build learning rate schedule
    schedule = _build_schedule(cfg)

    println("Training with $(length(train_data)) train / $(length(fc_val)) val structures.")
    train!(
        model,
        train_data,
        fc_val;
        train_method="sgd",
        optimizer=cfg["optimizer"],
        lr=cfg["lr"],
        lr_schedule=schedule,
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
        grad_clip=cfg["grad_clip"],
        beta=(cfg["beta1"], cfg["beta2"]),
        batch_size=cfg["batch_size"],
        max_epochs=cfg["max_epochs"],
        earlystop=cfg["earlystop"],
        show_progress=cfg["show_progress"],
        checkpoint_every=cfg["checkpoint_every"],
        checkpoint_prefix=cfg["checkpoint_prefix"],
    )

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
