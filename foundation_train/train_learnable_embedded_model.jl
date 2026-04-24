using EDDPotentials
using EDDPotentials:
    FeatureContainer,
    RandomEmbeddedCellFeature,
    MultiRandomEmbeddedCellFeature,
    StructureContainer,
    TrainingResults,
    ginit,
    get_elemental_energy,
    mae_per_atom,
    nfeatures,
    rmse_per_atom,
    train_lm_with_embedding!,
    train_lm_batched_with_embedding!,
    split,
    standardize!
using EDDPotentials.EDDPotentialsCore: ManualFluxBackPropInterface, nparams
using CellBase: species
using JSON

const DEFAULT_DATASET = joinpath(
    @__DIR__,
    "mp_finetuning-MACE_matpes_ft_run-42_combined.xyz",
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
        "dataset" => DEFAULT_DATASET,
        "max_structures" => 10000,
        "embedding_size" => 4,
        "n_multi" => 1,
        "hidden_size" => [10, 10, 10],
        "p2" => [2, 4, 6, 8, 10],
        "p3" => [2, 4, 6, 8, 10],
        "q3" => [3, 5, 7, 9],
        "val_ratio" => 0.1,
        "train_method" => "lm_batched",
        "batch_size" => 1024,
        "max_iter" => 20,
        "earlystop" => 20,
        "seed" => 42,
        "show_progress" => true,
        "threshold" => 10.0,
        "normalize_embedding" => false,
        "recompute_xt" => false,
        "e0s" => DEFAULT_E0S,
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
        elseif key == "normalize_embedding"
            config["normalize_embedding"] = _parse_bool(value)
        elseif key == "recompute_xt"
            config["recompute_xt"] = _parse_bool(value)
        elseif key in ("e0s", "elemental_energies", "e0")
            config["e0s"] = value
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

function main(args=ARGS)
    cfg = _parse_args(args)
    dataset = cfg["dataset"]

    println("Loading dataset: $dataset")
    println(
        "Config: max_structures=$(cfg["max_structures"]) embedding_size=$(cfg["embedding_size"]) " *
        "n_multi=$(cfg["n_multi"]) " *
        "hidden_size=$(cfg["hidden_size"]) p2=$(cfg["p2"]) p3=$(cfg["p3"]) q3=$(cfg["q3"]) " *
        "train_method=$(cfg["train_method"]) batch_size=$(cfg["batch_size"]) val_ratio=$(cfg["val_ratio"]) " *
        "max_iter=$(cfg["max_iter"]) earlystop=$(cfg["earlystop"]) " *
        "normalize_embedding=$(cfg["normalize_embedding"]) recompute_xt=$(cfg["recompute_xt"])",
    )

    max_structures = cfg["max_structures"]
    dataset_path = if max_structures > 0 && endswith(lowercase(dataset), ".xyz") || endswith(lowercase(dataset), ".extxyz")
        "$(dataset)@:$(max_structures)"
    else
        dataset
    end

    # Load structures
    sc = StructureContainer(dataset_path; threshold=cfg["threshold"])
    println("Loaded $(length(sc)) structures after threshold filtering.")
    symbols = _unique_symbols(sc.structures)

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

    # Split into train/val at the StructureContainer level
    nval = max(1, Int(floor(length(sc) * cfg["val_ratio"])))
    ntrain = length(sc) - nval
    ntrain > 0 || error("Not enough structures after filtering to create a train/val split.")

    sc_train, sc_val = split(sc, ntrain, nval; shuffle=true, seed=cfg["seed"])

    e0s = load_e0s(cfg["e0s"])
    if !isempty(e0s)
        println("Elemental reference energies: $(length(e0s)) elements loaded from $(cfg["e0s"])")
    end

    # Build FeatureContainer from train split to get xt/yt statistics
    fc_train = FeatureContainer(sc_train, embedded; show_progress=true, elemental_energies=e0s)
    standardize!(fc_train)

    # Extract raw structures and energies for the learnable embedding path
    train_structures = sc_train.structures
    train_y = copy(sc_train.H)
    val_structures = sc_val.structures
    val_y = copy(sc_val.H)

    # Subtract elemental reference energies
    if !isempty(e0s)
        for i in eachindex(train_structures)
            train_y[i] -= get_elemental_energy(train_structures[i], e0s)
        end
        for i in eachindex(val_structures)
            val_y[i] -= get_elemental_energy(val_structures[i], e0s)
        end
    end

    nh = cfg["hidden_size"]
    model = ManualFluxBackPropInterface(
        embedded,
        nh...;
        init=ginit,
        σ=tanh,
        xt=fc_train.xt,
        yt=fc_train.yt,
    )
    println("Model parameters: $(nparams(model))  |  Embedding parameters: $(length(embedded.embedding))")

    normalize_emb = cfg["normalize_embedding"]
    recompute_xt = cfg["recompute_xt"]

    println("Training (learnable embedding) with $(length(train_structures)) train / $(length(val_structures)) val structures.")
    if cfg["train_method"] == "lm_batched"
        train_lm_batched_with_embedding!(
            model, train_structures, embedded, train_y;
            structures_val=val_structures,
            y_val=val_y,
            max_epochs=cfg["max_iter"],
            batch_size=cfg["batch_size"],
            show_progress=cfg["show_progress"],
            earlystop=cfg["earlystop"],
            normalize_embedding=normalize_emb,
            recompute_xt=recompute_xt,
            checkpoint_every=cfg["checkpoint_every"],
            checkpoint_prefix=cfg["checkpoint_prefix"],
            elemental_energies=e0s,
        )
    else
        train_lm_with_embedding!(
            model, train_structures, embedded, train_y;
            structures_val=val_structures,
            y_val=val_y,
            maxIter=cfg["max_iter"],
            show_progress=cfg["show_progress"],
            earlystop=cfg["earlystop"],
            normalize_embedding=normalize_emb,
            recompute_xt=recompute_xt,
            checkpoint_every=cfg["checkpoint_every"],
            checkpoint_prefix=cfg["checkpoint_prefix"],
            elemental_energies=e0s,
        )
    end

    # Evaluate — recompute features with final (trained) embedding
    fc_val_final = FeatureContainer(sc_val, embedded; show_progress=false, elemental_energies=e0s)
    fc_train_final = FeatureContainer(sc_train, embedded; show_progress=false, elemental_energies=e0s)

    train_results = TrainingResults(model, fc_train_final)
    val_results = TrainingResults(model, fc_val_final)

    println()
    println("Train RMSE/atom: $(rmse_per_atom(train_results)) eV")
    println("Train MAE/atom : $(mae_per_atom(train_results)) eV")
    println("Val RMSE/atom  : $(rmse_per_atom(val_results)) eV")
    println("Val MAE/atom   : $(mae_per_atom(val_results)) eV")
end

main()
