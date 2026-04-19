using TOML
using Flux: tanh
using EDDPotentials
using EDDPotentials:
    CellFeature,
    FeatureContainer,
    NNCalc,
    StructureContainer,
    TrainingResults,
    eddp_energy,
    feature_vector,
    ginit,
    load_feature_artifact,
    load_model_artifact,
    mae_per_atom,
    rmse_per_atom,
    save_feature_json,
    save_model_json,
    split,
    train!
using EDDPotentials.EDDPotentialsCore:
    ManualFluxBackPropInterface,
    forward!,
    nparams

const DEFAULT_DATA_DIR = normpath(joinpath(@__DIR__, "..", "..", "B-eddp-example"))
const DEFAULT_OUTDIR = normpath(joinpath(@__DIR__, "..", "tmp", "b-json-roundtrip"))

function _parse_bool(value::AbstractString)
    lowered = lowercase(value)
    lowered in ("1", "true", "yes", "y", "on") && return true
    lowered in ("0", "false", "no", "n", "off") && return false
    error("Expected a boolean value, got `$value`.")
end

function _parse_args(args)
    cfg = Dict{String,Any}(
        "data_dir" => DEFAULT_DATA_DIR,
        "outdir" => DEFAULT_OUTDIR,
        "max_structures" => 1000,
        "val_ratio" => 0.1,
        "hidden_size" => 8,
        "train_method" => "lm_batched",
        "batch_size" => 256,
        "max_iter" => 10,
        "earlystop" => 5,
        "threshold" => 10.0,
        "seed" => 42,
        "show_progress" => true,
        "prefix" => "b_model",
    )

    for arg in args
        occursin("=", arg) || error("Expected key=value arguments, got `$arg`.")
        key, value = split(arg, "="; limit=2)
        if key in ("data_dir", "workdir")
            cfg["data_dir"] = normpath(value)
        elseif key == "outdir"
            cfg["outdir"] = normpath(value)
        elseif key == "max_structures"
            cfg["max_structures"] = parse(Int, value)
        elseif key == "val_ratio"
            cfg["val_ratio"] = parse(Float64, value)
        elseif key == "hidden_size"
            cfg["hidden_size"] = parse(Int, value)
        elseif key == "train_method"
            cfg["train_method"] = value
        elseif key == "batch_size"
            cfg["batch_size"] = parse(Int, value)
        elseif key in ("max_iter", "maxIter")
            cfg["max_iter"] = parse(Int, value)
        elseif key == "earlystop"
            cfg["earlystop"] = parse(Int, value)
        elseif key == "threshold"
            cfg["threshold"] = parse(Float64, value)
        elseif key == "seed"
            cfg["seed"] = parse(Int, value)
        elseif key == "show_progress"
            cfg["show_progress"] = _parse_bool(value)
        elseif key == "prefix"
            cfg["prefix"] = value
        else
            error("Unknown argument `$key`.")
        end
    end

    cfg
end

function _collect_training_paths(data_dir::AbstractString; max_structures::Int=0)
    isdir(data_dir) || error("Data directory does not exist: $data_dir")
    gen_dirs = sort(filter(name -> occursin(r"^gen\d+-dft$", name), readdir(data_dir)))
    isempty(gen_dirs) && error("No `gen*-dft` folders found in $data_dir")

    paths = String[]
    for dirname in gen_dirs
        dirpath = joinpath(data_dir, dirname)
        res_files = sort(filter(name -> endswith(lowercase(name), ".res"), readdir(dirpath)))
        append!(paths, joinpath.(dirpath, res_files))
    end

    isempty(paths) && error("No `.res` training structures found in $data_dir")
    if max_structures > 0 && length(paths) > max_structures
        return paths[1:max_structures]
    end
    paths
end

function _load_link_config(data_dir::AbstractString)
    link_path = joinpath(data_dir, "link.toml")
    isfile(link_path) || error("Missing `link.toml` in $data_dir")
    TOML.parsefile(link_path)
end

function _feature_from_link_config(config::Dict{String,Any})
    cf_cfg = get(config, "cf", nothing)
    cf_cfg === nothing && error("Missing `[cf]` section in link.toml")

    elements = Symbol.(cf_cfg["elements"])
    p2 = get(cf_cfg, "p2", [2.0, 10.0, 5.0])
    p3 = get(cf_cfg, "p3", p2)
    q3 = get(cf_cfg, "q3", p3)
    geometry_sequence = get(cf_cfg, "geometry_sequence", true)
    rcut2 = Float64(get(cf_cfg, "rcut2", 6.0))
    rcut3 = Float64(get(cf_cfg, "rcut3", rcut2))

    CellFeature(
        elements;
        p2,
        p3,
        q3,
        geometry_sequence,
        rcut2,
        rcut3,
    )
end

function _elemental_energies_from_link_config(config::Dict{String,Any}, elements)
    state_cfg = get(config, "state", Dict{String,Any}())
    ref_cfg = get(state_cfg, "elemental_energies", Dict{String,Any}())

    energies = Dict{Symbol,Float64}()
    for elem in elements
        key = String(elem)
        haskey(ref_cfg, key) || continue
        energies[elem] = Float64(ref_cfg[key])
    end
    energies
end

function _roundtrip_check(model, feature_spec, elemental_energies, cell, model_path, feature_path)
    input_before = feature_vector(feature_spec, cell)
    output_before = forward!(model, input_before)
    energy_before = eddp_energy(
        NNCalc(deepcopy(cell), feature_spec, model; elemental_energies, core=nothing),
    )

    artifact = load_model_artifact(model_path)
    feature_artifact = load_feature_artifact(feature_path)

    input_after = feature_vector(artifact.feature_spec, cell)
    feature_input_after = feature_vector(feature_artifact.feature_spec, cell)
    output_after = forward!(artifact.model, input_after)
    energy_after = eddp_energy(
        NNCalc(
            deepcopy(cell),
            artifact.feature_spec,
            artifact.model;
            elemental_energies=artifact.elemental_energies,
            core=nothing,
        ),
    )

    feature_artifact.elemental_energies == elemental_energies ||
        error("Feature artifact elemental energies do not match the original values.")
    artifact.elemental_energies == elemental_energies ||
        error("Model artifact elemental energies do not match the original values.")
    feature_input_after == input_before ||
        error("Feature artifact regenerates different feature vectors after JSON round-trip.")
    input_after == input_before ||
        error("Feature vectors differ before and after JSON round-trip.")
    output_after == output_before ||
        error("Model outputs differ before and after JSON round-trip.")
    energy_after == energy_before ||
        error("NNCalc energies differ before and after JSON round-trip.")

    (
        feature_rows=size(input_before, 1),
        feature_cols=size(input_before, 2),
        original_output=vec(output_before),
        output=vec(output_after),
        original_energy=energy_before,
        energy=energy_after,
    )
end

function main(args=ARGS)
    cfg = _parse_args(args)
    data_dir = cfg["data_dir"]
    outdir = cfg["outdir"]
    mkpath(outdir)

    println("Loading configuration from: $(joinpath(data_dir, "link.toml"))")
    link_cfg = _load_link_config(data_dir)
    cf = _feature_from_link_config(link_cfg)
    elemental_energies = _elemental_energies_from_link_config(link_cfg, cf.elements)

    println("Collecting training data from: $data_dir")
    paths = _collect_training_paths(data_dir; max_structures=cfg["max_structures"])
    println("Found $(length(paths)) training structures.")

    sc = StructureContainer(paths; threshold=cfg["threshold"])
    println("Loaded $(length(sc)) structures after threshold filtering.")
    length(sc) >= 2 || error("Need at least two structures after filtering to create a train/validation split.")

    fc = FeatureContainer(
        sc,
        cf;
        show_progress=cfg["show_progress"],
        elemental_energies,
    )

    nval = max(1, Int(floor(length(fc) * cfg["val_ratio"])))
    ntrain = length(fc) - nval
    ntrain > 0 || error("Not enough structures to create a train/validation split.")

    fc_train, fc_val = split(
        fc,
        ntrain,
        nval;
        shuffle=true,
        seed=cfg["seed"],
        standardize=true,
        apply_transform=true,
    )

    model = ManualFluxBackPropInterface(
        cf,
        cfg["hidden_size"];
        init=ginit,
        σ=tanh,
        xt=fc_train.xt,
        yt=fc_train.yt,
        apply_xt=false,
    )

    println("Model parameter count: $(nparams(model))")
    println("Training with $(length(fc_train)) train / $(length(fc_val)) validation structures.")

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
            fc_val;
            train_kwargs...,
            batch_size=cfg["batch_size"],
        )
    else
        train!(model, fc_train, fc_val; train_kwargs...)
    end

    train_results = TrainingResults(model, fc_train)
    val_results = TrainingResults(model, fc_val)

    model_path = joinpath(outdir, "$(cfg["prefix"]).json")
    feature_path = joinpath(outdir, "$(cfg["prefix"])_feature.json")

    metadata = Dict(
        "data_dir" => data_dir,
        "nstructures" => length(sc),
        "ntrain" => length(fc_train),
        "nval" => length(fc_val),
        "hidden_size" => cfg["hidden_size"],
        "train_method" => cfg["train_method"],
        "batch_size" => cfg["batch_size"],
        "max_iter" => cfg["max_iter"],
        "threshold" => cfg["threshold"],
    )

    save_model_json(
        model_path,
        model;
        feature_spec=cf,
        elemental_energies,
        metadata,
    )
    save_feature_json(feature_path, cf; elemental_energies)

    probe = _roundtrip_check(
        model,
        cf,
        elemental_energies,
        sc.structures[1],
        model_path,
        feature_path,
    )

    println()
    println("Train RMSE/atom: $(rmse_per_atom(train_results)) eV")
    println("Train MAE/atom : $(mae_per_atom(train_results)) eV")
    println("Val RMSE/atom  : $(rmse_per_atom(val_results)) eV")
    println("Val MAE/atom   : $(mae_per_atom(val_results)) eV")
    println()
    println("Saved model artifact  : $model_path")
    println("Saved feature artifact: $feature_path")
    println("Round-trip check passed on the first structure.")
    println("Feature matrix size   : $(probe.feature_rows) x $(probe.feature_cols)")
    println("Original output       : $(probe.original_output)")
    println("Recovered output      : $(probe.output)")
    println("Original energy       : $(probe.original_energy)")
    println("Recovered energy      : $(probe.energy)")
end

main()
