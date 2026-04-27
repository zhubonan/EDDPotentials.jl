# Profile learnable embedding training to measure time breakdown
# between feature computation, forward/backward passes, gradient
# computation, and LM solve.
#
# Usage:
#   julia --project=. foundation_train/profile_learnable_embedded.jl [max_structures=500] [embedding_size=4] [n_multi=1]

using EDDPotentials
using EDDPotentials:
    FeatureStore, StructureContainer, TrainingResults,
    RandomEmbeddedCellFeature, MultiRandomEmbeddedCellFeature,
    ginit, get_elemental_energy, mae_per_atom, nfeatures, rmse_per_atom,
    train_lm_batched_with_embedding!,
    split, standardize!,
    enable_eddp_timer!, reset_eddp_timer!
using EDDPotentials.EDDPotentialsCore: nparams, ManualFluxBackPropInterface
using EDDPotentials: to  # TimerOutput
using CellBase: species
using Printf: @printf
using JSON

const DEFAULT_E0S = joinpath(@__DIR__, "e0s.json")

function load_e0s(path::AbstractString=DEFAULT_E0S)
    (path == "" || lowercase(path) == "none") && return Dict{Symbol,Float64}()
    isfile(path) || error("Elemental energies file not found: `$path`")
    data = JSON.parsefile(path)
    Dict{Symbol,Float64}(Symbol(k) => Float64(v) for (k, v) in data)
end

function _parse_args(args)
    Dict(
        "max_structures" => length(args) >= 1 ? parse(Int, args[1]) : 500,
        "embedding_size" => length(args) >= 2 ? parse(Int, args[2]) : 4,
        "n_multi" => length(args) >= 3 ? parse(Int, args[3]) : 1,
        "batch_size" => 1024,
        "max_epochs" => 5,
        "hidden_size" => [10, 10, 10],
    )
end

function main(args=ARGS)
    cfg = _parse_args(args)

    dataset = joinpath(@__DIR__, "mp_finetuning-MACE_matpes_ft_run-42_combined.xyz")
    max_structures = cfg["max_structures"]
    dataset_path = "$(dataset)@:$(max_structures)"

    sc = StructureContainer(dataset_path; threshold=10.0)
    println("Loaded $(length(sc)) structures")
    symbols = sort!(collect(Set(reduce(vcat, unique.(species.(sc.structures))))))

    n_multi = cfg["n_multi"]
    embedded = if n_multi > 1
        MultiRandomEmbeddedCellFeature(
            symbols;
            nembed=cfg["embedding_size"],
            n_multi=n_multi,
            seeds=collect(0:n_multi-1) .+ 42,
            p2=2:2:10,
            p3=2:2:10,
            q3=[3, 5, 7, 9],
        )
    else
        RandomEmbeddedCellFeature(
            symbols;
            nembed=cfg["embedding_size"],
            seed=42,
            p2=2:2:10,
            p3=2:2:10,
            q3=[3, 5, 7, 9],
        )
    end
    n_emb_params = sum(length(emb.embedding) for emb in (n_multi > 1 ? embedded.embeddings : [embedded]))
    println("Features: $(nfeatures(embedded)), Species: $(length(symbols)), n_multi=$(n_multi), Embedding params: $(n_emb_params)")

    nval = max(1, Int(floor(length(sc) * 0.1)))
    sc_train, sc_val = split(sc, length(sc)-nval, nval; shuffle=true, seed=42)

    e0s = load_e0s()

    fc_train = FeatureStore(sc_train, embedded; show_progress=false, elemental_energies=e0s)
    standardize!(fc_train)

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
        embedded, nh...;
        init=ginit, σ=tanh,
        xt=fc_train.xt, yt=fc_train.yt,
    )
    println("NN params: $(nparams(model)), Total: $(nparams(model) + n_emb_params)")

    # Enable profiling and train
    enable_eddp_timer!()
    reset_eddp_timer!()

    println("\n--- Training $(cfg["max_epochs"]) epochs (batch_size=$(cfg["batch_size"])) ---\n")
    train_lm_batched_with_embedding!(
        model, train_structures, embedded, train_y;
        structures_val=val_structures,
        y_val=val_y,
        max_epochs=cfg["max_epochs"],
        batch_size=cfg["batch_size"],
        show_progress=true,
        earlystop=0,
    )

    # Print the timer report
    println("\n", "="^70)
    println("PROFILING REPORT")
    println("="^70)
    show(stdout, MIME("text/plain"), to)
    println("\n")

    # Aggregate timing by category across all nested timers
    function _collect_timers!(dict, timer)
        for (label, inner) in timer.inner_timers
            t_sec = inner.accumulated_data.time / 1e9  # ns -> s
            dict[label] = get(dict, label, 0.0) + t_sec
            _collect_timers!(dict, inner)
        end
    end
    flat = Dict{String,Float64}()
    _collect_timers!(flat, to)

    feature_time = get(flat, "embed: feature", 0.0)
    forward_time = get(flat, "embed: forward", 0.0)
    backward_time = get(flat, "embed: backward", 0.0)
    gradparam_time = get(flat, "embed: gradparam", 0.0)
    gradinp_time = get(flat, "embed: gradinp", 0.0)
    grad_emb_time = get(flat, "embed: grad_emb", 0.0)
    rmse_feat_time = get(flat, "rmse: feature recompute", 0.0)
    rmse_fwd_time = get(flat, "rmse: forward!", 0.0)

    loop_total = feature_time + forward_time + backward_time + gradparam_time + gradinp_time + grad_emb_time
    loop_total > 0 || (loop_total = 1.0)

    println("-"^60)
    println("BREAKDOWN of per-structure operations (training loop)")
    println("-"^60)
    @printf "  %-32s %8.3f s  (%5.1f%%)\n" "Feature recomputation" feature_time (feature_time / loop_total * 100)
    @printf "  %-32s %8.3f s  (%5.1f%%)\n" "Embedding gradient" grad_emb_time (grad_emb_time / loop_total * 100)
    @printf "  %-32s %8.3f s  (%5.1f%%)\n" "NN forward pass" forward_time (forward_time / loop_total * 100)
    @printf "  %-32s %8.3f s  (%5.1f%%)\n" "NN backward pass" backward_time (backward_time / loop_total * 100)
    @printf "  %-32s %8.3f s  (%5.1f%%)\n" "gradinp (dE/dfvec)" gradinp_time (gradinp_time / loop_total * 100)
    @printf "  %-32s %8.3f s  (%5.1f%%)\n" "NN gradparam" gradparam_time (gradparam_time / loop_total * 100)
    @printf "  %-32s %8.3f s\n" "Loop subtotal" loop_total

    callback_time = rmse_feat_time + rmse_fwd_time
    println("\n-"^60)
    println("CALLBACK (RMSE evaluation, outside LM loop)")
    println("-"^60)
    @printf "  %-32s %8.3f s\n" "RMSE feature recomputation" rmse_feat_time
    @printf "  %-32s %8.3f s\n" "RMSE forward pass" rmse_fwd_time
    @printf "  %-32s %8.3f s\n" "Callback subtotal" callback_time
end

main()
