# Getting Started

This guide walks through training your first EDDP potential and using it for predictions and structure optimization. No prior ML potential experience is required.

## Prerequisites

- Julia 1.9 or later
- A dataset of DFT-calculated structures with energies (`.res`, `.xyz`, or `.extxyz` format)

## Installation

```julia
using Pkg
Pkg.add("EDDPotentials")
```

## Recommended: Single-Call Training with `train_model`

The `train_model` function handles the entire training pipeline — structure loading, feature computation, data splitting, standardisation, model creation, and training — in a single call.

### Step 1: Train a Model

The simplest way — species and feature defaults are auto-inferred from your data:

```julia
using EDDPotentials

# Auto-inferred species + default feature parameters
tm = train_model("my_structures/*.res";
    nnodes=[20, 20],
    max_epochs=300,
    earlystop=30)
```

Or with explicit feature control:

```julia
# Explicit feature specification for finer control
tm = train_model("my_structures/*.res",
    CellFeature([:C]; p2=2:12, rcut2=5.0);
    nnodes=[20, 20],
    max_epochs=300,
    earlystop=30)
```

Or auto-inferred species with custom feature parameters:

```julia
tm = train_model("my_structures/*.res";
    p2=2:12, p3=2:4, q3=2:4, rcut2=5.0, rcut3=4.0,
    nnodes=[20, 20],
    max_epochs=300,
    earlystop=30)
```

By default (`verbose=true`), `train_model` prints a training configuration summary
and the final RMSE/MAE statistics. Set `verbose=false` to suppress.

This returns a `TrainedModel` containing the model, feature specification, and training results.

**Key parameters:**
- `CellFeature([:C])`: (optional) defines the atomic environment descriptors. If omitted, species are auto-detected from the data and defaults are used. Feature parameters (`p2`, `p3`, `q3`, `rcut2`, `rcut3`) can be passed directly as keyword arguments.
- `nnodes=[10, 10]`: hidden layer sizes. Use `[10]` for simpler models or `[30, 30, 30]` for more capacity.
- `max_epochs`: maximum training epochs (batched LM mode, default).
- `earlystop`: stop if validation error hasn't improved for this many epochs.
- `val_ratio=0.1`: fraction of data held out for validation (10% default).
- `train_method="lm_batched"`: training algorithm. Use `"lm"` for small datasets (<2,000 structures), `"lm_batched"` (default) for larger ones.
- `verbose=true`: print configuration summary and training results.

### Step 2: Inspect Results

```julia
# Pretty-print training and validation statistics
show(stdout, MIME("text/plain"), tm)
```

Output shows per-atom RMSE, MAE, maximum error, and Spearman scores for both training and validation sets.

### Step 3: Save the Model

```julia
save_model_artifact("my_model.json", tm)
```

Saves a self-contained JSON file with model weights, feature specification, and elemental energy offsets. This can be loaded later for predictions without re-training.

### Step 4: Predict Energy and Forces

```julia
# One-shot prediction on a new structure
cell = ...  # a CellBase Cell object
result = predict(tm, cell)

println("Energy: ", result.energy, " eV")
println("Forces: ", size(result.forces))
println("Stress: ", result.stress)
```

The `predict` function returns a NamedTuple with `:energy`, `:forces` (3×N matrix), and `:stress` (3×3 matrix). Use `predict(tm, cell; forces=false, stress=false)` to compute only energy.

### Step 5: Geometry Optimization

```julia
# Create a calculator from the trained model
calc = EDDPCalculator(tm)
set_cell!(calc, cell)

# Relax the structure
result = relax!(calc; method="fire", f_tol=0.01, itermax=500)

if result.converged
    println("Relaxed energy: ", eddp_energy(calc), " eV")
end
```

Or load a saved model and use it directly:

```julia
calc = load_calculator("my_model.json")
result = predict(calc, cell)
```

### Step 6: Evaluate on a Test Set

```julia
# Create a FeatureContainer for a test dataset
sc_test = StructureContainer(["test_data/*.res"])
fc_test = FeatureContainer(sc_test, tm.feature_spec)

# Standardize with training transforms
standardize!(fc_test; xt=tm.train_results.fc.xt, yt=tm.train_results.fc.yt)

# Create results for evaluation
using EDDPotentials: TrainingResults
test_results = TrainingResults(tm.model, fc_test)
println("Test RMSE: ", rmse_per_atom(test_results), " eV/atom")
```

---

## Advanced: Step-by-Step Training

For users who need fine-grained control over each stage of the training pipeline, the individual components are still available. This is useful when you need custom data filtering, non-standard model architectures, or ensemble training.

### Step A1: Load Training Data

```julia
using EDDPotentials

# From .res files with glob pattern
sc = StructureContainer(["my_structures/*.res"])

# From structures in memory
# sc = StructureContainer(structures, energies, labels)
```

The `StructureContainer` supports `.res`, `.xyz`, and `.extxyz` files. Files with per-atom enthalpy above `threshold` (default 10 eV/atom) are automatically filtered out.

### Step A2: Define and Compute Features

```julia
# Define feature specification
cf = CellFeature([:C]; p2=2:12, p3=2:4, q3=2:4, rcut2=5.0, rcut3=4.0)

# Compute feature vectors (threaded)
fc = FeatureContainer(sc, cf; show_progress=true)

println("Features per atom: ", nfeatures(cf))
```

### Step A3: Split and Standardize

```julia
# Split into training/validation (absolute counts or ratios)
fc_train, fc_val = split(fc, 0.8, 0.2; standardize=true)
# or: fc_train, fc_val = train_val_split(fc, ratio_val=0.2)

# Get raw feature matrices for manual use
x_train, y_train = get_fit_data(fc_train)
```

### Step A4: Build and Train a Model

```julia
using EDDPotentials.EDDPotentialsCore: ManualFluxBackPropInterface
using Flux: Chain, Dense

# Build a neural network architecture
n = nfeatures(cf)
chain = Chain(
    Dense(ginit(n, 20), leaky_tanh),
    Dense(ginit(20, 1))
)
itf = ManualFluxBackPropInterface(chain, xt=fc_train.xt, yt=fc_train.yt)

# Train
train!(itf, fc_train, fc_val; train_method="lm_batched", max_epochs=300, earlystop=30)
```

The `ManualFluxBackPropInterface` also has a convenience constructor:

```julia
itf = ManualFluxBackPropInterface(cf, 20; xt=fc_train.xt, yt=fc_train.yt)
```

### Step A5: Build an Ensemble

```julia
using EDDPotentials: create_ensemble, reinit

models = []
for _ in 1:10
    m = reinit(itf)  # fresh random weights, same architecture
    train!(m, fc_train, fc_val; train_method="lm_batched", max_epochs=100, show_progress=false)
    push!(models, m)
end

ensemble = create_ensemble(models, fc_train)
```

### Step A6: Use the Ensemble

```julia
using EDDPotentials: EDDPCalculator, predict

calc = EDDPCalculator(cf, ensemble, Dict{Symbol,Float64}())
result = predict(calc, cell)
```

### Step A7: Geometry Optimization

```julia
set_cell!(calc, cell)

# FIRE optimizer
relax!(calc; method="fire", f_tol=0.01, itermax=500)

# Or TPSD optimizer for volume-critical problems
relax!(calc; method="tpsd", f_tol=1e-4, itermax=1000)

# Or variable-cell (lattice + positions)
set_cell!(calc, cell)
relax!(calc; relax_cell=true, method="fire", itermax=500)
```

---

## What's Next?

- **Automated potential building**: [AIRSS Workflow](tutorials/airss_workflow.md) for iterative training with DFT integration
- **Feature customization**: [Understanding Features](concepts/features.md) for tuning features to your system
- **Large-scale training**: [Batched LM Training](howto/batched_lm_training.md) for datasets with thousands of structures
- **Structure prediction**: [Crystal Structure Prediction](howto/structure_prediction.md)
- **GPU acceleration**: [Enable GPU Acceleration](howto/enable_gpu_acceleration.md) for faster training
