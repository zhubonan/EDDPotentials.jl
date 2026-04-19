# Getting Started

This guide walks through training your first EDDP potential and using it for structure optimization. No prior ML potential experience is required.

## Prerequisites

- Julia 1.9 or later
- A dataset of DFT-calculated structures with energies (`.res`, `.xyz`, or `.cif` format)

## Installation

```julia
using Pkg
Pkg.add("EDDPotentials")
```

Optional dependencies for this guide:

```julia
Pkg.add(["ExtXYZ", "Flux"])
```

## Step 1: Load Your Training Data

EDDPotentials works with crystal structures from DFT calculations. Load them into a `StructureContainer`:

```julia
using EDDPotentials, ExtXYZ

# Load structures from an extxyz file
structures = ExtXYZ.load("my_structures.xyz")

# Or from CASTEP .res files
# using CellBase: read_res
# structures = read_res("path/to/structures/*.res")

# Wrap in a StructureContainer for feature computation
sc = StructureContainer(structures)
println("Loaded $(length(sc)) structures")
```

The structures must have total energies stored (in eV). ExtXYZ files store energies in the `energy` key.

## Step 2: Define Features

Features describe the local atomic environment around each atom. Use `CellFeature` with default settings to start:

```julia
using EDDPotentials

# Single-element system (e.g., carbon)
cf = CellFeature([:C])

# Multi-element system (e.g., carbon and nitrogen)
# cf = CellFeature([:C, :N])

println("Total features: ", nfeatures(cf))
```

The defaults work well for most systems. See [Understanding Features](concepts/features.md) for tuning guidance.

## Step 3: Compute Feature Vectors

Convert structures into numerical feature vectors that the neural network can use:

```julia
fc = FeatureContainer(cf, sc)

# Split into training and validation sets (80/20 default)
fc_train, fc_val = train_val_split(fc)

println("Training: $(length(fc_train)) structures")
println("Validation: $(length(fc_val)) structures")
```

## Step 4: Train a Model

Train a shallow neural network using the Levenberg-Marquardt algorithm:

```julia
using Flux: Chain, Dense
import EDDPotentials: FluxInterface, train!

n = nfeatures(cf)

# Define a network: input -> 20 neurons (tanh) -> 1 output
chain = Chain(Dense(n, 20, tanh_fast), Dense(20, 1))
model = FluxInterface(chain)

# Train on the training set, validate on the validation set
train!(model, fc_train, fc_val; maxIter=300, earlystop=30)
```

Training output shows the RMSE (root mean square error) per atom decreasing over iterations. Typical converged values are 1-10 meV/atom for well-behaved systems.

## Step 5: Build an Ensemble

An ensemble of independently trained models provides more robust predictions and uncertainty estimates:

```julia
import EDDPotentials: create_ensemble, get_fit_data

# Train multiple models with different random initializations
models = []
for i in 1:10
    chain = Chain(Dense(n, 20, tanh_fast), Dense(20, 1))
    model = FluxInterface(chain)
    train!(model, fc_train, fc_val; maxIter=300, show_progress=false, earlystop=30)
    push!(models, model)
end

# Combine into a weighted ensemble using non-negative least squares
x_train, y_train = get_fit_data(fc_train)
ensemble = create_ensemble(models, x_train, y_train)

# Save the ensemble
using JLD2
jldopen("my_ensemble.jld2", "w") do f
    save_as_jld2(f, ensemble)
    f["cf"] = cf
end
```

## Step 6: Use the Potential

Create a calculator and use it for energy/force evaluation and structure optimization:

```julia
import EDDPotentials: opt_fire

# Pick a structure from your dataset
structure = first(structures)

# Create calculator
calc = NNCalc(structure, cf, ensemble)

# Evaluate energy and forces
energy = eddp_energy(calc)
forces = eddp_forces(calc)
stress = eddp_stress(calc)

println("Energy: $(energy) eV")
println("Energy/atom: $(energy / length(structure)) eV")

# Geometry optimization with variable cell
vc_calc = VariableCellCalc(calc)
opt = opt_fire(vc_calc; f_tol=0.01, itermax=500)

if converged(opt, vc_calc)
    println("Optimization converged!")
    println("Relaxed energy: $(eddp_energy(vc_calc)) eV")
end
```

## Step 7: Evaluate Quality

Check your potential's accuracy on the validation set:

```julia
import EDDPotentials: TrainingResults, rmse_per_atom, mae_per_atom

results = TrainingResults(ensemble, fc_val)
println("Validation RMSE: $(rmse_per_atom(results)) eV/atom")
println("Validation MAE:  $(mae_per_atom(results)) eV/atom")
```

If the RMSE is too high, try:
- More training iterations (`maxIter=1000`)
- A wider hidden layer (`Dense(n, 40, tanh_fast)`)
- More training data
- Adjusting feature cutoffs (see [Understanding Features](concepts/features.md))

## What's Next?

- **Automated potential building**: [AIRSS Workflow](tutorials/airss_workflow.md) for iterative training with DFT integration
- **Feature customization**: [Understanding Features](concepts/features.md) for tuning features to your system
- **Large-scale structure search**: [Crystal Structure Prediction](howto/structure_prediction.md)
- **GPU acceleration**: [Enable GPU Acceleration](howto/enable_gpu_acceleration.md) for faster training
