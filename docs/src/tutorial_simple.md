# Tutorial: Building Your First EDDP Potential

This tutorial walks you through a complete workflow for building a machine learning potential for carbon structures. By the end, you'll have trained an ensemble model and used it to predict energies for new structures.

## Prerequisites

Make sure you have EDDPotentials.jl installed:
```julia
using Pkg
Pkg.add("EDDPotentials")
```

You'll also need:
```julia
Pkg.add(["ExtXYZ", "Flux", "Plots"])
```

## Overview

The workflow consists of these steps:
1. Load structures from files
2. Create a StructureContainer
3. Generate features with FeatureContainer
4. Split data for training and validation
5. Train a single model
6. Create an ensemble of models
7. Use NNCalc for predictions
8. (Optional) Geometry optimization

## Step 1: Load Structures

First, we need training data. In a real application, you would have structures with energies from DFT calculations. For this tutorial, we'll assume you have Extended XYZ files.

```julia
using EDDPotentials
using Flux  # For neural network construction
using ExtXYZ  # For loading .xyz files
import EDDPotentials: train!, FluxInterface, create_ensemble, get_fit_data

# Load structures
# Each structure should have an "energy" property (in eV)
structures = ExtXYZ.load("carbon_data.xyz")

println("Loaded $(length(structures)) structures")

# Inspect the first structure
println("First structure has $(length(structures[1])) atoms")
println("Energy: $(structures[1].energy) eV")
```

### Creating Test Data

If you don't have DFT data yet, here's how you might generate it:

```julia
# 1. Create or load initial structures (graphite, diamond, random configs)
# 2. Run DFT calculations using your preferred code (CASTEP, VASP, etc.)
# 3. Extract structures and energies
# 4. Save to Extended XYZ format with energies

# Example structure in Extended XYZ format:
# 8
# Lattice="3.567 0.0 0.0 0.0 3.567 0.0 0.0 0.0 3.567" Properties=species:S:1:pos:R:3 energy=-62.451 pbc="T T T"
# C  0.0  0.0  0.0
# C  1.783  1.783  0.0
# ...
```

## Step 2: Create StructureContainer

The `StructureContainer` organizes your structures and their energies for training:

```julia
# Create container from loaded structures
sc = StructureContainer(structures)

# Or load directly from files (supports wildcards)
# sc = StructureContainer("carbon_data/*.xyz")

println("Container has $(length(sc)) structures")
println("Total atoms: $(sum(natoms(sc)))")

# Access energies
energies = sc.H
println("Energy range: $(minimum(energies)) to $(maximum(energies)) eV")

# Container also stores metadata for each structure
# (useful for tracking which structures have high errors)
println("First structure label: $(sc.labels[1])")
```

## Step 3: Generate Features

Features are mathematical descriptors of the atomic environment. The `CellFeature` type sets up both two-body and three-body features:

```julia
# Create default features for carbon
cf = CellFeature([:C])

println("Number of features per atom: ", nfeatures(cf))
println("Two-body features: ", nfeatures(cf.features2))
println("Three-body features: ", nfeatures(cf.features3))

# Compute features for all structures
fc = FeatureContainer(cf, sc)

println("Feature container ready with $(length(fc)) structures")
```

### Customizing Features

You can customize the feature parameters:

```julia
# More detailed features with longer cutoff
cf_detailed = CellFeature(
    [:C];
    p2=2:7,          # Two-body powers (more powers = more features)
    p3=2:5,          # Three-body powers
    q3=1:3,          # Three-body angular powers
    rcut2=4.5,       # Two-body cutoff radius (Å)
    rcut3=3.5,       # Three-body cutoff radius (Å)
    geometry_sequence=true  # Use geometric sequence for powers
)

println("Detailed features: ", nfeatures(cf_detailed), " per atom")
```

More features can improve accuracy but increase training time. Start with defaults and add complexity if needed.

## Step 4: Split Data

Split your data into training and test sets to detect overfitting:

```julia
# 90% training, 10% test (default)
fc_train, fc_test = train_test_split(fc)

println("Training set: $(length(fc_train)) structures")
println("Test set: $(length(fc_test)) structures")

# You can also control the split ratio and shuffling
# fc_train, fc_test = train_test_split(fc; ratio_test=0.2, shuffle=true)
```

The test set should represent the same distribution as your training set. If you have different compositions or structure types, make sure both sets include them proportionally.

## Step 5: Train a Single Model

Now we'll train a neural network to predict energies from features:

```julia
using Flux

# Create a simple neural network
# Architecture: input_size → 20 hidden neurons → 1 output
nfeat = nfeatures(cf)
chain = Chain(
    Dense(nfeat, 20, tanh),  # Hidden layer with tanh activation
    Dense(20, 1)              # Output layer (energy per atom)
)

# Wrap in EDDPotentials interface
model = FluxInterface(chain)

# Train using Levenberg-Marquardt algorithm
println("Training model...")
opt_res, params, history = train!(
    model,
    fc_train,
    fc_test;
    maxIter=300,           # Maximum iterations
    show_progress=true,    # Print progress during training
    earlystop=30          # Stop if no improvement for 30 iterations
)

println("Training complete!")
println("Final train RMSE: $(history[end, 1]) eV/atom")
println("Final test RMSE: $(history[end, 2]) eV/atom")
```

### Monitoring Training

The `history` matrix contains RMSE for each iteration:

```julia
using Plots

# Plot training curves
plot(history[:, 1], label="Train RMSE", xlabel="Iteration", ylabel="RMSE (eV/atom)")
plot!(history[:, 2], label="Test RMSE")
```

Good training should show:
- Decreasing train and test error
- Test error following train error (not diverging = overfitting)
- Convergence before reaching maxIter (if not, increase maxIter)

## Step 6: Evaluate the Model

Check detailed performance metrics:

```julia
# Create results objects
results_train = TrainingResults(model, fc_train)
results_test = TrainingResults(model, fc_test)

# Print summary statistics
println("=" ^ 50)
println("Training Set Performance:")
println(results_train)

println("\n" * "=" ^ 50)
println("Test Set Performance:")
println(results_test)
```

This shows:
- RMSE and MAE per atom
- Maximum error and which structure it occurred on
- Spearman correlation (ranking accuracy)

### Analyzing Errors

Identify structures with high errors:

```julia
# Get per-atom errors for each structure
errors = ae_per_atom(results_test)

# Find worst predictions
worst_idx = sortperm(errors, rev=true)[1:5]
println("Top 5 worst predictions:")
for (i, idx) in enumerate(worst_idx)
    println("  $i. $(fc_test.labels[idx]): $(errors[idx]) eV/atom")
end

# These structures might need more similar training data
```

## Step 7: Create an Ensemble

Single models can have high variance. Ensemble models combine multiple models for better predictions and uncertainty estimates:

```julia
println("Training ensemble of 10 models...")

models = []
for i in 1:10
    # Create new model with random initialization
    chain = Chain(Dense(nfeat, 20, tanh), Dense(20, 1))
    itf = FluxInterface(chain)

    # Train (suppress progress output)
    train!(itf, fc_train, fc_test; maxIter=300, show_progress=false, earlystop=30)

    push!(models, itf)
    println("Model $i complete")
end

# Combine models using NNLS weighting
x_train, y_train = get_fit_data(fc_train)
ensemble = create_ensemble(models, x_train, y_train)

println("\nEnsemble created:")
println("  Number of models: $(length(ensemble.models))")
println("  Weights: $(ensemble.weights)")
```

The `create_ensemble` function:
- Computes optimal weights using non-negative least squares
- Removes models with very low weights (they don't help)
- Typically results in 6-8 models with meaningful contributions

### Evaluate Ensemble Performance

```julia
# Compare single model vs ensemble
results_ensemble = TrainingResults(ensemble, fc_test)

println("\nSingle Model Test RMSE: $(rmse_per_atom(results_test)) eV/atom")
println("Ensemble Test RMSE: $(rmse_per_atom(results_ensemble)) eV/atom")

# Ensemble should be better or similar, rarely worse
```

## Step 8: Use the Calculator

Now we can use the trained potential to make predictions:

```julia
# Load a new structure
new_structure = structures[1]  # Use first structure as example

# Create a calculator from the ensemble
# NNCalc requires the structure, feature spec, and neural network interface
calc = NNCalc(new_structure, cf, ensemble)

# Predict total energy (calculator operates on its internal structure)
energy = eddp_energy(calc)
println("Predicted energy: $energy eV")

# Predict forces on all atoms
forces = eddp_forces(calc)
println("Forces shape: $(size(forces))")  # (3, natoms)
println("Max force magnitude: $(maximum(abs.(forces))) eV/Å")

# Forces point in the direction atoms should move to lower energy
```

### Calculator Properties

The calculator stores everything needed for predictions:

```julia
println("Calculator feature spec: ", calc.fspec)
println("Calculator has $(length(calc.itf.models)) models")

# You can save the calculator for later use
using JLD2
save("carbon_potential.jld2", "calc", calc)

# And load it back
calc_loaded = load("carbon_potential.jld2", "calc")
```

## Step 9: Geometry Optimization (Optional)

Use the potential to optimize structures:

```julia
# Start with a structure that's not at equilibrium
initial = structures[1]

# Create calculator with the initial structure
calc = NNCalc(initial, cf, ensemble)
initial_energy = eddp_energy(calc)
println("Initial energy: $initial_energy eV")

# Optimize atomic positions (fixed cell)
# relax! modifies the calculator's internal structure in-place
relax!(calc; fmax=0.01, max_steps=100)
optimized_energy = eddp_energy(calc)
println("Optimized energy: $optimized_energy eV")

# Check if optimization converged
final_forces = eddp_forces(calc)
max_force = maximum(abs.(final_forces))
println("Max force after optimization: $max_force eV/Å")

# Converged if max_force < 0.01 eV/Å
if max_force < 0.01
    println("Optimization converged!")
else
    println("Optimization did not fully converge, consider more steps")
end

# The optimized structure is stored in calc.cell
optimized_structure = calc.cell
```

### Variable Cell Optimization

To optimize both atomic positions and cell parameters:

```julia
# This is more expensive but allows cell shape to relax
using EDDPotentials: VariableCellCalc

initial = structures[1]
vc_calc = VariableCellCalc(initial, cf, ensemble)

println("Initial volume: $(volume(initial)) Å³")

# Optimize both positions and cell
relax!(vc_calc; fmax=0.01, max_steps=200)

println("Optimized volume: $(volume(vc_calc.cell)) Å³")
optimized_cell = vc_calc.cell
```

## Complete Example Script

Here's the full workflow in one script:

```julia
using EDDPotentials
using ExtXYZ
using Flux
import EDDPotentials: train!, FluxInterface, create_ensemble, get_fit_data

# 1. Load data
structures = ExtXYZ.load("carbon_data.xyz")
sc = StructureContainer(structures)

# 2. Generate features
cf = CellFeature([:C])
fc = FeatureContainer(cf, sc)

# 3. Split data
fc_train, fc_test = train_test_split(fc)

# 4. Train ensemble
nfeat = nfeatures(cf)
models = []
for i in 1:10
    chain = Chain(Dense(nfeat, 20, tanh), Dense(20, 1))
    itf = FluxInterface(chain)
    train!(itf, fc_train, fc_test; maxIter=300, show_progress=false, earlystop=30)
    push!(models, itf)
end

# 5. Create ensemble
x_train, y_train = get_fit_data(fc_train)
ensemble = create_ensemble(models, x_train, y_train)

# 6. Evaluate
results = TrainingResults(ensemble, fc_test)
println(results)

# 7. Create calculator (requires a structure)
test_structure = structures[1]
calc = NNCalc(test_structure, cf, ensemble)

# 8. Save for later use
using JLD2
save("carbon_potential.jld2", "calc", calc, "cf", cf)

println("Potential ready for use!")
```

## Tips for Better Potentials

**Training Data Quality**:
- Include diverse structures (different densities, coordinations, distortions)
- More data is usually better (aim for 500+ structures for simple systems)
- Balance your dataset - don't have 90% graphite, 10% diamond

**Feature Selection**:
- Start with defaults, only increase complexity if RMSE is too high
- Longer cutoffs capture more environment but increase cost
- Three-body features are crucial for angular information

**Neural Network Architecture**:
- Shallow networks (1-2 layers, 20-30 neurons) work well for EDDP
- Deeper networks rarely improve accuracy significantly
- tanh activation is standard for EDDP

**Training**:
- Always use early stopping to prevent overfitting
- If test RMSE diverges from train, reduce network size or add more data
- Train ensembles for production use - single models are unreliable

**Validation**:
- Always test on structures similar to your target application
- Check force errors if using for MD or optimization
- Compare energy rankings (Spearman) not just absolute errors

## Next Steps

Now that you have a working potential, you can:
- Run molecular dynamics simulations (see ASE integration)
- Perform crystal structure prediction (see AIRSS workflows)
- Calculate phonons (see Phonopy integration)
- Use with AtomsCalculators.jl for standardized interface

See the main documentation for advanced topics like:
- Force training (including forces in the loss function)
- Custom neural network architectures
- Automated potential building workflows
- Integration with structure search tools
