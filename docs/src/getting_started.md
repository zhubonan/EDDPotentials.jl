# Getting Started

This guide will help you build your first machine learning potential using EDDPotentials.jl. No prior experience with interatomic potentials or computational materials science is required.

## Installation

### Prerequisites

EDDPotentials.jl requires Julia version 1.9 or higher. If you don't have Julia installed, download it from [julialang.org](https://julialang.org/downloads/).

To verify your Julia installation:
```bash
julia --version
```

### Installing EDDPotentials.jl

Start Julia and enter the package manager by pressing `]`:

```julia
julia> ]
pkg> add EDDPotentials
```

To verify the installation:
```julia
using EDDPotentials
```

If you see no errors, you're ready to start!

## Core Concepts

### What is EDDP?

EDDP (Ephemeral Data Derived Potential) is a machine learning approach for predicting atomic energies and forces. Instead of running expensive quantum mechanical (DFT) calculations every time you need to evaluate the energy of a structure, you train a neural network to reproduce DFT results. Once trained, the potential can predict energies thousands of times faster than DFT.

EDDP is designed for running random structures searching, rather than performing accurate molecular dynamics simulations.

### Key Components

**Training Data**: A collection of atomic structures with known energies (typically from DFT calculations). The more diverse your training data, the more reliable your potential. The training data is built by generating random structure and relaxing them using the current best model.
This is an iterative process as new regions of the potentials surface getting explored and the models get
better.
Unlike most other machine learning interatomic potentials, EDDP trains only on the energy. The gradients, i.e. forces, are learned implicitly by training with *shaken* structures.

For automated iterative training data generation and potential building, see [Understanding Iterative Potential Building](concepts_iterative_building.md).

**Features**: Mathematical descriptors that encode the local atomic environment around each atom. EDDPotentials.jl uses two types:
- **Two-body features**: Describe pairwise interactions between atoms
- **Three-body features**: Capture angular information (bond angles)

**Neural Network**: A simple feedforward network that learns to map features to atomic energies. The network is typically shallow (1-2 hidden layers) but is trained on many different structures.

**Calculator**: An interface that uses your trained potential to predict energies and forces for new structures, similar to how you would use a DFT calculator.

### When to Use EDDP

EDDP is ideal when you need to:
- Explore thousands of candidate structures (crystal structure prediction)
- Perform geometry optimizations repeatedly on small unit cells.

EDDP is NOT a replacement for DFT when you need:
- Chemical accuracy for a single structure
- Electronic properties (band structure, DOS)
- Properties outside your training data distribution

## A Potential for Carbon

Let's build a simple potential for carbon structures. We'll train on graphite and diamond, then use the potential to predict energies of new carbon configurations.

### Step 1: Prepare Training Data

First, you need structures with energies. In practice, these come from DFT calculations. For this tutorial, assume you have files in the Extended XYZ format:

```julia
using EDDPotentials
using Flux  # For neural network construction
using ExtXYZ  # For loading .xyz files
import EDDPotentials: train!, FluxInterface, create_ensemble, get_fit_data

# Load structures from files
structures_train = ExtXYZ.load("carbon_train.xyz")
structures_test = ExtXYZ.load("carbon_test.xyz")

println("Loaded $(length(structures_train)) training structures")
println("Loaded $(length(structures_test)) test structures")
```

Each structure should have an energy property. If you're starting from scratch, you would run DFT calculations on a diverse set of carbon structures (different densities, coordination environments, etc.).

### Step 2: Create Feature Containers

Next, convert your structures into feature vectors that the neural network can understand:

```julia
# Define features for carbon
# This creates two-body features with powers [2,3,4,5,6] and cutoff 4.0 Å
cf = CellFeature([:C])

# Create containers that compute features for all structures
sc_train = StructureContainer(structures_train)
sc_test = StructureContainer(structures_test)

fc_train = FeatureContainer(cf, sc_train)
fc_test = FeatureContainer(cf, sc_test)

println("Feature dimension: ", nfeatures(cf))
```

The `CellFeature` automatically sets up sensible defaults for carbon. The feature dimension is typically 20-50 depending on the cutoff radius and powers used.

### Step 3: Create and Train a Neural Network

Now create a simple neural network and train it:

```julia
using Flux

# Create a neural network: input → 20 hidden neurons → output
# nfeatures(cf) gives the input size
chain = Chain(
    Dense(nfeatures(cf), 20, tanh),
    Dense(20, 1)
)

# Wrap the Flux chain in an EDDPotentials interface
model = FluxInterface(chain)

# Train the model (this may take a few minutes)
train!(
    model,
    fc_train,
    fc_test;
    maxIter=300,
    show_progress=true,
    earlystop=30
)
```

During training, you'll see output showing the RMSE (root mean square error) for both training and test sets. Training stops early if the test error doesn't improve for 30 iterations.

### Step 4: Evaluate the Model

After training, check how well the model performs:

```julia
# Create results objects for analysis
results_train = TrainingResults(model, fc_train)
results_test = TrainingResults(model, fc_test)

println("Training set performance:")
println(results_train)

println("\nTest set performance:")
println(results_test)
```

This will show you:
- RMSE and MAE (mean absolute error) per atom
- Maximum error and which structure it occurs on
- Spearman correlation (how well the model ranks structures by energy)

### Step 5: Create an Ensemble (Recommended)

Single models can be unreliable. It's better to train multiple models and combine them:

```julia
# Train 10 models with different random initializations
models = []
for i in 1:10
    chain = Chain(Dense(nfeatures(cf), 20, tanh), Dense(20, 1))
    itf = FluxInterface(chain)
    train!(itf, fc_train, fc_test; maxIter=300, show_progress=false)
    push!(models, itf)
end

# Combine into an ensemble (automatically weights by performance)
x_train, y_train = get_fit_data(fc_train)
ensemble = create_ensemble(models, x_train, y_train)

println("Ensemble contains $(length(ensemble.models)) models")
println("Weights: ", ensemble.weights)
```

The ensemble typically performs better than any single model and provides uncertainty estimates.

### Step 6: Use the Potential

Now you can use your potential to predict energies and forces for new structures:

```julia
# Load a new carbon structure
new_structure = ExtXYZ.load("new_carbon.xyz")[1]

# Create a calculator from your ensemble
# NNCalc requires the structure, feature spec, and neural network interface
calc = NNCalc(new_structure, cf, ensemble)

# Predict energy (calculator operates on its internal structure)
energy = eddp_energy(calc)
println("Predicted energy: $energy eV")

# Predict forces
forces = eddp_forces(calc)
println("Max force: $(maximum(abs.(forces))) eV/Å")

# For a different structure, create a new calculator
another_structure = ExtXYZ.load("another_carbon.xyz")[1]
calc2 = NNCalc(another_structure, cf, ensemble)
energy2 = eddp_energy(calc2)
```

You can also use the calculator with other packages like ASE (via PyCall) for molecular dynamics or other simulations.

## Metrics

Understanding the error metrics helps you assess whether your potential is good enough for your application.

### RMSE (Root Mean Square Error)

RMSE is the standard deviation of prediction errors. For carbon potentials, typical values are:
- **< 1 meV/atom**: Excellent - suitable for phase stability predictions
- **1-5 meV/atom**: Good - suitable for structure optimization and MD
- **5-20 meV/atom**: Fair - may be useful for qualitative trends
- **> 20 meV/atom**: Poor - likely unreliable for most applications

Note that 1 meV/atom = 0.001 eV/atom. Energy differences between carbon structures (graphite vs diamond) are ~20 meV/atom, so you need sub-meV accuracy for reliable phase predictions.

### MAE (Mean Absolute Error)

MAE is less sensitive to outliers than RMSE. If MAE is much smaller than RMSE, you have a few structures with very large errors - investigate these carefully.

### Spearman Correlation

Spearman measures how well your potential ranks structures by energy (not the absolute error). A correlation of 0.8-1.0 means the potential gets the relative ordering nearly perfect, even if absolute energies have some error.

For structure prediction, ranking is often more important than absolute energy accuracy.

### Force Errors

If your training data includes forces (from DFT), also check force RMSE:
- **< 0.3 eV/Å**: Acceptable for geometry optimization
- **> 0.5 eV/Å**: May cause issues in simulations

### When is a Potential "Good Enough"?

This depends on your application:
- **Phase stability**: Need < 1 meV/atom RMSE and high Spearman correlation. **In practice, you should always use DFT for this.**
- **Molecular dynamics**: Need force RMSE < 0.1 eV/Å and energy conserved over time.
- **Geometry optimization**: Need force RMSE < 0.3 eV/Å; energy accuracy less critical
- **Structure screening**: Need high Spearman correlation; absolute error less critical

Always test your potential on structures similar to those you'll study. A potential trained on bulk structures may fail for surfaces or defects.

## Properties

Once you have a trained potential, you can use it to predict various properties.

### Structure Optimization

Optimize the geometry of a structure to its local energy minimum:

```julia
# Load initial structure
initial_structure = ExtXYZ.load("carbon_initial.xyz")[1]

# Create calculator with the initial structure
calc = NNCalc(initial_structure, cf, ensemble)
initial_energy = eddp_energy(calc)
println("Initial energy: $initial_energy eV")

# Optimize atomic positions (fixed cell)
# relax! modifies the calculator's internal structure in-place
relax!(calc; fmax=0.01, max_steps=100)
optimized_energy = eddp_energy(calc)
println("Optimized energy: $optimized_energy eV")

# The optimized structure is stored in calc.cell
optimized_structure = calc.cell

# For variable cell optimization, use VariableCellCalc
using EDDPotentials: VariableCellCalc
vc_calc = VariableCellCalc(initial_structure, cf, ensemble)
relax!(vc_calc; fmax=0.01, max_steps=200)
println("Optimized volume: $(volume(vc_calc.cell)) Å³")
```

### Crystal Structure Prediction

EDDPotentials.jl can perform crystal structure prediction using the AIRSS (Ab Initio Random Structure Searching) approach:

```julia
# Generate random structures and optimize them
# This requires the AIRSS buildcell tool to be installed
results = run_rss(
    "my_seed.cell",
    ensemble,
    cf;  # cell_features
    num_structures=100,  # Number of random structures to try
    cell_volume=20:30,   # Volume per atom range
    max_atoms=8         # Maximum atoms in cell
)

# Results are sorted by energy
best_structure = results[1]
```

This is useful for discovering new crystal structures or finding ground states.

### Limitations of EDDP

Be aware of what EDDP cannot do:

**Extrapolation**: EDDP potentials are only reliable for structures similar to the training data. Testing on high-pressure structures when trained on ambient pressure will likely fail.

**Chemical Accuracy**: EDDP errors are typically 1-10 meV/atom. For reactions where energy differences are < 10 meV/atom, use DFT instead.

**Electronic Properties**: EDDP only predicts total energy and forces. For band structures, DOS, or magnetic properties, you need DFT.

Despite these limitations, EDDP potentials are powerful tools for exploring large configuration spaces where running DFT thousands of times would be impractical.