# Understanding Iterative Potential Building

This guide explains how EDDPotentials.jl's automated iterative workflow builds machine learning potentials through active learning. Read this before using `link!()` to understand what happens during each iteration and how to monitor and configure the process.

## Overview

### What is Active Learning for EDDP?

Active learning is an iterative approach where:
1. A model is trained on currently available data
2. The model searches for structures where predictions are uncertain
3. Those structures are calculated with DFT
4. The model is retrained with the expanded dataset
5. The process repeats until convergence criteria are met

The model guides its own training data generation by identifying regions of configuration space where it lacks knowledge.

### When to Use `link!()` vs Manual Training

**Use `link!()` when**:
- Exploring chemical spaces (binary, ternary systems) across compositions
- Automated DFT infrastructure is available (CASTEP + AIRSS)
- Performing crystal structure prediction workflows

**Use manual training when**:
- Working with a pre-existing training dataset
- Targeting specific regions of configuration space
- Using non-CASTEP DFT codes
- Full control over data selection is required

## The Active Learning Loop

This is the core of iterative potential building. Understanding Generation 0 vs Generation N is crucial.

### Generation 0: Cold Start

Generation 0 creates the initial training dataset without any existing model.

**Step 1: Generate Random Structures**

Uses AIRSS `buildcell` to create completely random structures:

```julia
using EDDPotentials
link!(Builder("link.toml"))
```

During Generation 0, `buildcell` generates `n_initial` structures (default: 500) from your SEARCH.cell seed file. These structures are:
- **Unrelaxed**: Atomic positions haven't been optimized
- **Diverse**: Random density, coordination, symmetry
- **Unbiased**: No model influences structure generation

**Why unrelaxed structures?**

Unrelaxed structures include high-energy configurations (atoms too close, stretched bonds, etc.) that would not be present if structures were relaxed to local minima. These configurations provide information about energy gradients and regions outside local energy minima.

**Step 2: DFT Calculations**

All `n_initial` structures get single-point energy calculations (no relaxation):

```
gen0/ (random .res files) → DFT → gen0-dft/ (.res files with energies)
```

**Step 3: Train First Ensemble**

The workflow trains 256 neural networks with different random initializations, then combines them using Non-Negative Least Squares (NNLS) weighting. This ensemble becomes the model for Generation 1.

**File Structure After Generation 0**:
```
workdir/
├── gen0/              # Initial random structures (before DFT)
├── gen0-dft/          # After DFT (with energies)
├── training/
│   ├── gen-0-dataset.jld2           # Features + energies
│   ├── gen-0-model-xxxx.jld2        # Individual models (256 files)
│   └── gen-0-dataset.yaml           # Structure labels
└── ensemble-gen0.jld2                # Final ensemble for next iteration
```

### Generation N: Model-Guided Exploration (N ≥ 1)

Starting from Generation 1, each iteration uses the **previous generation's model** to guide structure search.

**Step 1: Random Structure Search (RSS) with Model**

The workflow generates `per_generation` structures (default: 100) and relaxes them using the EDDP model:

```julia
# Internally, link!() does this:
ensemble = load_ensemble(builder, N-1)  # Load previous model
run_rss(seed_file, ensemble, cell_features; max=100)
```

The model proposes structures that are locally stable according to its current predictions.

**Step 2: Structure Filtering**

Structures are filtered by ensemble uncertainty (standard deviation across models):
- ensemble_std < min: All models agree, structure skipped
- min < ensemble_std < max: Models disagree, structure kept for DFT
- ensemble_std > max: Large disagreement, structure skipped as potentially pathological

**Step 3: Shake Structures**

For each relaxed structure, create `shake_per_minima` randomly perturbed variants (default: 10):

```
1 relaxed structure → shake atoms → 10 shaken variants
100 relaxed → 1000 shaken variants → total: 1100 structures
```

**Why shaking?**

EDDP trains on energies, not forces. Shaking creates structures near local minima with slightly different energies, providing information about energy gradients without requiring explicit force training.

**Step 4: DFT Calculations**

All relaxed + shaken structures get single-point DFT calculations:

```
gen1/ (1100 structures) → DFT → gen1-dft/ (.res files with energies)
```

**Step 5: Accumulative Training**

Train on ALL data from Generation 0 to N:

```julia
all_structures = [gen0_dft, gen1_dft, gen2_dft, ..., genN_dft]
train_ensemble(all_structures)
```

**Why accumulative?**

Training only on new data can cause models to lose accuracy on previously learned regions (catastrophic forgetting). Including all previous data maintains model performance across the entire explored configuration space.

**File Structure After Generation N**:
```
workdir/
├── gen0/, gen0-dft/              # Generation 0 data
├── gen1/, gen1-dft/              # Generation 1 data
├── ...
├── genN/, genN-dft/              # Current generation data
├── training/
│   ├── gen-N-dataset.jld2        # ALL data (Gen 0→N)
│   ├── gen-N-model-*.jld2        # New models (256 files)
│   └── ...
└── ensemble-genN.jld2             # Latest ensemble
```

### The Active Learning Cycle Visualization

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EDDP Active Learning Cycle                       │
│                                                                     │
│  START (Gen 0)                                                      │
│     │                                                               │
│     ▼                                                               │
│  ┌──────────────┐       ┌──────────────┐       ┌──────────────┐   │
│  │   Generate   │       │   Run DFT    │       │    Train     │   │
│  │   Random     │──────>│  Calculate   │──────>│   Ensemble   │   │
│  │  Structures  │       │   Energies   │       │   (256 NN)   │   │
│  │(n_initial=500│       │(single-point)│       │(ALL data 0→N)│   │
│  └──────────────┘       └──────────────┘       └──────────────┘   │
│                                                         │           │
│  ┌──────────────────────────────────────────────────────┘           │
│  │                                                                  │
│  ▼  Generation N (N ≥ 1)                                           │
│  ┌──────────────┐       ┌──────────────┐       ┌──────────────┐   │
│  │  RSS + Model │──────>│    Shake     │──────>│   Run DFT    │   │
│  │ (per_gen=100)│       │   Near Min   │       │  Calculate   │   │
│  │+ Filter by   │       │×shake_per_min│       │   Energies   │   │
│  │ensemble std  │       │    (×10)     │       │              │   │
│  └──────────────┘       └──────────────┘       └──────────────┘   │
│        │                                                 │          │
│        └─────────────────────────────────────────────────┘          │
│                                                                     │
│  CONVERGENCE CHECK: Has model stopped improving?                   │
│     - RMSE stable for multiple generations                         │
│     - Walk-forward test shows good generalization                  │
│     - N ≥ max_iterations                                           │
│                                                                     │
│  If YES → STOP, use ensemble-genN.jld2                             │
│  If NO  → Continue to Generation N+1                               │
└─────────────────────────────────────────────────────────────────────┘
```

### Generation 0 vs Generation N Comparison

```
┌────────────────────────────────┬────────────────────────────────────┐
│      GENERATION 0              │     GENERATION N (N ≥ 1)          │
├────────────────────────────────┼────────────────────────────────────┤
│ buildcell (random)             │ Load previous ensemble model       │
│          ↓                     │              ↓                     │
│ No relaxation                  │ RSS with model relaxation          │
│          ↓                     │              ↓                     │
│ n_initial structures           │ Relax per_generation structures    │
│ (typically 500)                │ (typically 100)                    │
│          ↓                     │              ↓                     │
│ DFT single-point               │ Filter by ensemble uncertainty     │
│          ↓                     │              ↓                     │
│ Train first ensemble           │ Shake: ×shake_per_minima variants  │
│ (on Gen 0 data only)           │ (typically ×10 → 1000 total)       │
│                                │              ↓                     │
│                                │ DFT single-point on all variants   │
│                                │              ↓                     │
│                                │ Train on ALL data (Gen 0→N)        │
│                                │ (accumulative learning)            │
└────────────────────────────────┴────────────────────────────────────┘
```

### Convergence

**What does "converged" mean?**

A potential has converged when adding more iterations does not significantly change model performance. Indicators of convergence:

1. **Test RMSE plateaus**: Error on test set stops decreasing
2. **High Spearman correlation**: Model ranks structures correctly
3. **Walk-forward test stability**: Model from Gen N-1 accurately predicts Gen N data
4. **Duplicate structures**: RSS repeatedly finds the same local minima

**Monitoring During Training**:

```julia
# Check current iteration
readdir("workdir")  # Lists: gen0/, gen1/, ..., genN/

# Load latest ensemble
ensemble = load("workdir/ensemble-gen4.jld2", "model")

# Check training history (printed to stdout during link!)
# Look for lines like:
# Iter: 300  RMSE Train 0.00234 eV | Test 0.00456 eV
```

**When to Stop Iterations**:

The workflow stops automatically when:
- **max_iterations** reached (default: 10)
- **STOP file** detected: Create `workdir/STOP` for graceful exit

Manual inspection of results is recommended to assess convergence.

**Walk-Forward Testing**:

This validation technique tests model generalization to new data:

```julia
# Uses model from Gen N-1 to predict structures from Gen N
walk_forward_tests(builder)
```

RMSE on Gen N data is compared to training RMSE from Gen N-1. Large differences indicate overfitting or expanding configuration space.

## Configuration Guide

The `link.toml` file controls all aspects of iterative building. Here's an annotated minimal configuration:

### Essential Parameters

```toml
[state]
# Core iteration control
iteration = 0                    # Starting iteration (auto-detected from files)
max_iterations = 10              # Stop after this many generations
n_initial = 500                  # Generation 0 structures (binary systems)
per_generation = 100             # Generation N structures (N≥1)
shake_per_minima = 10            # Shaken variants per relaxed structure

# File paths
workdir = "."                    # All outputs go here
seedfile = "C-SEARCH.cell"       # buildcell input for RSS
seedfile_calc = "C.cell"         # CASTEP .cell/.param base name

# DFT execution mode
dft_mode = "crud"                # "crud", "disp-castep", "mock", etc.

# Elemental energies (MUST be pseudoatomic energies)
[state.elemental_energies]
C = -147.4951                    # From QC5 CASTEP OTFG potential

# DFT calculation settings
[state.dft_kwargs]
mpinp = 8                        # MPI processes per CASTEP job
nparallel = 8                    # Parallel CASTEP jobs
```

**Key Parameter Choices**:

- **n_initial**: Default 500 for binary systems, 5000 for ternary systems
- **per_generation**: Default 100 for binary systems, 1000 for ternary
- **shake_per_minima**: Default 5-10; higher values increase DFT cost
- **max_iterations**: Default 5-10; adjust based on convergence monitoring

### Feature Configuration

```toml
[cf]
elements = ["C"]                 # Elements in your system
geometry_sequence = true         # Use geometric spacing for powers

# Two-body features
p2 = [2, 10, 5]                  # [start, end, count]: 2,3.6,6.5,10 (5 powers)
rcut2 = 6.0                      # Cutoff radius (Å)

# Three-body features
q3 = [2, 10, 5]                  # Angular sensitivity powers
```

See [Understanding Features](tutorial_features.md) for detailed guidance.

### Training Parameters

```toml
[trainer]
nmodels = 128                    # Number of models in ensemble
num_workers = 8                  # Parallel training processes
external = true                  # Use separate Julia processes

# Neural network architecture
n_nodes = [20]                   # Hidden layer sizes [20] = 1 layer with 20 neurons
training_mode = "manual_backprop"  # "manual_backprop" or "linear"

# Training stopping criteria
max_iter = 300                   # Maximum training iterations per model
earlystop = 30                   # Stop if no improvement for N iterations

# Data splits
train_split = [0.8, 0.1, 0.1]    # [train, test, validation] ratios
```

**Key Choices**:

- **nmodels**: Default 256; can be reduced to 64 for testing
- **num_workers**: Set based on available CPU cores
- **n_nodes**: [20] for single hidden layer, [20,20] for two layers
- **external**: true for parallel training (production), false for sequential (debugging)

### RSS Parameters

```toml
[state]
rss_pressure_gpa = 1.0           # Positive pressure prevents empty cells
core_size = 1.2                  # Hard-core radius (< bond length)
rss_nprocs = 16                  # Parallel RSS processes

# Active learning filters
ensemble_std_min = -1.0           # Minimum uncertainty to keep. A negative value disables it.
ensemble_std_max = 0.2          # Maximum uncertainty (filters pathological)
```

**Explanation**:

- **rss_pressure_gpa**: Positive pressure (0.5-2.0 GPa) to reduce the chance of having *holes* in structures 
- **core_size**: Set below typical bond length to prevent atom overlap
- **ensemble_std filters**: Keep structures where models disagree (min < std < max)

### DFT Mode Selection

```toml
[state]
dft_mode = "crud"                # Choose calculator type
```

**Available modes**:

1. **"crud"**: Local CASTEP via crud.pl
   - Monitors `hopper/` directory
   - Runs CASTEP when .res files appear
   - Moves results to `good_castep/`

2. **"crud-queue"**: CASTEP via job scheduler
   - Submits jobs to SLURM queue

3. **"mock"**: Dummy calculator for testing
   - Returns random energies
   - For testing workflow without DFT

4. **"disp-castep"**: Distributed CASTEP via Distribute Structure Prediction Package (DISP)
   - For multi-node clusters
   - Automatic job distribution
   - Requires DISP setup

## Monitoring and Troubleshooting

### Progress Monitoring

**Check Current Iteration**:

```julia
using EDDPotentials

# Method 1: Look at directory structure
readdir("workdir")
# Output: ["gen0", "gen0-dft", "gen1", "gen1-dft", "ensemble-gen0.jld2", ...]

# Method 2: Check for ensemble files
glob("workdir/ensemble-gen*.jld2")
# Output: ["ensemble-gen0.jld2", "ensemble-gen1.jld2", "ensemble-gen2.jld2"]
# Currently on Generation 3
```

**Monitor Training Output**:

During `link!()`, watch stdout for:

```
Iteration 4: Generating structures...
RSS generated 100 structures
Running DFT calculations...
DFT complete: 1100/1100 structures (100 relaxed + 1000 shaken)
Training ensemble...
Iter: 50  2.1s elapsed  RMSE Train 0.00345 eV | Test 0.00523 eV
Iter: 100  4.3s elapsed  RMSE Train 0.00234 eV | Test 0.00456 eV
...
Ensemble saved: ensemble-gen4.jld2
```

**File Structure**:

```
workdir/
├── .eddp_builder               # UUID (persistent across runs)
├── C-SEARCH.cell               # Seed file copy
├── C.cell, C.param             # CASTEP inputs
├── link.toml                   # Configuration
│
├── gen0/                       # Generation 0 random structures
│   └── C-*.res (500 files)
├── gen0-dft/                   # After DFT
│   └── C-*.res (with energies)
│
├── gen1/                       # Generation 1 RSS structures
│   └── C-*.res (1100 files: 100 relaxed + 1000 shaken)
├── gen1-dft/                   # After DFT
│
├── training/                   # Training artifacts
│   ├── gen-0-dataset.jld2
│   ├── gen-0-model-xxxx.jld2 (256 files)
│   ├── gen-1-dataset.jld2
│   ├── gen-1-model-yyyy.jld2 (256 files)
│   └── ...
│
├── ensemble-gen0.jld2          # Checkpoints
├── ensemble-gen1.jld2
└── ...
```

### Common Issues

**Problem: Iteration stalled, no progress**

Symptoms: No new files appearing, crud.pl running but not finishing

Solutions:
1. Check `crud_work/hopper/` directory: Should be empty when DFT completes
2. Check `crud_work/good_castep/` directory: Should contain results
3. Look for failed CASTEP calculations:
   ```bash
   grep -r "error" crud_work/workdir/bad_castep/*.castep
   ```
4. Check disk space: `df -h`
5. Re-run failed CASTEP calculation and check the content `*.err` file. CASTEP DOST NOT print error messages to STDOUT.

**Problem: Too many duplicate structures**

Symptoms: RSS repeatedly generates the same structures

This indicates the model has explored local configuration space. Options:
1. Stop iterations (potential convergence)
2. Adjust RSS parameters to explore different regions:
   ```toml
   ensemble_std_min = 0.02           # Minimum uncertainty to keep, bias towards unseen structures.
   ```

### Debugging Tips

**Graceful Exit with STOP File**:

```bash
touch workdir/STOP
```

The workflow checks for this file before each iteration. When detected:
1. Completes current iteration
2. Saves ensemble
3. Exits cleanly

**Resume Interrupted Workflow**:

The workflow automatically detects the current iteration and resumes:

```julia
# If interrupted at Generation 3, just restart
link!(Builder("link.toml"))
# Will see: "Resuming from iteration 3" (auto-detected from ensemble files)
```

## Understanding the Components

This section explains what's happening under the hood. Starting users can skip this initially.

### Builder and State

The **Builder** object orchestrates the entire workflow. It contains:

```julia
struct Builder
    cf::CellFeature              # Feature specification
    state::BuilderState          # Iteration state and paths
    trainer::TrainingOption      # Training configuration
    rss::RssSetting              # RSS configuration
    # ...
end
```

**BuilderState** tracks:
- `iteration`: Current iteration number (0-based, auto-detected)
- `workdir`: Where all files go
- `max_iterations`: Stop condition
- `seedfile`, `seedfile_calc`: File names for buildcell and CASTEP
- All DFT and RSS parameters

**UUID Persistence**:

The file `workdir/.eddp_builder` stores a unique identifier for this project. This enables:
- Reliable restart detection
- Avoiding mixing data from different projects
- Tracking provenance of ensemble files

### Training Pipeline

**Step 1: Dataset Creation and Splitting**

```julia
# Load ALL structures from Gen 0 to current
all_structures = load_structures(builder, 0:N)

# Compute features
fc = FeatureContainer(cell_feature, all_structures)

# Split into train/test/validation (default: 80%/10%/10%)
fc_train, fc_test, fc_valid = split(fc, train_split...)

# Save to disk
save("gen-N-dataset.jld2", Dict("train"=>fc_train, "test"=>fc_test, "valid"=>fc_valid))
```

**Step 2: Parallel Model Training**

The workflow trains `nmodels` independently in parallel:


Each model gets:
- Different random initialization
- Same training data
- Independent Levenberg-Marquardt optimization

**Step 3: NNLS Ensemble Weighting**

After all models are trained, combine them:

```julia
# Load all individual models

# Compute optimal weights using Non-Negative Least Squares
ensemble = create_ensemble(builder)
# Uses Fast NNLS algorithm to minimize:
#   argmin ||A·w - y||²  subject to w ≥ 0
# where A[i,j] = prediction of model j on structure i

# Models with near-zero weights are removed
# Typical result: 5-10 models with meaningful weights
```

### RSS Integration

**buildcell Seed Files**:

The `seedfile` parameter in `[state]` specifies the AIRSS seed:

```
#TARGVOL=20-30           # Volume per atom range
#SPECIES=C               # Elements to include
#NATOM=4-8               # Atoms per cell
#SYMMOPS=0               # No symmetry (important!)
#SLACK=0.25
#OVERLAP=0.1
#COMPACT
#MINSEP=1.5 C-C          # Minimum separation
```

During Generation N, RSS calls:
```julia
_run_rss(seedfile, ensemble, cell_features;
        max=builder.state.per_generation)
```

### DFT Calculator Abstraction

The `dft_mode` parameter selects a calculator. All calculators share this interface:

```julia
calculator(indir, outdir) -> success::Bool
```

**crud.pl Workflow** as example code:

```julia
function run_crud(workdir, seedfile_calc, indir, outdir; nparallel=8, mpinp=8)
    hopper = joinpath(workdir, "hopper")
    good_castep = joinpath(workdir, "good_castep")

    # Copy .res files to hopper
    cp.(glob(indir * "/*.res"), hopper)

    # Launch parallel crud.pl instances
    @sync for i = 1:nparallel
        @async run(`crud.pl -mpinp $mpinp`, dir=workdir)
    end

    # crud.pl workflow:
    # 1. Check hopper/ for .res files
    # 2. Move one to workdir
    # 3. Build CASTEP input from .res + seedfile_calc.cell/.param
    # 4. Run CASTEP
    # 5. Move result to good_castep/
    # 6. Repeat until hopper/ is empty

    # Move results to output
    mv.(glob(good_castep * "/*.res"), outdir)
end
```

**hopper/ and good_castep/ Pattern used by crud.pl**:

This is the standard pattern for batch refine-calculations for AIRSS:
- **hopper/**: Queue of structures to calculate
- **good_castep/**: Successful calculations
- **bad_castep/**: Failed calculations

Typically, AIRSS searches are done with DFT settings that optimizes the speed. Hence, the final results
should need re-optimized with higher precision DFT calculation for a final ranking of the obtained
structures.

In EDDPotentials.jl, we reuse the `crud.pl` script to perform many **singlepoint** calculations in parallel
in order to generate the training data.

Multiple crud.pl instances can run simultaneously, each taking structures from hopper.
Unique selection of the structures are guaranteed by the atomic `mv` Linux command.

**Parallel DFT Execution**:

With `nparallel=8` and `mpinp=8`:
- 8 concurrent CASTEP jobs
- Each using 8 MPI processes
- Total: 64 CPU cores utilized

Adjust based on your hardware: `nparallel × mpinp ≤ total_cores`.

### Computational Resources

**CPU Allocation Strategy**:

For a 64-core node:

```toml
[state.dft_kwargs]
nparallel = 8                # 8 concurrent CASTEP jobs
mpinp = 8                    # 8 cores per job

[trainer]
num_workers = 8              # 8 parallel training processes
num_threads_per_worker = 8   # 8 threads per worker (for BLAS)
```

**When to Use External Training**:

```toml
[trainer]
external = true              # Launch separate Julia processes
```

`external=true` (default) enables parallel training. `external=false` enables sequential training for debugging.

**GPU Considerations**:

```toml
[trainer]
use_cuda = true              # Enable if CUDA available
cuda_visible_devices = "0,1,2,3"  # Only use these GPUs for training
```

### Data Management

**What to Archive**:

After project completes, keep:
- `ensemble-genN.jld2` (final model)
- `link.toml` (configuration)
- `C-SEARCH.cell`, `C.cell`, `C.param` (inputs)
- `gen*-dft/*.res` (DFT-calculated structures with energies)
- `.eddp_builder` (UUID for provenance)

**What to Delete Safely**:

Can delete to save space:
- `training/gen-N-model-*.jld2` (individual models, ensemble has combined result)
- `training/gen-N-dataset.jld2` (can recreate from gen*-dft/)
- `hopper/`, `good_castep/` (temporary DFT directories)

With `clean_individual_models=true`, individual models are deleted automatically.

## Advanced Topics

### Multi-Composition Searches

When searching across compositions (e.g., Li₁₋ₓMₓ), elemental energies are crucial:

```toml
[state.elemental_energies]
Li = -194.8254               # MUST be pseudoatomic energies
M = -1673.6962               # From your pseudopotential library
```

**Why pseudoatomic energies?**

These are the energy of an isolated atom in the pseudopotential approximation. They define the zero-point for formation energies.
Using these reference energies allows to network to learn the *interaction* energies which have phyiscal
meanings.
Otherwise, different elements may have wildly different reference energies (from tens of electron volts to thousands of electron volts), causing the optimizer to spend a lot of time and effort for learning them.

**FOCUS Parameter in buildcell**:

Control which compositions to sample:

```
#FOCUS=1                # Only pure elements
#FOCUS=2               # Binary compounds only
#FOCUS=3              # Include ternary only
#FOCUS=2,3            # Include ternary and binary only
#FOCUS={1,2,2,2,2,3,3,3}   # Randomly select 1,2,3 element compounds in 1:4:3 possiblity
```

For Li-Ti-O system, use `#FOCUS=2,3` to sample LiₓTiᵧOᵤ, LiₓTiᵧ,  TiᵧOᵤ LiₓOᵤ ompounds.

### Custom Calculators

**Using Mock Calculator for Testing**:

```toml
[state]
dft_mode = "mock"
```

Returns random energies instantly. Perfect for testing workflow without DFT overhead.

**Beyond CASTEP**:

The calculator abstraction allows other DFT codes. The simplest approach is to implement an external program compatible with the `acrud` function:

```julia
# Use acrud with custom external calculator
acrud(workdir, seedfile_calc, indir, outdir, "my_calculator")
```

Your external program should:
1. Read `.res` files from the input directory
2. Perform DFT calculations
3. Write results as `.res` files with energies to the output directory
4. Return exit code 0 on success

See `acrud` documentation for the expected interface specification.

### Walk-Forward Testing

This validation technique is built into the workflow:

```julia
using EDDPotentials

# Load your builder
builder = Builder("link.toml")

# Run walk-forward tests on all iterations
results = walk_forward_tests(builder)

# Check results
for (iter, rmse, mae) in results
    println("Gen $iter: RMSE=$(rmse) eV/atom, MAE=$(mae) eV/atom")
end
```

**What It Measures**:

For each iteration N:
1. Load ensemble from Generation N-1
2. Predict energies of structures from Generation N
3. Compare to DFT energies
4. Report RMSE and MAE

**Good walk-forward results**: RMSE comparable to training RMSE (within 2×)

**Bad walk-forward results**: RMSE much higher than training → model not generalizing, needs more diverse data, iterative training process should not be stopped.

## Summary

Iterative EDDP building with `link!()` automates potential creation through active learning:

1. **Generation 0**: Create initial dataset from random structures
2. **Generation N**: Use model to guide structure search, train on accumulated data
3. **Convergence**: Stop when model performance plateaus
4. **Result**: Ensemble model covering explored configuration space

**Key Concepts**:
- Focuses DFT calculations on uncertain structures
- Accumulative training prevents catastrophic forgetting
- Shaking provides gradient information without explicit force training
- NNLS ensemble weighting combines model predictions

**Getting Started**:
1. Prepare seed files (C-SEARCH.cell, C.cell, C.param)
2. Create link.toml configuration
3. Test with mock calculator
4. Run with DFT calculator
5. Monitor convergence metrics

**Reference**:
- Use walk-forward testing to assess generalization
- Archive final ensemble and configuration for reproducibility

