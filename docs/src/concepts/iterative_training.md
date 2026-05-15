# Iterative Training

This guide explains how EDDPotentials.jl builds machine learning potentials through iterative training. Understanding this process is essential before using `link!()` for automated potential building.

## Overview

### What is Iterative Training?

Iterative training builds EDDP potentials through cycles of training and exploration:

1. Train model on currently available data
2. Use model to search for structures where predictions are uncertain
3. Calculate DFT energies for those structures
4. Retrain model with expanded dataset
5. Repeat until convergence

The model guides its own training data generation by identifying regions of configuration space where it lacks knowledge.

### When to Use link!() vs Manual Training

**Use link!() when:**
- Exploring chemical spaces across compositions
- Automated DFT infrastructure available (CASTEP + AIRSS)
- Performing crystal structure prediction workflows

**Use manual training when:**
- Working with pre-existing training dataset
- Targeting specific configuration space regions
- Using non-CASTEP DFT codes
- Full control over data selection required

## The Training Cycle

Understanding Generation 0 vs Generation N is crucial.

### Generation 0: Cold Start

Generation 0 creates the initial training dataset without any existing model.

**Workflow:**

```julia
using EDDPotentials
link!(Builder("link.toml"))
```

**Steps:**

1. **Generate Random Structures**: AIRSS `buildcell` creates `n_initial` structures (default: 500)
   - Unrelaxed (includes high-energy configurations)
   - Diverse (random density, coordination, symmetry)
   - Unbiased (no model guidance)

2. **DFT Calculations**: Single-point energy calculations for all structures

3. **Train First Ensemble**: Train 256 neural networks with different initializations, combine using NNLS weighting

**File Structure:**
```
workdir/
├── gen0/              # Initial random structures
├── gen0-dft/          # After DFT (with energies)
├── training/
│   ├── gen-0-dataset.jld2
│   ├── gen-0-model-xxxx.json  (256 files)
│   └── gen-0-dataset.yaml
└── ensemble-gen0.json
```

### Generation N: Model-Guided Exploration (N ≥ 1)

Starting from Generation 1, each iteration uses the previous generation's model to guide structure search.

**Workflow:**

```julia
ensemble = load_ensemble(builder, N-1)
run_rss(seed_file, ensemble, cell_features; max=100)
```

**Steps:**

1. **Random Structure Search (RSS)**: Generate `per_generation` structures (default: 100) and relax with EDDP model

2. **Structure Filtering**: Filter by ensemble uncertainty (std across models):
   - `std < min`: All models agree → skip
   - `min < std < max`: Models disagree → keep for DFT
   - `std > max`: Large disagreement → skip (potentially pathological)

3. **Shake Structures**: For each relaxed structure, create `shake_per_minima` perturbed variants (default: 10)
   - Example: 100 relaxed → 1000 shaken → 1100 total structures
   - Provides gradient information for energy-only training

4. **DFT Calculations**: Single-point energies for all relaxed + shaken structures

5. **Accumulative Training**: Train on ALL data from Generation 0 to N
   ```julia
   all_structures = [gen0_dft, gen1_dft, ..., genN_dft]
   train_ensemble(all_structures)
   ```
   - Prevents catastrophic forgetting
   - Maintains accuracy across entire explored space

**File Structure:**
```
workdir/
├── gen0/, gen0-dft/
├── gen1/, gen1-dft/
├── ...
├── genN/, genN-dft/
├── training/
│   ├── gen-N-dataset.jld2     # ALL data (Gen 0→N)
│   ├── gen-N-model-*.json     (256 files)
└── ensemble-genN.json
```

### Generation 0 vs Generation N

| Generation 0 | Generation N (N ≥ 1) |
|--------------|---------------------|
| buildcell (random) | Load previous ensemble |
| No relaxation | RSS with model relaxation |
| n_initial structures (typically 500) | per_generation structures (typically 100) |
| DFT single-point | Filter by ensemble uncertainty |
| Train first ensemble | Shake: ×shake_per_minima variants |
| (on Gen 0 data only) | DFT single-point on all variants |
| | Train on ALL data (Gen 0→N) |

### Convergence

A potential has converged when adding more iterations does not significantly improve model performance.

**Indicators of convergence:**
1. Validation RMSE plateaus
2. High Spearman correlation (> 0.98)
3. Walk-forward test stability
4. Duplicate structures in RSS (PES well-sampled)

**Monitoring:**

```julia
# Check current iteration
readdir("workdir")  # Lists: gen0/, gen1/, ..., genN/

# Load latest ensemble
artifact = load_model_artifact("workdir/ensemble-gen4.json")
ensemble = artifact.model

# Training output shows:
# Iter: 300  RMSE Train 0.00234 eV | Test 0.00456 eV
```

**When to stop:**
- `max_iterations` reached (default: 10)
- Create `workdir/STOP` file for graceful exit
- Manual inspection shows convergence

**Walk-Forward Testing:**

Validates model generalization to new data:

```julia
walk_forward_tests(builder)
```

Uses model from Gen N-1 to predict structures from Gen N. Large RMSE differences indicate overfitting or expanding configuration space.

## Configuration Guide

The `link.toml` file controls all aspects of iterative training.

### Essential Parameters

```toml
[state]
iteration = 0                    # Starting iteration (auto-detected)
max_iterations = 10              # Stop after this many generations
n_initial = 500                  # Generation 0 structures (binary)
per_generation = 100             # Generation N structures (N≥1)
shake_per_minima = 10            # Shaken variants per relaxed structure
buildcell_nprocs = 8             # Parallel buildcell jobs for generation 0
build_timeout = 60.0             # Seconds allowed per buildcell launch

workdir = "."                    # All outputs go here
seedfile = "C-SEARCH.cell"       # buildcell input for RSS
seedfile_calc = "C.cell"         # CASTEP .cell/.param base name

dft_mode = "crud"                # "crud", "disp-castep", "mock", etc.

# Elemental energies (MUST be pseudoatomic energies)
[state.elemental_energies]
C = -147.4951                    # From QC5 CASTEP OTFG potential

# DFT calculation settings
[state.dft_kwargs]
mpinp = 8                        # MPI processes per CASTEP job
nparallel = 8                    # Parallel CASTEP jobs
```

**Key parameter choices:**
- **n_initial**: 500 for binary systems, 5000 for ternary systems
- **per_generation**: 100 for binary, 1000 for ternary
- **shake_per_minima**: 5-10; higher values increase DFT cost
- **max_iterations**: 5-10; adjust based on convergence

### Feature Configuration

```toml
[cf]
elements = ["C"]
geometry_sequence = true

# Two-body features
p2 = [2, 10, 5]                  # [start, end, count]
rcut2 = 6.0                      # Cutoff radius (Å)

# Three-body features
q3 = [2, 10, 5]
```

See [Understanding Features](features.md) for detailed guidance.

### Training Parameters

```toml
[trainer]
nmodels = 128                    # Number of models in ensemble
num_workers = 8                  # Parallel training processes
external = true                  # Use separate Julia processes

# Neural network architecture
n_nodes = [20]                   # [20] = 1 layer with 20 neurons
training_mode = "manual_backprop"

# Stopping criteria
max_iter = 300                   # Maximum training iterations
earlystop = 30                   # Stop if no improvement for N iterations

# Data splits
train_split = [0.8, 0.1, 0.1]    # [train, validation, test] ratios
```

**Key choices:**
- **nmodels**: Default 256; reduce to 64 for testing
- **num_workers**: Based on available CPU cores
- **n_nodes**: [20] for single layer, [20,20] for two layers
- **external**: true for parallel (production), false for sequential (debugging)

### RSS Parameters

```toml
[state]
rss_pressure_gpa = 1.0           # Positive pressure prevents empty cells
core_size = 1.2                  # Hard-core radius (< bond length)
buildcell_nprocs = 16            # Parallel buildcell jobs for generation 0
rss_nprocs = 16                  # Parallel RSS processes

# Uncertainty filters
ensemble_std_min = -1.0          # Minimum uncertainty (-1 disables)
ensemble_std_max = 0.2           # Maximum uncertainty

# Optional RSS output pruning for single-composition searches
[state.rss_prune]
enabled = true                   # Disabled by default
pool_size = 100                  # Relaxed candidates per pruning pool
keep_fraction = 0.10             # Keep lowest-energy unique 10%
dedup_tol = 0.10                 # Fingerprint-distance duplicate threshold
```

**Explanation:**
- **rss_pressure_gpa**: 0.5-2.0 GPa reduces holes in structures
- **core_size**: Set below typical bond length to prevent overlap
- **ensemble_std filters**: Keep structures where models disagree
- **rss_prune**: Buffers accepted relaxed structures, ranks by predicted enthalpy per atom, removes fingerprint duplicates, and writes only the low-energy unique subset. `per_generation` remains the number of written structures, so pruning may relax more raw candidates internally.
- **single-composition only**: Pruning is disabled automatically for formula-sampling/random-composition workflows; multi-instance `link!` workers prune independently and log pool diagnostics to their `rss-process-*` stderr logs.

### DFT Mode Selection

```toml
[state]
dft_mode = "crud"
```

**Available modes:**

1. **"crud"**: Local CASTEP via crud.pl
   - Monitors `hopper/` directory
   - Runs CASTEP when .res files appear
   - Moves results to `good_castep/`

2. **"crud-queue"**: CASTEP via job scheduler (SLURM)

3. **"mock"**: Dummy calculator for testing (random energies)

4. **"disp-castep"**: Distributed CASTEP via DISP package

### File-Backed HPC Server-Client Mode

Use this mode when DFT calculations and model training should run as separate
cluster client jobs coordinated through files on a shared filesystem. The master
publishes the active phase in `.eddp-hpc/control.json`; clients read that file
and run either `crud.pl` or a one-model training subprocess.

Minimal `link.toml` settings:

```toml
[state]
mpinp = 8                         # MPI ranks used by each crud.pl child

[trainer]
external = true
nmodels = 128
external_completion_ratio = 0.9

[hpc]
enabled = true
coordination_dir = ".eddp-hpc"
submit_client_jobs = true
dft_client_jobs = 16
training_client_jobs = 8
dft_crud_workdir = "crud-work"
control_poll_interval_seconds = 10.0
client_heartbeat_interval_seconds = 30.0
client_child_grace_seconds = 300.0
client_child_kill_seconds = 30.0
training_slot_stale_seconds = 86400.0

[scheduler]
script = "hpc_client_job.sh"
type = "SLURM"                    # or "SGE"
```

When `[hpc] enabled = true`, the normal DFT and external-training backends are
overridden by the file-backed HPC workflow. You do not need to set
`state.dft_mode = "crud-hpc"` or `trainer.external_mode = "hpc"` in this mode.

The scheduler script launches EDDP client jobs. It must contain the placeholder
`{{EDDP_HPC_CLIENT_CMD}}`, which the master replaces with the correct Julia
client command:

```bash
#!/bin/bash
#SBATCH --job-name=eddp-hpc-client
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

{{EDDP_HPC_CLIENT_CMD}}
```

Start the master with:

```bash
julia --project=. -e 'using EDDPotentials; hpc_master("link.toml")'
```

The master then:

1. Generates structures into `genN/`.
2. Prepares `crud-work/genN/hopper/`, `good_castep/`, and `bad_castep/`.
3. Submits DFT client jobs when `submit_client_jobs = true`.
4. Stops DFT clients after enough calculations reach `good_castep/` or `bad_castep/`.
5. Publishes successful results into `genN-dft/`.
6. Writes `training/gen-N-dataset.jld2`.
7. Submits training clients, each claiming one deterministic model slot.
8. Creates `ensemble-genN.json` after enough model artifacts are available.

If you prefer to submit client jobs yourself, set:

```toml
[hpc]
submit_client_jobs = false
```

Then run clients manually or through your own job script:

```bash
julia --project=. -e 'using EDDPotentials; EDDPotentials.hpc_client()' -- --file link.toml --worker-id 1
```

Client status is written to:

```
.eddp-hpc/workers/<worker-id>.json
```

Training claims and completion markers are stored under:

```
.eddp-hpc/phases/genN-train/slots/
```

To stop the workflow gracefully:

```bash
touch <workdir>/STOP
```

Clients first ask their current child process to exit, wait
`client_child_grace_seconds`, and then force-kill after
`client_child_kill_seconds` if needed.

## Monitoring and Troubleshooting

### Progress Monitoring

**Check Current Iteration:**

```julia
using EDDPotentials

# Method 1: Directory structure
readdir("workdir")
# Output: ["gen0", "gen0-dft", "gen1", "gen1-dft", ...]

# Method 2: Ensemble files
glob("workdir/ensemble-gen*.json")
# Output: ["ensemble-gen0.json", "ensemble-gen1.json", "ensemble-gen2.json"]
```

**Monitor Training Output:**

During `link!()`, watch stdout:

```
Iteration 4: Generating structures...
RSS generated 100 structures
Running DFT calculations...
DFT complete: 1100/1100 structures
Training ensemble...
Iter: 50  RMSE Train 0.00345 eV | Test 0.00523 eV
Iter: 100  RMSE Train 0.00234 eV | Test 0.00456 eV
Ensemble saved: ensemble-gen4.json
```

### Common Issues

**Problem: Iteration stalled, no progress**

Solutions:
1. Check `crud_work/hopper/` (should be empty when DFT completes)
2. Check `crud_work/good_castep/` (should contain results)
3. Look for failed CASTEP calculations:
   ```bash
   grep -r "error" crud_work/workdir/bad_castep/*.castep
   ```
4. Check disk space: `df -h`
5. Check CASTEP `*.err` files (CASTEP doesn't print errors to stdout)

**Problem: Too many duplicate structures**

This indicates the model has explored local configuration space.

Options:
1. Stop iterations (potential convergence)
2. Adjust RSS to explore different regions:
   ```toml
   ensemble_std_min = 0.02    # Bias towards unseen structures
   ```

### Debugging Tips

**Graceful Exit:**

```bash
touch <workdir>/STOP
```

Workflow completes current iteration, saves ensemble, exits cleanly.

**Resume Interrupted Workflow:**

Workflow auto-detects current iteration:

```julia
link!(Builder("link.toml"))
# Output: "Resuming from iteration 3"
```

## Understanding the Components

### Builder and State

The **Builder** object orchestrates the entire workflow:

```julia
struct Builder
    cf::CellFeature              # Feature specification
    state::BuilderState          # Iteration state and paths
    trainer::TrainingOption      # Training configuration
    rss::RssSetting              # RSS configuration
end
```

**BuilderState** tracks:
- `iteration`: Current iteration number (auto-detected)
- `workdir`: Output directory
- `max_iterations`: Stop condition
- `seedfile`, `seedfile_calc`: File names for buildcell and CASTEP
- All DFT and RSS parameters

**UUID Persistence:**

The file `workdir/.eddp_builder` stores a unique identifier for reliable restart detection and provenance tracking.

### Training Pipeline

**Step 1: Dataset Creation**

```julia
# Load ALL structures from Gen 0 to current
all_structures = load_structures(builder)

# Compute features
fc = FeatureContainer(cell_feature, all_structures)

# Split into train/validation/test (default: 80%/10%/10%)
fc_train, fc_val, fc_test = split(fc, train_split...)

# Save to disk
save("gen-N-dataset.jld2", Dict("train"=>fc_train, "validate"=>fc_val, "test"=>fc_test))
```

**Step 2: Parallel Model Training**

Train `nmodels` independently in parallel:
- Different random initialization
- Same training data
- Independent Levenberg-Marquardt optimization

**Step 3: NNLS Ensemble Weighting**

```julia
ensemble = EDDPotentials.create_ensemble(builder)
# Uses Non-Negative Least Squares to minimize:
#   argmin ||A·w - y||²  subject to w ≥ 0
# where A[i,j] = prediction of model j on structure i

# Models with near-zero weights removed
# Typical result: 5-10 models with meaningful weights
```

### DFT Calculator Abstraction

The `dft_mode` parameter selects a calculator. All calculators share this interface:

```julia
calculator(indir, outdir) -> success::Bool
```

**crud.pl Workflow:**

Structures from `genx` folders are moved into a dedicated working directory `crud-work` to run DFT single-point energy
calculation using `curd.pl`.

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

    # crud.pl: hopper/ → workdir → good_castep/

    # Move results to output
    mv.(glob(good_castep * "/*.res"), outdir)
end
```

Inside the `crud-work` folder, there three folders.

- **hopper/**: Queue of structures to calculate
- **good_castep/**: Successful calculations
- **bad_castep/**: Failed calculations

Multiple crud.pl instances run simultaneously, each taking structures from hopper (atomic `mv` guarantees unique selection).

On HPC, use either `crud-queue` for the legacy scheduler-backed DFT runner or
enable `[hpc]` for the file-backed server-client workflow. In HPC mode the
master prepares the shared `crud-work/genN/hopper/` directory, and client jobs
launch `crud.pl` when the master publishes the DFT phase.


### Data Management

**Keep after project completes:**
- `ensemble-genN.json` (final model)
- `link.toml` (configuration)
- `C-SEARCH.cell`, `C.cell`, `C.param` (inputs)
- `gen*-dft/*.res` (DFT-calculated structures)
- `.eddp_builder` (UUID for provenance)

**Can delete to save space:**
- `training/gen-N-model-*.json` (individual models)
- `training/gen-N-dataset.jld2` (can recreate from gen*-dft/)
- `hopper/`, `good_castep/` (temporary)

Use `clean_individual_models=true` to auto-delete individual models.

## Advanced Topics

### Multi-Composition Searches

When searching across compositions (e.g., Li₁₋ₓMₓ), elemental energies are crucial:

```toml
[state.elemental_energies]
Li = -194.8254               # MUST be pseudoatomic energies
M = -1673.6962               # From pseudopotential library
```

**Why pseudoatomic energies?**

These define the zero-point for formation energies, allowing the network to learn interaction energies. Without them, elements have wildly different reference energies (tens to thousands of eV), causing optimization difficulties.

**FOCUS Parameter in buildcell:**

Control which compositions to sample:

```
#FOCUS=1                     # Only pure elements
#FOCUS=2                     # Binary compounds only
#FOCUS=3                     # Ternary compounds only
#FOCUS=2,3                   # Binary and ternary
#FOCUS={1,2,2,2,2,3,3,3}     # Random 1,2,3 in 1:4:3 ratio
```

For Li-Ti-O: use `#FOCUS=2,3` to sample LiₓTiᵧOᵤ, LiₓTiᵧ, TiᵧOᵤ, LiₓOᵤ.

**Formula-space sampling in `Builder` (explicit `#FORMULA` injection):**

Use this when you want to sample from a pre-generated pool of reduced formulas, optionally filtered by oxidation states and charge neutrality.

```toml
[state]
# Import known structures into gen0 before random generation
known_structure_dir = "known_structures"

# Enable formula-space sampling
formula_sampling_enabled = true

# Maximum coefficient allowed in reduced formulas (e.g. Li2TaO3 -> max coeff is 3)
formula_max_reduced_formula_coeff = 6

# Optional explicit list (if empty, formulas are enumerated automatically)
# formula_list = ["LiTaO3", "Li2TaO3", "LiTaO2Cl"]

# Keep only charge-neutral formulas under allowed oxidation states
formula_require_charge_neutral = true

[state.formula_oxidation_states]
Li = [1]
Ta = [5]
O = [-2]
Cl = [-1]
```

Behavior notes:
- Formula sampling requires exactly one resolved seed file (no multi-seed / glob-expanded multi-file setup).
- `#FORMULA=...` is injected per build attempt.
- Existing `#SPECIES`, `#NATOM`, and `#FORMULA` lines are removed from the runtime seed text to avoid conflicts.
- `#NFORM` and other buildcell controls are kept.
- Enumerated formulas are constrained by `#NFORM` and numeric `#NATOM` ranges.

### Custom Calculators

**Testing with Mock Calculator:**

```toml
[state]
dft_mode = "mock"
```

Returns random energies instantly. Perfect for testing workflow without DFT.

**Beyond CASTEP:**

The calculator abstraction supports other DFT codes. Implement an external program compatible with `acrud`:

```julia
acrud(workdir, seedfile_calc, indir, outdir, "my_calculator")
```

Your program should:
1. Read `.res` files from input directory
2. Perform DFT calculations
3. Write results as `.res` files with energies to output directory
4. Return exit code 0 on success

See `acrud` documentation for interface specification.

### Walk-Forward Testing

Built-in validation technique:

```julia
using EDDPotentials

builder = Builder("link.toml")
results = walk_forward_tests(builder)

for (iter, rmse, mae) in results
    println("Gen $iter: RMSE=$(rmse) eV/atom, MAE=$(mae) eV/atom")
end
```

**What it measures:**

For each iteration N:
1. Load ensemble from Generation N-1
2. Predict energies of structures from Generation N
3. Compare to DFT energies
4. Report RMSE and MAE

**Good results**: RMSE comparable to training RMSE (within 2×)
**Bad results**: RMSE >> training RMSE → model not generalizing, needs more diverse data

## Summary

Iterative EDDP training with `link!()` automates potential creation through guided exploration:

1. **Generation 0**: Create initial dataset from random structures
2. **Generation N**: Use model to guide structure search, train on accumulated data
3. **Convergence**: Stop when model performance plateaus
4. **Result**: Ensemble model covering explored configuration space

**Key Concepts:**
- Focuses DFT calculations on uncertain structures
- Accumulative training prevents catastrophic forgetting
- Shaking provides gradient information without force training
- NNLS ensemble weighting combines model predictions

**Getting Started:**
1. Prepare seed files (C-SEARCH.cell, C.cell, C.param)
2. Create link.toml configuration
3. Test with mock calculator
4. Run with DFT calculator
5. Monitor convergence metrics

**Best Practices:**
- Use walk-forward testing to assess generalization
- Archive final ensemble and configuration for reproducibility
- Monitor RMSE and Spearman correlation across generations

## Further Reading

- [What is EDDP?](what_is_eddp.md) - Design philosophy and when to use EDDP
- [Understanding Features](features.md) - Feature generation and theory
- [AIRSS Workflow](../tutorials/airss_workflow.md) - Complete hands-on tutorial
- [Getting Started](../getting_started.md) - Build your first potential
