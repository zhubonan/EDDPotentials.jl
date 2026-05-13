# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EDDPotentials.jl implements Ephemeral Data Derived Potentials (EDDP) — physically-motivated machine learning interatomic potentials for computational materials science. It trains simple N-body models to reproduce DFT calculations, enabling geometry optimization, crystal structure prediction, and molecular dynamics.

## Commands

```bash
# Run full test suite
julia --project=. test/runtests.jl

# Run a single test file
julia --project=. -e 'include("test/test_calc.jl")'

# Run native optimizer tests (opt-in)
EDDP_RUN_NATIVE_OPT_TESTS=1 julia --project=. test/runtests.jl

# Run all tests that are relevant to code changes (recommended after any change)
julia --project=. -e '
  using Test
  test_files = [  # ordered as in runtests.jl
    "test_layer_boundaries.jl", "test_source_topology.jl",
    "test_core_module_namespace.jl", "test_records.jl",
    "test_cellfeature.jl", "test_random_embedded_feature.jl",
    "test_multi_embedding.jl", "test_embedding.jl",
    "test_gradients.jl", "test_backprop.jl", "test_nninterface.jl",
    "test_json_serialization.jl", "test_calc.jl",
    "test_preprocess.jl", "test_dotcastep.jl", "test_train.jl",
    "test_evaluation.jl", "test_threading.jl",
    "test_sgd_training.jl", "test_builder.jl",
    "test_modifiers.jl", "test_metacalculator.jl",
    "test_md.jl", "test_hot.jl",
  ]
  for f in test_files; include(joinpath("test", f)); end
'

# Run native optimizers individually
EDDP_RUN_NATIVE_OPT_TESTS=1 julia --project=. -e 'include("test/test_native_optimizers.jl")'

# Ensure any code change passes at minimum:
#   test_calc.jl, test_modifiers.jl, test_metacalculator.jl, test_builder.jl
# plus any test file directly related to the modified source.

# Instantiate dependencies
julia --project=. -e "using Pkg; Pkg.instantiate()"

# Format code (excludes scripts/ per .JuliaFormatter.toml)
julia -e "using Pkg; Pkg.add(\"JuliaFormatter\"); using JuliaFormatter; format(\".\")"

# Build system image + CLI
./build.sh

# Build documentation
julia --project=docs/ docs/make.jl
```

## Architecture

### Two-Layer Module Design

The source tree has an internal core/functional split. The public API is available through `using EDDPotentials` regardless of layer.

**Core layer** (`src/core/EDDPotentialsCore.jl`): Standalone submodule accessed as `EDDPotentials.EDDPotentialsCore`. Contains:
- Feature generation (`core/feature/`) — two-body and three-body atomic environment descriptors
- Gradient computation (`core/gradient/`) — chain rule for forces/stress from feature gradients
- NN interface (`core/nn/`) — abstract `AbstractNNInterface` with Linear, Flux, Ensemble, and ManualFluxBackProp implementations
- Calculator kernel (`core/calculator.jl`) — `EDDPCalculator` (model-level with optional cell binding), `create_calculator` factory
- Calculator modifiers (`core/calculator/modifiers/`) — pipeline-based modifiers for MetaCalculator
- MetaCalculator (`core/calculator/metacalculator.jl`) — composes source calculator + modifier pipeline; delegates cell ownership to source

**Functional layer** (`src/EDDPotentials.jl`): Includes core, then adds:
- `training.jl` — model training, ensemble construction via NNLS weighting
- `preprocessing.jl` — data loading and preprocessing
- `functional/` — structure generation (buildcell), relaxation (`Relax`/`RelaxOption`), random structure search (RSS)
- `link/` — automated active-learning workflow (`Builder`, `link!`, `step!`, `run_rss`)
- `hyper.jl` — hyperdimensional relaxation with scheduled penalty via `MetaCalculator` + `ExtraDimensionPenaltyModifier`
- `optimizer/` — native Julia optimizers (`NativeTPSD`, `NativeFIRE`) with `AbstractOptimizer` interface
- `opt.jl` — high-level entry points `opt_tpsd()` and `opt_fire()`
- `md.jl` — molecular dynamics simulation (NVE Velocity Verlet, NVT Langevin/Berendsen) with `MDRunner`, `MDState`

**Experimental** (`src/experimental/`): Archived prototypes, intentionally not in the runtime include graph.

### Calculator Interface

All calculators inherit from `AbstractCalc` and implement:
- `get_cell(calc)` — return the `Cell` object
- `eddp_energy(calc)` — total energy
- `eddp_forces(calc)` — atomic forces (matrix)
- `eddp_stress(calc)` — stress tensor (3×3)
- `calculate!(calc; forces=true, rebuild_nl=true)` — trigger computation (caching handled internally)

Key calculator types:
- **`EDDPCalculator`** — the primary public calculator. Supports both unbound (model-level) and bound (cell-attached) modes via `set_cell!`. Auto-bind constructor: `EDDPCalculator(cell, cf, nn; ...)`.
- **`MetaCalculator`** — source calculator + ordered modifier pipeline with pre/post-calculate hooks; delegates cell ownership to source. Use with `ExtraDimensionPenaltyModifier` for hyperspace relaxation. Variable-cell optimization is controlled via `relax_cell` parameter on optimizer entry points (`opt_tpsd`, `opt_fire`, `relax!`).
- **`create_calculator(calc, cell; modifiers=[])`** — factory that returns a bound calculator or `MetaCalculator` with modifier pipeline.

### Hyperspace Relaxation

Hyperdimensional relaxation uses `MetaCalculator` with `ExtraDimensionPenaltyModifier`:

```julia
# Build the hyperspace calculator
source = create_calculator(eddp_calc, hcell)
penalty = ExtraDimensionPenaltyModifier(1e-4; start_dim=4)
metacalc = MetaCalculator(source, [penalty])

# Relax with scheduled penalty
results, dnorm = scheduled_penalty_relax!(metacalc;
    schedule_every=50,
    max_iterations=500,
    alpha_start=1e-4,
    relax_fn=opt_tpsd,
)
```

For variable-cell hyperspace relaxation, pass `relax_kwargs=(relax_cell=true,)` to `scheduled_penalty_relax!`:

The `hyper_gen_relax` function provides high-level entry points for both workflows:
- `hyper_gen_relax(bu::Builder; ...)` — uses a Builder to access the model/seed configuration; hyperspace controls are passed as keyword arguments to `hyper_gen_relax`, while `link!`/RSS hyperspace settings belong under `[state.relax]` in `link.toml`.
- `hyper_gen_relax(calc::EDDPCalculator, cell::Cell; ...)` — standalone mode with a pre-trained model and a cell, no Builder needed. Uses `create_calculator` internally to assemble the modifier pipeline.

### Feature System

Features are N-body atomic environment descriptors defined in `src/core/feature/`:
- **`TwoBodyFeature(p, species_pair, rcut)`** — pairwise features: `f(r)^p` where `f(r) = 2(1 - r/rcut)`
- **`ThreeBodyFeature(p, q, species_triplet, rcut)`** — triplet features: `f(r_ij)^p × f(r_ik)^p × f(r_jk)^q`
- **`CellFeature`** — bundles multiple features with species resolution
- Use `genp(pmin, pmax, np)` to generate geometric exponent sequences
- `fast_pow(x, y)` provides optimized integer powers for performance-critical feature computation

### NN Interface Hierarchy

```
AbstractNNInterface
├── LinearInterface       — simple linear model
├── FluxInterface         — standard Flux.jl forward pass
├── ManualFluxBackPropInterface — custom backpropagation (default for training)
└── EnsembleNNInterface   — weighted ensemble of models (via NNLS)
```

Models are serialized to/from JLD2 files. `load_from_jld2(path)` auto-detects the interface type.

### Active Learning Workflow (`link/`)

The `Builder` struct drives iterative potential construction:
1. `Builder("link.toml")` — load configuration
2. `link!(builder)` — run automated cycles of RSS → DFT → training
3. `step!(builder)` — single iteration
4. `run_rss(builder)` — structure search with trained model

Submodules: `options.jl` (config types), `workflow.jl` (Builder/link!), `trainer.jl` (training orchestration), `external.jl` (DFT integrations: DISP, PP3, CASTEP), `data.jl`, `analysis.jl`.

### Modifier Pipeline

Modifiers extend calculator behavior through `pre_calculate!` / `post_calculate!` hooks:
- **`ExtraDimensionPenaltyModifier`** — penalize extra-dimensional coordinates (used by hyperspace relaxation)
- **`AdditiveCalculatorModifier`** — combine additional calculator contributions
- Variable-cell optimization is handled by the `relax_cell` keyword argument to `opt_tpsd`, `opt_fire`, and `relax!`.

Access modifiers via `get_modifier(calc, Type)`, `get_parameter`/`set_parameter!`.

See the [Hyperdimensional Relaxation](docs/src/howto/hyperdimensional_relaxation.md) guide for end-to-end examples.

## Fortran repose (Reference Implementation)

The Fortran EDDP implementation lives at `eddp/` (symlink to the eddp-dev tree).

### Paths

- Source: `eddp/repose/src/` (main), `eddp/nn/src/nn.f90` (neural network module)
- Binary: `eddp/repose/bin/repose`
- Benchmark: `eddp/repose/benchmarks/SiO2-1536-ramble/` (1536-atom SiO₂ with trained model)

### Compilation

```bash
cd eddp/repose/src

# Step 1: compile dependencies (spglib first — others need its .mod file)
gfortran -Ofast -march=native -ffast-math -funroll-loops -g \
  -fno-strict-aliasing -fno-inline -cpp -fopenmp -D OPENMP \
  -c spglib_f08.f90 constants.f90 rng.f90

# Step 2: compile nn module
gfortran -Ofast -march=native -ffast-math -funroll-loops -g \
  -fno-strict-aliasing -fno-inline -cpp -fopenmp -D OPENMP \
  -c ../../nn/src/nn.f90

# Step 3: compile repose sources
gfortran -Ofast -march=native -ffast-math -funroll-loops -g \
  -fno-strict-aliasing -fno-inline -cpp -fopenmp -D OPENMP \
  -c cell.f90 ddp.f90 opt.f90 md.f90 repose.f90

# Step 4: link
gfortran -Ofast -march=native -ffast-math -funroll-loops -g \
  -fno-strict-aliasing -fno-inline -fopenmp \
  nn.o opt.o md.o ddp.o cell.o constants.o rng.o spglib_f08.o repose.o \
  -o repose -llapack -lblas -fopenmp

# Step 5: install
cp repose ../bin/repose
```

Build notes:
- `make BUILD=fast` may fail due to missing spglib version; the manual steps above are reliable.
- Without OpenMP: omit `-fopenmp -D OPENMP` from all compilation and link lines.
- Default build (no `BUILD=fast`) uses `-O2`; `BUILD=fast` adds `-Ofast -march=native -ffast-math`.

### Testing

```bash
# Run the SiO₂-1536 benchmark (known-working with current source)
cd eddp/repose/benchmarks/SiO2-1536-ramble
eddp/repose/bin/repose -n -q -ompnp 1 -m 1 SiO2
# Expected: "num ions: 1536", "converged in 1 steps", timing table with -ball/-feat/-netw

# Run the Julia vs Fortran feature benchmark
cd <repo root>
julia --project=. scripts/bench_sio2.jl --repose eddp/repose/bin/repose --runs 3
```

### `.ddp` File Format

The recompiled binary uses the **current** format. Read order in `nn_load` (`eddp/nn/src/nn.f90`):
1. Header: `depth, inputs, outputs, nwgts, nnds` (5 integers)
2. Layers: `depth` integers
3. Weights: `nwgts` reals
4. Activation functions: `depth` strings (e.g. `linear linear linear`)
5. `in_shift`: `inputs` reals
6. `in_scale`: `inputs` reals
7. `out_shift`: `outputs` reals
8. `out_scale`: `outputs` reals
9. Comment: metadata string

Comment line is parsed by `read_ddp` (`eddp/repose/src/ddp.f90`) to determine feature parameters:
```
composition: O-Si rmax:  5.000 length: 8 64 0 0  powers: 2.000 2.517 ...
```
Fields: `ntwo` (2-body features per pair), `nthree` (3-body features per triplet), `pow`/`pow2` (geometric power sequences).

Feature count must be consistent: `nfeat = ncomp + ntwo*ncomp² + nthree*ncomp²*(ncomp+1)/2`.

Network dimension: `nnds = Σ(layers[i]+1)`, `nwgts = Σ((layers[i]+1)*layers[i+1])`.

### Timing Output

`repose -n -q -ompnp 1 -m 1 <seedname>` prints to stderr:
```
total  :        3.8096 sec 100.00 %
ddp    :        3.4986 sec  91.84 %
 -ball :        1.3261 sec  34.81 %    ← neighbour list construction
 -feat :        1.3593 sec  35.68 %    ← features + force/stress gradients
 -netw :        0.8131 sec  21.34 %    ← NN forward pass
```
Note: `-feat` includes feature values AND force/stress gradient computation (they are interleaved in the same loops). `-ball` is the neighbor list build (separate from feature computation).

## Key Dependencies

- **CellBase.jl** (`../CellBase.jl/`) — crystal structure representation, neighbor lists, file I/O
- **Flux.jl** — neural network framework
- **Optim.jl** — optimization (LBFGS, BFGS)
- **JLD2.jl** — model serialization
- **NonNegLeastSquares.jl** — ensemble weighting
- **TimerOutputs.jl** — profiling (`enable_eddp_timer!()` / `disable_eddp_timer!()`)

Extensions (weak deps): CUDA, DataFrames, Plots, PlotlyJS.

## Code Style

- Formatter config in `.JuliaFormatter.toml`: ignores `scripts/`, `whitespace_in_kwargs = false`
- Core module uses `EDDPotentialsCore` namespace internally; functional layer imports selected symbols unqualified
- The `const to = EDDPotentialsCore.to` pattern gives top-level access to the shared `TimerOutput`
- Test helpers defined in `test/utils.jl` (e.g., `_get_lco_calc()`) are shared across test files
- CellBase functions like `distance_between`, `masses`, `volume` may return Unitful quantities (e.g. `Quantity{Float64, Å}`). In tests, always use `ustrip()` to extract the bare value before comparing with plain numbers.
