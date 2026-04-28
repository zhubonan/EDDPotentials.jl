# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EDDPotentials.jl implements Ephemeral Data Derived Potentials (EDDP) — physically-motivated machine learning interatomic potentials for computational materials science. It trains simple N-body models to reproduce DFT calculations, enabling geometry optimization, crystal structure prediction, and molecular dynamics.

## Commands

```bash
# Run full test suite
julia --project=test/ test/runtests.jl

# Run a single test file
julia --project=test/ -e 'include("test/test_calc.jl")'

# Run native optimizer tests (opt-in)
EDDP_RUN_NATIVE_OPT_TESTS=1 julia --project=test/ test/runtests.jl

# Run all tests that are relevant to code changes (recommended after any change)
julia --project=test/ -e '
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
  for f in test_files; include(f); end
'

# Run native optimizers individually
EDDP_RUN_NATIVE_OPT_TESTS=1 julia --project=test/ -e 'include("test/test_native_optimizers.jl")'

# Ensure any code change passes at minimum:
#   test_calc.jl, test_modifiers.jl, test_metacalculator.jl, test_builder.jl
# plus any test file directly related to the modified source.

# Instantiate dependencies
julia --project=. -e "using Pkg; Pkg.instantiate()"
julia --project=test/ -e "using Pkg; Pkg.instantiate()"

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
- Calculator kernel (`core/calculator.jl`) — `NNCalc`, deprecated `VariableCellCalc` shim (→ `MetaCalculator` + `CellRelaxationModifier`), `EDDPCalculator` + `create_calculator` factory
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
- **`NNCalc`** — primary calculator binding Cell + CellFeature + NNInterface with neighbor list caching
- **`MetaCalculator`** — source calculator + ordered modifier pipeline with pre/post-calculate hooks; delegates cell ownership to source. Use with `CellRelaxationModifier` for variable-cell optimization (replaces the deprecated `VariableCellCalc`).
- **`EDDPCalculator`** — model-level calculator (trained model not bound to any cell). Provides `create_calculator(calc, cell; modifiers=[])` factory that returns `NNCalc` (no modifiers) or `MetaCalculator(NNCalc, modifiers)`.

### Hyperspace Relaxation

Hyperdimensional relaxation uses `MetaCalculator` with `ExtraDimensionPenaltyModifier`:

```julia
# Build the hyperspace calculator
source = NNCalc(hcell, cf, ensemble)
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

For variable-cell hyperspace relaxation, add a `CellRelaxationModifier`:

```julia
modifiers = [ExtraDimensionPenaltyModifier(1e-4), CellRelaxationModifier(hcell)]
metacalc = MetaCalculator(source, modifiers)
```

The `hyper_gen_relax` function provides high-level entry points for both workflows:
- `hyper_gen_relax(bu::Builder; ...)` — uses a Builder configuration; hyperspace parameters in `link.toml` under `[state]`: `hyper_ndim`, `hyper_alpha_start`, `hyper_schedule_every`, `hyper_target_iterations`.
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
- **`CellRelaxationModifier`** — variable-cell optimization within MetaCalculator
- **`AdditiveCalculatorModifier`** — combine additional calculator contributions

Access modifiers via `get_modifier(calc, Type)`, `get_parameter`/`set_parameter!`.

See the [Hyperdimensional Relaxation](docs/src/howto/hyperdimensional_relaxation.md) guide for end-to-end examples.

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
