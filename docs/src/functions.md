# Functions

Reference documentation for the exported and documented functions and types in EDDPotentials.jl.

The package still exports a small number of compatibility constants and implementation hooks, especially in the distributed workflow. Those names are listed by Julia's namespace introspection but are not recommended as stable user API unless they are documented here or in a task guide.

## Feature Generation

Types and functions for defining and computing atomic environment descriptors. See [Understanding Features](concepts/features.md) for a conceptual introduction.

The primary user-facing type is `CellFeature`, a parametric immutable struct that centralizes all two-body/three-body feature parameters. The keyword constructor `CellFeature(elements; p2=..., ...)` is the main public API. `FeatureProto` is a lightweight internal container used by `RandomEmbeddedCellFeature`.

```@autodocs
Modules = [EDDPotentials.Potential]
Pages = [
    "potential/feature/feature.jl",
    "potential/feature/cell.jl",
    "potential/feature/embedded.jl",
    "potential/feature/multi_embedded.jl",
    "potential/feature/spline_table.jl",
    "potential/embedding.jl",
    "potential/embedding_rules.jl",
]
```

## Training

Functions for training models from feature vectors, evaluating accuracy, and building ensembles.

```@autodocs
Modules = [EDDPotentials.Training]
Pages = [
    "training/context.jl",
    "training/data.jl",
    "training/training.jl",
    "training/lmsolve.jl",
    "training/lmsolve_batched.jl",
    "training/sgd_training.jl",
]
```

## Model Interfaces

Neural network interfaces for energy prediction. `ManualFluxBackPropInterface` is the default for training; `EnsembleNNInterface` is used for prediction with uncertainty.

```@autodocs
Modules = [EDDPotentials.Potential]
Pages = [
    "potential/nn/interface.jl",
    "potential/nn/linear.jl",
    "potential/nn/flux.jl",
    "potential/nn/manual_backprop.jl",
    "potential/nn/ensemble.jl",
    "potential/nn/json_serialization.jl",
    "potential/nntools.jl",
]
```

## Calculators

`EDDPCalculator` is the primary calculator type for energy, force, and stress evaluation. `MetaCalculator` composes a source calculator with an ordered modifier pipeline. Variable-cell optimization uses the `relax_cell` parameter on optimizer entry points (`opt_tpsd`, `opt_fire`, `relax!`). The `create_calculator` factory assembles calculator pipelines from a pre-trained `EDDPCalculator`.

```@autodocs
Modules = [EDDPotentials.Potential]
Pages = [
    "potential/calculator.jl",
    "potential/calculator/metacalculator.jl",
    "potential/calculator/eddpcalculator.jl",
    "potential/pairpot.jl",
]
```

## Calculator Modifiers

Modifiers extend calculator behavior through pre/post-calculate hooks in the `MetaCalculator` pipeline.

```@autodocs
Modules = [EDDPotentials.Potential]
Pages = [
    "potential/calculator/modifiers/abstract.jl",
    "potential/calculator/modifiers/dimension_penalty.jl",
    "potential/calculator/modifiers/additive_calc.jl",
    "potential/calculator/modifiers/volume_constraint.jl",
]
```

## Optimization

Structure optimization algorithms and high-level entry points. Use `opt_fire` for general-purpose geometry optimization or `opt_tpsd` for the TPSD algorithm.

```@autodocs
Modules = [EDDPotentials.Simulation]
Pages = [
    "simulation/opt.jl",
    "simulation/optimizer/abstract.jl",
    "simulation/optimizer/constraints.jl",
    "simulation/optimizer/tpsd.jl",
    "simulation/optimizer/fire.jl",
    "simulation/optimizer/utils.jl",
]
```

## Molecular Dynamics

Run molecular dynamics simulations with trained potentials. Supports NVE (Velocity Verlet) and NVT (Langevin, Berendsen) ensembles.

```@autodocs
Modules = [EDDPotentials.Simulation]
Pages = [
    "simulation/md.jl",
]
```

## Relaxation

High-level relaxation interface wrapping optimizers with cell handling, convergence checking, and trajectory recording.

For variable-cell relaxation under hydrostatic pressure, prefer
`RelaxOption.external_pressure_gpa`, which is specified in GPa. Positive values
are compressive and negative values are tensile. The legacy
`RelaxOption.external_pressure` field is deprecated because it uses internal
eV/Å³ stress units; setting both pressure interfaces to non-default values is an
error.

`RelaxOption.symmetry` accepts `"auto"`, `"on"`, and `"off"`. Automatic mode
uses symmetry projection only for cells marked by `symmetry_generate`; `"on"`
attempts projection for every cell; `"off"` disables it. `honor_cell_constraints`
controls whether buildcell-derived cell constraints are appended to optimizer
step constraints. Projected forces and stresses are used for convergence checks,
while calculator forces and stresses remain the raw model values.

Trajectory output is controlled by `keep_trajectory`, `trajectory_file`,
`trajectory_format`, and `trajectory_every`. RSS assigns candidate-specific
trajectory stems when nested relaxation requests trajectory files.

### Lennard-Jones baseline helper

EDDP now includes a fixed-weight force-shifted Lennard-Jones baseline built on top of
`CellFeature`, `LinearInterface`, and the calculator stack.

Use `EDDPotentials.lj_baseline_calc(...)` when you want an analytic pair potential in
the standard calculator stack without training a model.

```julia
using EDDPotentials
using CellBase

cell = Cell(Lattice(12.0, 12.0, 12.0), [:Ar, :Ar], [0.0 1.4; 0.0 0.0; 0.0 0.0])
calc = EDDPotentials.lj_baseline_calc(cell; sigma=1.0, epsilon=1.0, rcut=3.0)

energy = eddp_energy(calc)
forces = eddp_forces(calc)
```

The pairwise energy inside the cutoff is

```math
V(r) = 4\epsilon \left[\left(\frac{\sigma}{r}\right)^{p_{\mathrm{rep}}} - \left(\frac{\sigma}{r}\right)^{p_{\mathrm{att}}}\right]
```

and the helper applies a force shift at `rcut` so both the energy and radial force go
smoothly to zero there.

For multi-species systems, pass symmetric per-pair `sigma` and `epsilon` dictionaries:

```julia
sigma = Dict((:H, :H) => 1.0, (:H, :O) => 1.25, (:O, :O) => 1.4)
epsilon = Dict((:H, :H) => 0.4, (:H, :O) => 0.9, (:O, :O) => 1.1)
calc = EDDPotentials.lj_baseline_calc(cell; sigma, epsilon, rcut=4.0)
```

```@autodocs
Modules = [EDDPotentials.Simulation]
Pages = [
    "simulation/relax.jl",
]
```

## Experimental Hyperdimensional Relaxation

Experimental functions for structure relaxation in extended (hyperdimensional) configuration spaces. These APIs are intended for enhanced exploration and method development; validate any output with standard 3D relaxation and reference calculations.

```@autodocs
Modules = [EDDPotentials.Search]
Pages = [
    "search/hyper.jl",
]
```

## Structure Search

Functions for generating random structures, running RSS and Hot-AIRSS workflows, computing minimum-separation statistics, and constructing small model structures for feature inspection. Buildcell readers preserve selected AIRSS/CASTEP ionic and cell constraints for constrained RSS workflows; see [Constrained Structure Search](howto/constrained_structure_search.md).

```@autodocs
Modules = [EDDPotentials.Search]
Pages = [
    "search/buildcell.jl",
    "search/rss.jl",
    "search/hot.jl",
    "search/minsep.jl",
    "search/model_structures.jl",
    "search/transforms.jl",
]
```

## Automated Potential Building

Functions for the iterative active-learning workflow: `Builder` configuration, `link!` automated cycles, `run_rss` structure search, formula-space sampling, training-data pruning, and file-backed distributed execution. See [Iterative Training](concepts/iterative_training.md) for details.

```@autodocs
Modules = [EDDPotentials.Workflow]
Pages = [
    "workflow/link/link.jl",
    "workflow/link/options.jl",
    "workflow/link/formula_sampling.jl",
    "workflow/link/workflow.jl",
    "workflow/link/analysis.jl",
    "workflow/link/trainer.jl",
    "workflow/link/data.jl",
    "workflow/link/utils.jl",
    "workflow/link/distributed.jl",
    "workflow/link/external.jl",
    "workflow/link/hyper_builder.jl",
]
```

## Analysis

Evaluation, records, phase diagrams, plotting entry points, and lazy SHELX parsing.

```@autodocs
Modules = [EDDPotentials.Analysis]
Pages = [
    "analysis/evaluation.jl",
    "analysis/records.jl",
    "analysis/plotting_recipes.jl",
    "analysis/lazy_res.jl",
    "analysis/plotting.jl",
]
```

## Miscellaneous Utilities

Support utilities including file I/O helpers, finite-difference checks, timer controls, and low-level feature-gradient helpers.

```@docs
EDDPotentials.enable_eddp_timer!
EDDPotentials.disable_eddp_timer!
EDDPotentials.reset_eddp_timer!
EDDPotentials.Potential.NNLS
```

```@autodocs
Modules = [EDDPotentials.Potential]
Pages = [
    "potential/utilities.jl",
    "potential/utils.jl",
    "potential/gradient/gradient.jl",
    "potential/gradient/repulsive_core.jl",
]
```
