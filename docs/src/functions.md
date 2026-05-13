# Functions

Reference documentation for all public functions and types in EDDPotentials.jl.

## Feature Generation

Types and functions for defining and computing atomic environment descriptors. See [Understanding Features](concepts/features.md) for a conceptual introduction.

The primary user-facing type is [`CellFeature`](@ref), a parametric immutable struct that centralizes all two-body/three-body feature parameters. The keyword constructor `CellFeature(elements; p2=..., ...)` is the main public API. The [`FeatureProto`](@ref) type is a lightweight internal container used by [`RandomEmbeddedCellFeature`](@ref).

```@autodocs
Modules = [EDDPotentials]
Pages = [
    "core/feature/feature.jl",
    "core/feature/cell.jl",
]
```

## Training

Functions for training models from feature vectors, evaluating accuracy, and building ensembles.

```@autodocs
Modules = [EDDPotentials]
Pages = [
    "data/preprocessing.jl",
    "training/training.jl",
    "training/lmsolve.jl",
    "training/lmsolve_batched.jl",
]
```

## Model Interfaces

Neural network interfaces for energy prediction. [`ManualFluxBackPropInterface`](@ref) is the default for training; [`EnsembleNNInterface`](@ref) is used for prediction with uncertainty.

```@autodocs
Modules = [EDDPotentials]
Pages = [
    "core/nn/interface.jl",
    "core/nn/linear.jl",
    "core/nn/flux.jl",
    "core/nn/manual_backprop.jl",
    "core/nn/ensemble.jl",
]
```

## Calculators

[`EDDPCalculator`](@ref) is the primary calculator type for energy, force, and stress evaluation. The [`MetaCalculator`](@ref) composes a source calculator with an ordered modifier pipeline. Variable-cell optimization uses the `relax_cell` parameter on optimizer entry points ([`opt_tpsd`](@ref), [`opt_fire`](@ref), [`relax!`](@ref)). The [`create_calculator`](@ref) factory assembles calculator pipelines from a pre-trained [`EDDPCalculator`](@ref).

```@autodocs
Modules = [EDDPotentials]
Pages = [
    "core/calculator.jl",
    "core/calculator/metacalculator.jl",
    "core/calculator/eddpcalculator.jl",
]
```

## Calculator Modifiers

Modifiers extend calculator behavior through pre/post-calculate hooks in the [`MetaCalculator`](@ref) pipeline.

```@autodocs
Modules = [EDDPotentials]
Pages = [
    "core/calculator/modifiers/abstract.jl",
    "core/calculator/modifiers/dimension_penalty.jl",
    "core/calculator/modifiers/additive_calc.jl",
]
```

## Optimization

Structure optimization algorithms and high-level entry points. Use [`opt_fire`](@ref) for general-purpose geometry optimization or [`opt_tpsd`](@ref) for the TPSD algorithm.

```@autodocs
Modules = [EDDPotentials]
Pages = [
    "simulation/opt.jl",
    "simulation/optimizer/abstract.jl",
    "simulation/optimizer/tpsd.jl",
    "simulation/optimizer/fire.jl",
]
```

## Molecular Dynamics

Run molecular dynamics simulations with trained potentials. Supports NVE (Velocity Verlet) and NVT (Langevin, Berendsen) ensembles.

```@autodocs
Modules = [EDDPotentials]
Pages = [
    "simulation/md.jl",
]
```

## Relaxation

High-level relaxation interface wrapping optimizers with cell handling, convergence checking, and trajectory recording.

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
Modules = [EDDPotentials]
Pages = [
    "simulation/relax.jl",
]
```

## Hyperdimensional Relaxation

Functions for structure relaxation in extended (hyperdimensional) configuration spaces, used for enhanced exploration in structure prediction.

```@autodocs
Modules = [EDDPotentials]
Pages = [
    "workflow/hyper.jl",
]
```

## Automated Potential Building

Functions for the iterative active-learning workflow: [`Builder`](@ref) configuration, [`link!`](@ref) for automated cycles, [`run_rss`](@ref) for structure search. See [Iterative Training](concepts/iterative_training.md) for details.

```@autodocs
Modules = [EDDPotentials]
Pages = [
    "link/link.jl",
    "link/options.jl",
    "link/workflow.jl",
    "link/analysis.jl",
    "link/trainer.jl",
]
```

## Miscellaneous Tools

Support utilities including serialization, plotting recipes, and file I/O helpers.

```@autodocs
Modules = [EDDPotentials]
Pages = [
    "tools.jl",
    "analysis/records.jl",
    "analysis/plotting/recipes.jl",
    "analysis/lazy_res.jl",
]
```
