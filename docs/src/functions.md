# Functions

Reference documentation for all public functions and types in EDDPotentials.jl.

## Feature Generation

Types and functions for defining and computing atomic environment descriptors. See [Understanding Features](concepts/features.md) for a conceptual introduction.

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
    "preprocessing.jl",
    "training.jl",
    "lmsolve.jl",
    "lmsolve_batched.jl",
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

Calculator types for energy, force, and stress evaluation. [`NNCalc`](@ref) is the primary calculator; [`MetaCalculator`](@ref) with [`CellRelaxationModifier`](@ref) adds lattice degrees of freedom for cell optimization (replaces the deprecated `VariableCellCalc`); [`EDDPCalculator`](@ref) bundles a trained model and provides the [`create_calculator`](@ref) factory for assembling calculator pipelines.

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
    "core/calculator/modifiers/cell_relaxation.jl",
    "core/calculator/modifiers/additive_calc.jl",
]
```

## Optimization

Structure optimization algorithms and high-level entry points. Use [`opt_fire`](@ref) for general-purpose geometry optimization or [`opt_tpsd`](@ref) for the TPSD algorithm.

```@autodocs
Modules = [EDDPotentials]
Pages = [
    "opt.jl",
    "optimizer/abstract.jl",
    "optimizer/tpsd.jl",
    "optimizer/fire.jl",
]
```

## Molecular Dynamics

Run molecular dynamics simulations with trained potentials. Supports NVE (Velocity Verlet) and NVT (Langevin, Berendsen) ensembles.

```@autodocs
Modules = [EDDPotentials]
Pages = [
    "md.jl",
]
```

## Relaxation

High-level relaxation interface wrapping optimizers with cell handling, convergence checking, and trajectory recording.

### Lennard-Jones baseline helper

EDDP now includes a fixed-weight force-shifted Lennard-Jones baseline built on top of
`CellFeature`, `LinearInterface`, and `NNCalc`.

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
    "functional/relax.jl",
]
```

## Hyperdimensional Relaxation

Functions for structure relaxation in extended (hyperdimensional) configuration spaces, used for enhanced exploration in structure prediction.

```@autodocs
Modules = [EDDPotentials]
Pages = [
    "hyper.jl",
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
    "records.jl",
    "plotting/recipes.jl",
    "eddpf90.jl",
    "lazy_res.jl",
]
```
