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

Calculator types for energy, force, and stress evaluation. [`NNCalc`](@ref) is the primary calculator; [`VariableCellCalc`](@ref) adds lattice degrees of freedom for cell optimization.

```@autodocs
Modules = [EDDPotentials]
Pages = [
    "core/calculator.jl",
    "core/calculator/metacalculator.jl",
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

## Relaxation

High-level relaxation interface wrapping optimizers with cell handling, convergence checking, and trajectory recording.

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
