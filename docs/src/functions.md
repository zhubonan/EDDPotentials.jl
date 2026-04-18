# Functions

Documentation for the functions included in the package.

## Feature Generation

Function and types for generating features 

```@autodocs
Modules = [EDDPotentials]
Pages = [
    "core/feature/feature.jl",
    "core/feature/cell.jl",
    "core/gradient/gradient.jl",
    "core/embedding.jl",
    "core/embedding_rules.jl",
    "core/gradient/repulsive_core.jl",
]
```

## Training

Routines for training models feature vectors 

```@autodocs
Modules = [EDDPotentials]
Pages = [
    "preprocessing.jl",
    "training.jl",
    "lmsolve.jl",
    "lmsolve_batched.jl",
    "core/nntools.jl",
]
```

## Model interfaces

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

## Using potentials

Routines for using the trained potentials for energy/forces/stress calculations.

```@autodocs
Modules = [EDDPotentials]
Pages = [
    "core/calculator.jl",
    "core/calculator/metacalculator.jl",
    "opt.jl",
    "optimizer/abstract.jl",
    "optimizer/tpsd.jl",
    "optimizer/fire.jl",
]
```

## Miscellaneous tools

Support utilities and tools.

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

## Automated potentials building

Function for providing automatic iterative potential building via random structure structure searching.

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

## Non-negative least square routines

Internal routines in the [NNLS.jl](https://github.com/rdeits/NNLS.jl) package embedded in the package.

```@autodocs
Modules = [EDDPotentials.EDDPotentialsCore.NNLS]
```
