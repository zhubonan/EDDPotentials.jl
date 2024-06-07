# Functions

Documentation for the functions included in the package.

## Feature Generation

Function and types for generating features 

```@autodocs
Modules = [EDDPotential]
Pages = [
    "feature.jl",
    "gradient.jl",
    "embedding.jl",
    "embedding_rules.jl",
    "repulsive_core.jl",
]
```

## Training

Routines for training models feature vectors 

```@autodocs
Modules = [EDDPotential]
Pages = [
    "preprocessing.jl",
    "training.jl",
    "lmsolve.jl",
    "nntools.jl",
]
```

## Model interfaces

```@autodocs
Modules = [EDDPotential]
Pages = [
    "nn/interface.jl",
    "nn/linear.jl",
    "nn/flux.jl",
    "nn/manual_backprop.jl",
    "nn/ensemble.jl",
]
```

## Using potentials 

Routines for using the trained potentials for energy/forces/stress calculations.

```@autodocs
Modules = [EDDPotential]
Pages = [
    "calculator.jl",
    "opt.jl",
]
```

## Miscellaneous tools

Support utilities and tools.

```@autodocs
Modules = [EDDPotential]
Pages = [
    "tools.jl",
    "records.jl",
    "plotting/recipes.jl",
    "eddpf90.jl",
]
```

## Automated potentials building

Function for providing automatic iterative potential building via random structure structure searching.

```@autodocs
Modules = [EDDPotential]
Pages = [
    "link/link.jl",
    "link/trainer.jl",
]
```

## None-negative least square routines

Internal routines in the [NNLS.jl](https://github.com/rdeits/NNLS.jl) package embedded in the package.

```@autodocs
Modules = [EDDPotential.NNLS]
```