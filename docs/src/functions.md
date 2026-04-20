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
