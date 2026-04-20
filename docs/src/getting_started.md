# Getting Started

This page gives a compact starting point for new `EDDPotentials.jl` users.

## Recommended first steps

1. Read the [home page](index.md) for the package overview and the standard workflow.
2. Read [Understanding Features](concepts/features.md) for the core `CellFeature` model.
3. Follow [AIRSS Workflow](tutorials/airss_workflow.md) for an end-to-end iterative build example.

## First calculator example

```julia
using EDDPotentials
using CellBase

cell = Cell(Lattice(12.0, 12.0, 12.0), [:Ar, :Ar], [0.0 1.4; 0.0 0.0; 0.0 0.0])
calc = EDDPotentials.lj_baseline_calc(cell; sigma=1.0, epsilon=1.0, rcut=3.0)

energy = eddp_energy(calc)
forces = eddp_forces(calc)
```

For a trainable EDDP model rather than an analytic baseline, continue with
[Understanding Features](concepts/features.md) and [AIRSS Workflow](tutorials/airss_workflow.md).
