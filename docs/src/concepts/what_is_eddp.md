# What Is EDDP?

EDDP stands for **Ephemeral Data Derived Potentials**. The package provides
physically motivated machine-learning interatomic potentials aimed at crystal
structure prediction and rapid structural screening.

## Core idea

EDDP combines:

- compact two-body and three-body atomic environment descriptors,
- lightweight trainable models such as `LinearInterface` and shallow neural networks,
- a fast calculator stack for energies, forces, and stresses.

The approach is designed to be simpler and faster than large universal models,
while remaining expressive enough for iterative structure-search workflows.

## Where to go next

- [Understanding Features](features.md)
- [Iterative Training](iterative_training.md)
- [AIRSS Workflow](../tutorials/airss_workflow.md)
- [API Reference](../functions.md)
