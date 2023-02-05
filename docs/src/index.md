# EDDP.jl

A Julia package that implements the [Ephemeral data derived potentials](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.106.014102) (EDDP).
EDDP can be seen as a kind of Machine Learning [(Interatomic) Potentials](https://en.wikipedia.org/wiki/Interatomic_potential) (MLP). 
Normally such potentials are aim at *accurately reproduce the results of first-principles calculations* within a certain region of the configuration space.
Once trained, such potentials can be used to run large scale molecular dynamics simulation which are otherwise intractable with first-principles calculations. 

EDDP takes a simple and physically motivated form that resembles a generalized N-body Lenard-Jones-like potential.
While not its originally design goal, EDDP can still give sufficiently accurate forces to allow molecular dynamics simulation (which does not blow-up) and phonon properties for certain systems. 
In comparison, other state-of-the art MLPs often requires  description of the local atomic environment and complex gaussian process/deep learning/(graph) neutron network architectures.
The hope is being physically sounds allows EDDP to give sufficiently good representations for most of the configuration space,
and hence enables one to carry out crystal structure prediction with much reduced computational and time resources.

## Features

- Generating EDDP feature vectors.
- Train EDDP ensemble models.
- Perform geometry optimization using trained models. 
- Interface to other package for property calculations.
- Workflow script for automated potential building and crystal structure prediction.
- Plot and data analysis for the training dataset and the potentials.

## Documentations

```@contents
Pages = ["index.md", "getting_started.md", "faq.md"]
```

```@meta
CurrentModule = EDDP
```

```@docs
```

## Index

```@index
```