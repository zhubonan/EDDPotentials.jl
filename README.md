![](docs/src/assets/logo_small.png)
# EDDP.jl

[Documentation](https://zhubonan.github.io/EDDP.jl)

A [Julia](https://julialang.org/) package that implements the [Ephemeral data derived potentials](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.106.014102) (EDDP).
EDDP can be seen as a kind of Machine Learning [(Interatomic) Potentials](https://en.wikipedia.org/wiki/Interatomic_potential) (MLP). 
Normally such potentials are aim at *accurately reproduce the results of first-principles calculations* within a certain region of the configuration space.
Once trained, such potentials can be used to run large scale molecular dynamics simulation which are otherwise intractable with first-principles calculations. 

EDDP takes a simple and physically motivated form that resembles a generalized N-body Lenard-Jones-like potential.
While not its originally design goal, EDDP can still give sufficiently accurate forces to allow molecular dynamics simulation (which does not blow-up) and phonon properties for certain systems. 
In comparison, other state-of-the art MLPs often requires  description of the local atomic environment and complex gaussian process/deep learning/(graph) neutron network architectures.
The hope is being physically sounds allows EDDP to give sufficiently good representations for most of the configuration space,
and hence enables one to carry out crystal structure prediction with much reduced computational and time resources.

## Features

- Generating EDDP feature vectors (local descriptors).
- Train EDDP ensemble models.
- Perform geometry optimization using trained models. 
- Interface to other package for property calculations.
- Workflow script for automated potential building and crystal structure prediction.
- Plot and data analysis for the training dataset and the potentials.

## Related packages

- [airss](https://www.mtg.msm.cam.ac.uk/Codes/AIRSS) - *ab initio* random structure (AIRSS) is used for building random structure through the `buildcell` program included in the bundle.
- [CASTEP](http://www.castep.org) - A plane-wave DFT code used for efficient generation of  training datasets. 
- [eddp](https://www.mtg.msm.cam.ac.uk/Codes/EDDP) - The Fortran EDDP code. EDDP.jl provides limited interoperability with the eddp fortran package. While it is not possible to use the model trained by one with the other, the training datasets are compatible as both use the AIRSS-style SHELX format.
- [disp](https://zhubonan.github.io/disp) - Distributed structure prediction (DISP) is used to schedule and run data generation workloads on (multiple) remote computing clusters. 
