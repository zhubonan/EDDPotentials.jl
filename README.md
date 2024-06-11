![](docs/src/assets/logo_small.png)
# EDDPotentials.jl

[Documentation](https://zhubonan.github.io/EDDPotentials.jl)

A [Julia](https://julialang.org/) package that implements the [Ephemeral data derived potentials](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.106.014102) (EDDP).
EDDP can be seen as a kind of Machine Learning [(Interatomic) Potentials](https://en.wikipedia.org/wiki/Interatomic_potential) (MLP). 
Normally such potentials are aim at accurately reproduce the results of first-principles calculations to run large scale molecular dynamics simulation which are otherwise intractable with first-principles calculations. 

EDDP takes a simple and physically motivated form that resembles a generalized N-body *Lennard-Jones-like* potential, making it very easy to train.
Being physically motivated allow EDDP to give sufficiently good representations for most of the configuration space, allowing crystal structure prediction to be carried out with much reduced computational resources and wall-time.
In many cases, EDDP can still give sufficiently accurate forces to allow molecular dynamics simulation and phonon properties.

A recent pre-print explaining more applications of the model can found here: https://arxiv.org/abs/2306.06475. 

## Features

- Generating EDDPotentials feature vectors (local descriptors).
- Train EDDPotentials ensemble models.
- Perform geometry optimization using trained models. 
- Interface to other package for property calculations.
- Automated workflows for automated potential building and crystal structure prediction.
  - Training data generation using standard scheduler queue system.
  - Training data generation through [DISP](https://github.com/zhubonan/disp).
- Analysis and visualisation for potential quality verification and convergence.

## Related packages

- [airss](https://www.mtg.msm.cam.ac.uk/Codes/AIRSS) - *ab initio* random structure (AIRSS) is used for building random structure through the `buildcell` program included in the bundle.
- [eddp](https://www.mtg.msm.cam.ac.uk/Codes/EDDP) - The Fortran EDDPotentials code. EDDPotentials.jl provides limited interoperability with the eddp fortran package. Although directly loading models trained by eddp is not implemented, the training datasets are compatible as both use the AIRSS-style SHELX format.
- [CASTEP](http://www.castep.org) - A plane-wave DFT code used for efficient generation of training datasets, although in principle any atomistic modelling package that calculates total energy of a given structure is supported.
- [disp](https://zhubonan.github.io/disp) - Distributed structure prediction (DISP) package can be used to schedule and run data generation tasks (e.g. DFT calculations) on multiple remote computing clusters. 
