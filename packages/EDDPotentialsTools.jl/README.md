## EDDPotentialsTools.jl

Miscellaneous tools for working with [EDDPotentials.jl](https://github.com/zhubonan/EDDPotentials.jl)

## Python requirements

`PyCall` should be up and running with `numpy`,  `ase`, and `phonopy` installed in the active working environment.

## On-demand package loading

To avoid unnecessary code loading, interfaces in `EDDPotentialsTools` are only with specific packages are explicitly loaded:

- load `PyCall` to enable `ase` and `phonopy` interface
- load `Molly` to enable `EDDPotentialsInter` *General Interactions* type.

This is achieve via [Requires.jl](https://github.com/JuliaPackaging/Requires.jl)