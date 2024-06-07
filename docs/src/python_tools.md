# Using Python tools

## Setting up PyCall.jl

[PyCall.jl](https://github.com/JuliaPy/PyCall.jl) is used to call python funcition define the classes that can use the trained models are calculator. 

Guides for setting it up can be found at its [home page](https://github.com/JuliaPy/PyCall.jl).
On Linux, this is as simple as (from scratch): 

```bash
conda create -n pycall_env python ase phonopy <other packages2> <other package2> ...
conda activate pycall
export PYTHON=`which python`
julia -e 'using Pkg; Pkg.add("PyCall"); Pkg.build("PyCall")'
unset PYTHON
```

The key is to *build* the `PyCall.jl` package with a `PYTHON` environmental variable that points to the interpreter of the virtual environment to be used for Julia. 

You can verify if the environment is up and running by importing `ase` in the Julia REPL:

```julia-repl
julia> using PyCall

julia> pyimport("ase")

```


## Phonon calculations

Phonon calculations requires a fully relaxed structures, so one has to relax the structure first:

```julia-repl
julia> builder = Builder("link.toml")  # Load the train model

julia> @show builder.state.iteration  # Check if the latest iteration has been detected

julia> res = read_res("<path to SHLEX file>")  # Read in the structure file

julia> calc = NNCalc(res, builder.cf, load_ensemble(builder))   # Construct a NNCalc object

julia> EDDPotential.optimise!(calc |> VariableCellCalc)   # Optimise with variable cell shape
```

Finally, we run the finite displacement calculations:

```julia-repl
julia> using EDDPotentialTools

julia> EDDPotentialTools.run_phonon(calc;outdir="phonon")
```

This writes the YAML files and `FORCE_SETS` and `phonopy_params.yaml` files into a new folder called `phonon`.
One can use [phonopy](https://phonopy.github.io/) for further processing from this point.


!!! note

    The [sumo](https://smtg-ucl.github.io/sumo/) package provides a `sumo-phonon-bandplot` command which can be used to run a band structure calculation using phonopy with automatically generated band pathways. 