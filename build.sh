#!/bin/bash

# Compiler system image for environment @eddp
#julia scripts/dev_sysimg.jl

# Build the CLI interface through Comonicon.jl
julia -e 'using Pkg;Pkg.activate("packages/EDDPotentialsCLI.jl");Pkg.resolve();Pkg.build()'

# Run the CLI - this triggers the compilation of the CLI code
$HOME/.julia/bin/eddp --help
