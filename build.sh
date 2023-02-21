#!/bin/bash

# Compiler system image for environment @eddp
julia -e 'using Pkg;Pkg.activate(;temp=true);Pkg.add("PackageCompiler");include("scripts/sysimg.jl");build()'

# Build the CLI interface through Comonicon.jl
julia -e 'using Pkg;Pkg.activate("packages/EDDPCli.jl");Pkg.resolve();Pkg.build()'

# Run the CLI - this triggers the compilation of the CLI code
$HOME/.julia/bin/eddp --help