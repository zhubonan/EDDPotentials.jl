#=
Iterative potential building through active learning.

This module provides the Builder infrastructure for automated construction
of machine learning potentials through:
- Random structure search (RSS) for data generation
- External DFT calculator integration
- Iterative model training and validation
- Walk-forward testing

# Module Organization

The link module is organized into focused submodules:

- **options.jl**: Configuration types and options
- **trainer.jl**: Training orchestration and model creation
- **utils.jl**: Shared utilities (constants, directory management, UUID)
- **external.jl**: External DFT calculator integrations (DISP, PP3, CRUD, etc.)
- **workflow.jl**: Builder struct and workflow orchestration (link!, step!)
- **data.jl**: Data loading and management
- **analysis.jl**: Analysis and reporting tools

# Main Entry Points

- `Builder`: Main object for iterative potential construction
- `link!`: Run automated iterative building cycles
- `step!`: Execute a single iteration
- `run_rss`: Run random structure search with trained model

# Example Usage

```julia
# Load builder from configuration file
builder = Builder("link.toml")

# Run automated iterative building
link!(builder)

# Or run step by step
step!(builder)

# Perform structure search with trained model
run_rss(builder)
```
=#

import Base
using Parameters
using JSON
using YAML
using TOML
using ArgParse
using Random

# Include submodules in dependency order
include("options.jl")   # Configuration types
include("utils.jl")     # Shared utilities
include("formula_sampling.jl") # Formula-space sampling utilities
include("workflow.jl")  # Builder struct (needed by trainer.jl)
include("trainer.jl")   # Training functions (uses Builder)
include("external.jl")  # External calculators (uses Builder, utils)
include("data.jl")      # Data loading (uses Builder)
include("analysis.jl")  # Analysis tools (uses Builder, data functions)
include("distributed.jl") # File-backed distributed link workflow
