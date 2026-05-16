# Source Dependency Graph

Generated from the current runtime include graph. Experimental sources are intentionally excluded.

## Entry
- `EDDPotentials.jl`

## Runtime Modules
- `EDDPotentials.jl`: `EDDPotentials`, `Potential`, `Model`, `Tools`, `Training`, `Search`, `Workflow`, `Analysis`
- `core/EDDPotentialsCore.jl`: `EDDPotentialsCore`
- `core/nnls/nnls.jl`: `NNLS`
- `data/preprocessing.jl`: `Data`
- `simulation/Simulation.jl`: `Simulation`

## Include Edges
- `EDDPotentials.jl` -> `core/EDDPotentialsCore.jl`
- `EDDPotentials.jl` -> `data/preprocessing.jl`
- `EDDPotentials.jl` -> `simulation/Simulation.jl`
- `EDDPotentials.jl` -> `tools.jl`
- `EDDPotentials.jl` -> `training/context.jl`
- `EDDPotentials.jl` -> `training/lmsolve.jl`
- `EDDPotentials.jl` -> `training/lmsolve_batched.jl`
- `EDDPotentials.jl` -> `training/sgd_training.jl`
- `EDDPotentials.jl` -> `training/training.jl`
- `EDDPotentials.jl` -> `search/buildcell.jl`
- `EDDPotentials.jl` -> `search/transforms.jl`
- `EDDPotentials.jl` -> `search/hot.jl`
- `EDDPotentials.jl` -> `workflow/hyper.jl`
- `EDDPotentials.jl` -> `search/rss.jl`
- `EDDPotentials.jl` -> `search/minsep.jl`
- `EDDPotentials.jl` -> `search/model_structures.jl`
- `EDDPotentials.jl` -> `link/link.jl`
- `EDDPotentials.jl` -> `workflow/hyper_builder.jl`
- `EDDPotentials.jl` -> `analysis/evaluation.jl`
- `EDDPotentials.jl` -> `analysis/records.jl`
- `EDDPotentials.jl` -> `analysis/plotting/recipes.jl`
- `EDDPotentials.jl` -> `analysis/lazy_res.jl`
- `EDDPotentials.jl` -> `analysis/plotting.jl`
- `core/EDDPotentialsCore.jl` -> `core/constants.jl`
- `core/EDDPotentialsCore.jl` -> `core/nnls/nnls.jl`
- `core/EDDPotentialsCore.jl` -> `core/utils/utils.jl`
- `core/EDDPotentialsCore.jl` -> `core/feature/feature.jl`
- `core/EDDPotentialsCore.jl` -> `core/embedding.jl`
- `core/EDDPotentialsCore.jl` -> `core/embedding_rules.jl`
- `core/EDDPotentialsCore.jl` -> `core/gradient/gradient.jl`
- `core/EDDPotentialsCore.jl` -> `core/nn/interface.jl`
- `core/EDDPotentialsCore.jl` -> `core/nntools.jl`
- `core/EDDPotentialsCore.jl` -> `core/calculator.jl`
- `core/EDDPotentialsCore.jl` -> `core/calculator/eddpcalculator.jl`
- `core/EDDPotentialsCore.jl` -> `core/calculator/modifiers/modifiers.jl`
- `core/EDDPotentialsCore.jl` -> `core/calculator/metacalculator.jl`
- `core/EDDPotentialsCore.jl` -> `core/pairpot.jl`
- `core/calculator/modifiers/modifiers.jl` -> `core/calculator/modifiers/abstract.jl`
- `core/calculator/modifiers/modifiers.jl` -> `core/calculator/modifiers/dimension_penalty.jl`
- `core/calculator/modifiers/modifiers.jl` -> `core/calculator/modifiers/volume_constraint.jl`
- `core/calculator/modifiers/modifiers.jl` -> `core/calculator/modifiers/additive_calc.jl`
- `core/feature/feature.jl` -> `core/feature/spline_table.jl`
- `core/feature/feature.jl` -> `core/feature/cell.jl`
- `core/feature/feature.jl` -> `core/feature/embedded.jl`
- `core/feature/feature.jl` -> `core/feature/multi_embedded.jl`
- `core/gradient/gradient.jl` -> `core/gradient/repulsive_core.jl`
- `core/gradient/gradient.jl` -> `core/gradient/workspace.jl`
- `core/nn/interface.jl` -> `core/nn/manual_backprop.jl`
- `core/nn/interface.jl` -> `core/nn/linear.jl`
- `core/nn/interface.jl` -> `core/nn/ensemble.jl`
- `core/nn/interface.jl` -> `core/nn/flux.jl`
- `core/nn/interface.jl` -> `core/nn/json_serialization.jl`
- `link/link.jl` -> `link/options.jl`
- `link/link.jl` -> `link/utils.jl`
- `link/link.jl` -> `link/formula_sampling.jl`
- `link/link.jl` -> `link/workflow.jl`
- `link/link.jl` -> `link/trainer.jl`
- `link/link.jl` -> `link/external.jl`
- `link/link.jl` -> `link/data.jl`
- `link/link.jl` -> `link/analysis.jl`
- `link/link.jl` -> `link/client_server.jl`
- `simulation/Simulation.jl` -> `simulation/optimizer/abstract.jl`
- `simulation/Simulation.jl` -> `simulation/optimizer/utils.jl`
- `simulation/Simulation.jl` -> `simulation/optimizer/constraints.jl`
- `simulation/Simulation.jl` -> `simulation/optimizer/tpsd.jl`
- `simulation/Simulation.jl` -> `simulation/optimizer/fire.jl`
- `simulation/Simulation.jl` -> `simulation/opt.jl`
- `simulation/Simulation.jl` -> `simulation/relax.jl`
- `simulation/Simulation.jl` -> `simulation/md.jl`

## Reachability
- Runtime nodes reachable from `EDDPotentials.jl`: 70
- Orphans outside `experimental/`: none
