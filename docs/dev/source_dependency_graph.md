# Source Dependency Graph

Generated from the current runtime include graph. Experimental sources are intentionally excluded.

## Entry
- `EDDPotentials.jl`

## Runtime Modules
- `EDDPotentials.jl`: `EDDPotentials`
- `analysis/Analysis.jl`: `Analysis`
- `potential/Potential.jl`: `Potential`
- `potential/nnls/nnls.jl`: `NNLS`
- `search/Search.jl`: `Search`
- `simulation/Simulation.jl`: `Simulation`
- `training/Training.jl`: `Training`
- `workflow/Workflow.jl`: `Workflow`

## Include Edges
- `EDDPotentials.jl` -> `potential/Potential.jl`
- `EDDPotentials.jl` -> `training/Training.jl`
- `EDDPotentials.jl` -> `simulation/Simulation.jl`
- `EDDPotentials.jl` -> `search/Search.jl`
- `EDDPotentials.jl` -> `workflow/Workflow.jl`
- `EDDPotentials.jl` -> `analysis/Analysis.jl`
- `analysis/Analysis.jl` -> `analysis/evaluation.jl`
- `analysis/Analysis.jl` -> `analysis/records.jl`
- `analysis/Analysis.jl` -> `analysis/plotting_recipes.jl`
- `analysis/Analysis.jl` -> `analysis/lazy_res.jl`
- `analysis/Analysis.jl` -> `analysis/plotting.jl`
- `workflow/Workflow.jl` -> `workflow/link/link.jl`
- `workflow/Workflow.jl` -> `workflow/link/hyper_builder.jl`
- `workflow/link/link.jl` -> `workflow/link/options.jl`
- `workflow/link/link.jl` -> `workflow/link/utils.jl`
- `workflow/link/link.jl` -> `workflow/link/formula_sampling.jl`
- `workflow/link/link.jl` -> `workflow/link/workflow.jl`
- `workflow/link/link.jl` -> `workflow/link/trainer.jl`
- `workflow/link/link.jl` -> `workflow/link/external.jl`
- `workflow/link/link.jl` -> `workflow/link/data.jl`
- `workflow/link/link.jl` -> `workflow/link/analysis.jl`
- `workflow/link/link.jl` -> `workflow/link/client_server.jl`
- `search/Search.jl` -> `search/buildcell.jl`
- `search/Search.jl` -> `search/transforms.jl`
- `search/Search.jl` -> `search/hot.jl`
- `search/Search.jl` -> `search/hyper.jl`
- `search/Search.jl` -> `search/rss.jl`
- `search/Search.jl` -> `search/minsep.jl`
- `search/Search.jl` -> `search/model_structures.jl`
- `simulation/Simulation.jl` -> `simulation/optimizer/abstract.jl`
- `simulation/Simulation.jl` -> `simulation/optimizer/utils.jl`
- `simulation/Simulation.jl` -> `simulation/optimizer/constraints.jl`
- `simulation/Simulation.jl` -> `simulation/optimizer/tpsd.jl`
- `simulation/Simulation.jl` -> `simulation/optimizer/fire.jl`
- `simulation/Simulation.jl` -> `simulation/opt.jl`
- `simulation/Simulation.jl` -> `simulation/relax.jl`
- `simulation/Simulation.jl` -> `simulation/md.jl`
- `training/Training.jl` -> `training/data.jl`
- `training/Training.jl` -> `training/context.jl`
- `training/Training.jl` -> `training/lmsolve.jl`
- `training/Training.jl` -> `training/lmsolve_batched.jl`
- `training/Training.jl` -> `training/sgd_training.jl`
- `training/Training.jl` -> `training/training.jl`
- `potential/Potential.jl` -> `potential/constants.jl`
- `potential/Potential.jl` -> `potential/nnls/nnls.jl`
- `potential/Potential.jl` -> `potential/utils.jl`
- `potential/Potential.jl` -> `potential/feature/feature.jl`
- `potential/Potential.jl` -> `potential/embedding.jl`
- `potential/Potential.jl` -> `potential/embedding_rules.jl`
- `potential/Potential.jl` -> `potential/gradient/gradient.jl`
- `potential/Potential.jl` -> `potential/nn/interface.jl`
- `potential/Potential.jl` -> `potential/nntools.jl`
- `potential/Potential.jl` -> `potential/calculator.jl`
- `potential/Potential.jl` -> `potential/calculator/eddpcalculator.jl`
- `potential/Potential.jl` -> `potential/calculator/modifiers/modifiers.jl`
- `potential/Potential.jl` -> `potential/calculator/metacalculator.jl`
- `potential/Potential.jl` -> `potential/pairpot.jl`
- `potential/Potential.jl` -> `potential/utilities.jl`
- `potential/calculator/modifiers/modifiers.jl` -> `potential/calculator/modifiers/abstract.jl`
- `potential/calculator/modifiers/modifiers.jl` -> `potential/calculator/modifiers/dimension_penalty.jl`
- `potential/calculator/modifiers/modifiers.jl` -> `potential/calculator/modifiers/volume_constraint.jl`
- `potential/calculator/modifiers/modifiers.jl` -> `potential/calculator/modifiers/additive_calc.jl`
- `potential/nn/interface.jl` -> `potential/nn/manual_backprop.jl`
- `potential/nn/interface.jl` -> `potential/nn/linear.jl`
- `potential/nn/interface.jl` -> `potential/nn/ensemble.jl`
- `potential/nn/interface.jl` -> `potential/nn/flux.jl`
- `potential/nn/interface.jl` -> `potential/nn/json_serialization.jl`
- `potential/gradient/gradient.jl` -> `potential/gradient/repulsive_core.jl`
- `potential/gradient/gradient.jl` -> `potential/gradient/workspace.jl`
- `potential/feature/feature.jl` -> `potential/feature/spline_table.jl`
- `potential/feature/feature.jl` -> `potential/feature/cell.jl`
- `potential/feature/feature.jl` -> `potential/feature/embedded.jl`
- `potential/feature/feature.jl` -> `potential/feature/multi_embedded.jl`

## Reachability
- Runtime nodes reachable from `EDDPotentials.jl`: 74
- Orphans outside `experimental/`: none
