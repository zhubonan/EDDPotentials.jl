# Source Dependency Baseline

This baseline records the current logical source layout. It is intentionally
compact; the live guardrails are `test/test_source_topology.jl`,
`test/test_layer_boundaries.jl`, and `test/test_namespace_modules.jl`.

## Entrypoint

- `src/EDDPotentials.jl`

## Logical Groups

- `src/potential/`: potential runtime internals: features, gradients, NN interfaces,
  calculators, modifiers, model artifacts, pair potentials, finite-difference
  diagnostics, and small calculator utilities.
- `src/training/`: structure loading, feature stores, LM, batched LM, SGD,
  high-level training, ensembles, and training result helpers.
- `src/simulation/`: native optimizers, relaxation, and molecular dynamics.
- `src/search/`: buildcell integration, standalone RSS, Hot-AIRSS, minimum
  separation utilities, hyperspace relaxation, and model-structure generation.
- `src/workflow/`: active-learning Builder workflow, external DFT runners,
  workflow data/training adapters, and formula sampling.
- `src/analysis/`: evaluation metrics, phase diagrams/records, plotting hooks,
  and lazy SHELX result inspection.

## Dependency Direction

The intended dependency direction is:

```text
potential
  -> training
  -> simulation
  -> search
  -> workflow
  -> analysis
```

The top-level module wires the runtime modules together and re-exports the
public API. Lower-level groups should not depend on higher-level workflow or
analysis modules.

## Notes

- `src/experimental/` remains intentionally outside the runtime include graph.
- The public source modules are `Potential`, `Training`, `Simulation`,
  `Search`, `Workflow`, and `Analysis`.
- The old `Model` and `Tools` namespace wrappers have been removed; their
  remaining functionality lives in `Potential` or `Training`.
