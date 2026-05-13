# Source Dependency Baseline

This baseline records the current logical source layout. It is intentionally
compact; the live guardrails are `test/test_source_topology.jl`,
`test/test_layer_boundaries.jl`, and `test/test_namespace_modules.jl`.

## Entrypoint

- `src/EDDPotentials.jl`

## Logical Groups

- `src/core/`: potential runtime internals: features, gradients, NN interfaces,
  calculators, modifiers, model artifacts, and pair potentials.
- `src/data/`: structure loading, `StructureContainer`, `FeatureStore`,
  batching, splitting, and streaming feature-store IO.
- `src/training/`: LM, batched LM, SGD, high-level training, ensembles, and
  training result helpers.
- `src/simulation/`: native optimizers, relaxation, hyperspace support used by
  relaxation, and molecular dynamics.
- `src/search/`: buildcell integration, standalone RSS, Hot-AIRSS, minimum
  separation utilities, and model-structure generation.
- `src/link/`: active-learning Builder workflow, external DFT runners, workflow
  config, workflow data/training adapters, and formula sampling.
- `src/workflow/`: workflow-specific high-level helpers that depend on Builder
  state, currently including Builder-based hyperspace entry points.
- `src/analysis/`: evaluation metrics, phase diagrams/records, plotting hooks,
  and lazy SHELX result inspection.
- `src/tools.jl`: shared diagnostics and small utilities that have not yet been
  moved into a narrower logical group.
- `src/namespaces.jl`: compatibility-preserving namespace facades:
  `Potential`, `Model`, `Data`, `Training`, `Simulation`, `Search`, `Workflow`,
  `Analysis`, and `Tools`.

## Dependency Direction

The intended dependency direction is:

```text
core
  -> data
  -> training
  -> simulation
  -> search
  -> analysis
  -> link/workflow
```

The top-level module is a compatibility facade and may re-export symbols from
all groups. No lower-level group should depend on `link/` or `workflow/`.

## Notes

- `src/experimental/` remains intentionally outside the runtime include graph.
- `EDDPotentialsCore` is retained for compatibility.
- New code should prefer the logical namespace facades instead of adding more
  unqualified top-level coupling.
