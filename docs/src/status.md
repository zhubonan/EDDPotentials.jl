# Package Status

EDDPotentials.jl is an active research code for training and using Ephemeral
Data Derived Potentials in crystal-structure prediction workflows. The package
is usable for model training, calculator evaluation, relaxation, random
structure search, and iterative AIRSS-style active learning, but some advanced
workflows remain experimental.

## Supported Workflows

- Train energy-only or energy-plus-force EDDP models from `.res` structure data.
- Use trained models through the calculator interface for energies, forces, and
  stress.
- Relax structures with fixed-cell or variable-cell native optimizers.
- Run AIRSS/buildcell random structure searches with EDDP relaxation.
- Drive iterative training workflows through `Builder`, `step!`, `link!`, and
  `run_rss`.
- Use analytic Lennard-Jones baselines for calculator and optimizer checks.

## Advanced Supported Workflows

- Constrained RSS for surfaces, slabs, and seeded cells, including selected
  AIRSS/CASTEP ionic and cell constraints.
- Symmetry-aware relaxation for structures marked with `symmetry_generate`, with
  explicit controls to force or disable symmetry projection.
- Distributed link workflows where RSS, DFT, and training workers coordinate
  through shared filesystem state.
- RSS trajectory and timing metadata for diagnosing expensive or uneven searches.

## Experimental Workflows

- Hyperdimensional relaxation is intended for exploration and method
  development. Validate resulting structures with standard 3D relaxation and
  reference calculations.
- Batched LM training is available for larger datasets, but the API and tuning
  guidance may continue to evolve.
- SGD/Adam training and GPU acceleration are useful for experimentation, but
  the native CPU training path remains the reference workflow.

## Known Limits

- EDDP models are designed for rapid screening and ranking, not as a replacement
  for final high-accuracy DFT validation.
- Model quality depends strongly on the diversity and cleanliness of the
  training structures.
- Workflow integrations assume the external programs they call, such as AIRSS
  and DFT engines, are installed and configured separately.
- APIs in experimental modules can change between releases.

## Recommended Validation

For production structure prediction, use EDDP to generate and rank candidates,
then re-evaluate promising structures with reference DFT. Monitor validation
errors across generations, inspect high-uncertainty candidates, and keep final
DFT calculations as the source of truth for phase stability decisions.
