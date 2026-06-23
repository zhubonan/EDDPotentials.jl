# Constrained Structure Search

Use constrained structure search when random structures should preserve part of
the seed-cell intent during EDDP relaxation. This is useful for surfaces,
interfaces, slabs, epitaxial cells, and other workflows where unconstrained
variable-cell relaxation would erase the geometry you meant to sample.

This page describes how EDDPotentials carries AIRSS/CASTEP constraint intent
from `buildcell` output into the native relaxation stack. For the broader
production RSS workflow, see [Crystal Structure Prediction](structure_prediction.md).

## Quick Workflow

For a constrained RSS run:

1. Start from a normal working AIRSS/link setup.
2. Add the minimal ionic or lattice constraints needed to the buildcell search
   seed.
3. Run a small RSS batch before launching a production search.
4. Inspect the relaxed output cells and any trajectory files.
5. Scale out only after the constrained directions and relaxed structures look
   physically sensible.

Keep the CASTEP calculation template focused on DFT settings. EDDP relaxation
sees constraints and `symmetry_generate` from the buildcell seed or generated
buildcell output, not from the DFT template used later for external
calculations.

## Constraint Flow

`buildcell` creates a CASTEP-style cell. EDDPotentials reads that output and
records supported constraint intent on the returned `Cell`:

- position-line tags become per-atom Cartesian freeze masks
- `%BLOCK IONIC_CONSTRAINTS` becomes per-atom Cartesian freeze masks
- `FIX_ALL_CELL`, `FIX_VOL`, and `%BLOCK CELL_CONSTRAINTS` become cell-step
  constraints
- selected lattice tags in the input seed can supplement missing buildcell
  output constraints

During relaxation, `RelaxOption.honor_cell_constraints` is enabled by default.
When it is enabled, `relax!`, RSS, and link workflows append the recorded
constraints to the optimizer step constraints before the native optimizer runs.
The raw calculator energy, forces, and stress remain unchanged; constraints
project optimizer steps and convergence forces/stresses.

## Ionic Constraints

For atom positions, EDDPotentials preserves the supported AIRSS/CASTEP intent
as optimizer constraints:

- position comments containing `%FIX` freeze all Cartesian components of that
  atom
- `%BLOCK IONIC_CONSTRAINTS` freezes any Cartesian component whose constraint
  direction is nonzero

Position comments containing `%NOMOVE` are recorded on the generated cell for
inspection, but the current native optimizer constraints are derived from
`%FIX` and `%BLOCK IONIC_CONSTRAINTS`.

Use these for slabs with frozen substrate layers, adsorbates that should remain
on selected sites during a diagnostic relaxation, or seeded searches where only
part of a motif should move.

## Cell Constraints

For variable-cell relaxation, cell constraints act on the symmetric strain
block in Voigt order `(xx, yy, zz, yz, xz, xy)`.

Supported cell controls:

- `FIX_ALL_CELL : true` freezes all cell degrees of freedom
- `FIX_VOL : true` projects variable-cell steps to trace-free strain, preserving
  volume to first order
- `%BLOCK CELL_CONSTRAINTS` reads the first six values as the Voigt cell mask,
  where zero means the corresponding strain component is fixed

For surfaces, the most common choice is to keep the in-plane cell fixed while
allowing the surface-normal spacing and atoms to relax. Choose the cell mask
that matches the lattice directions used in the seed, and verify the output
cell before launching a large search.

## Seed Lattice Tags

Some AIRSS surface seeds put lattice-fixing tags directly in `%BLOCK
LATTICE_CART` or `%BLOCK LATTICE_ABC`. If the generated buildcell output does
not already contain explicit cell constraints, EDDPotentials records these seed
tags:

- `#FIX` freezes all cell degrees of freedom
- `#CFIX` freezes the `(zz, yz, xz)` strain components
- `#ABFIX` freezes the `(xx, yy, xy)` strain components

Explicit constraints in the buildcell output take precedence over these seed
tags. Treat seed lattice tags as a fallback for preserving intent that buildcell
does not echo into the generated cell.

## Symmetry-Aware Relaxation

When the seed or generated output contains `symmetry_generate`, EDDPotentials
marks the cell for symmetry-aware relaxation. `RelaxOption.symmetry` controls
how this is used:

- `"auto"` applies symmetry projection only to cells marked by
  `symmetry_generate`
- `"on"` attempts symmetry projection for every relaxed cell
- `"off"` disables symmetry projection

`RelaxOption.symmetry_tolerance` controls symmetry detection. For generated
structures, start with the default tolerance and only tighten or loosen it after
checking whether the intended symmetry is actually detected.

## RSS Trajectories and Timing Metadata

RSS can keep per-candidate relaxation trajectories when the nested relaxation
options request trajectory output. If `trajectory_file` is set for RSS, each
candidate receives a label-specific stem under the RSS output directory so CSV
and extxyz trajectory files do not overwrite one another.

Kept RSS structures also receive diagnostic timing metadata:

- `rss_wall_time_seconds`
- `rss_started_at_unix`
- `rss_finished_at_unix`

These fields are for diagnosing expensive searches and uneven candidate costs.
Do not use them as structural descriptors or training targets.

## Practical Guidance

For constrained searches:

- inspect generated buildcell structures before relaxation to confirm that the
  intended atoms, lattice directions, and surface/slab geometry are present
- run a small RSS batch and inspect relaxed structures to confirm that the
  constraints were honored during relaxation
- keep `honor_cell_constraints` enabled unless intentionally testing an
  unconstrained relaxation
- use `symmetry = "off"` if symmetry projection fights a deliberately
  low-symmetry surface reconstruction
- keep pressure settings at the RSS level, because RSS pressure is propagated to
  each candidate relaxation

For high-throughput work, combine constraints with RSS pruning only after
checking that all candidates share the same reduced composition. Pruning is
automatically disabled when mixed-composition pools make per-composition energy
statistics unreliable.
