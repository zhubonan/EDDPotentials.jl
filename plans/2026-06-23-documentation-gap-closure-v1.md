# Documentation Gap Closure

## Objective

Reduce or eliminate the gap between implemented EDDPotentials.jl functionality and user-facing documentation by auditing recent feature changes, updating the most relevant guide and API pages, and adding durable documentation checks that catch missing pages and undocumented public workflow options before they reach users. The plan targets the current known gaps around buildcell surface constraints, symmetry-aware relaxation, training-data pruning, RSS timing metadata, and workflow status/navigation consistency.

## Current Evidence

The implementation now preserves AIRSS and CASTEP constraint intent from buildcell output and seed lattice tags, including position tags, ionic constraints, cell constraints, fixed-cell flags, fixed-volume flags, and seed lattice tags such as FIX, CFIX, and ABFIX in `src/search/buildcell.jl:44`, `src/search/buildcell.jl:75`, `src/search/buildcell.jl:103`, `src/search/buildcell.jl:117`, and `src/search/buildcell.jl:157`. Relaxation consumes these constraints through `RelaxOption.honor_cell_constraints`, symmetry settings, and resolved step constraints in `src/simulation/relax.jl:26`, `src/simulation/relax.jl:130`, and `src/simulation/relax.jl:304`.

The workflow layer now documents some option-level behavior in source docstrings, including RSS-level pressure precedence in `src/workflow/link/options.jl:39`, training-data pruning options in `src/workflow/link/options.jl:141`, and RSS settings in `src/workflow/link/options.jl:240`. The actual training-data pruning behavior is implemented in `src/workflow/link/data.jl:114` and `src/workflow/link/data.jl:148`. RSS trajectory and timing metadata are handled in `src/search/rss.jl:43`, `src/search/rss.jl:416`, and `src/search/rss.jl:558`.

The current documentation has good pressure coverage in `docs/src/howto/structure_prediction.md:271` and `docs/src/functions.md:118`, but the advanced buildcell section only mentions basic symmetry buildcell controls at `docs/src/howto/structure_prediction.md:314`. The AIRSS tutorial shows a minimal seed and CASTEP template in `docs/src/tutorials/airss_workflow.md:51` and `docs/src/tutorials/airss_workflow.md:78`, and RSS pruning is explained in `docs/src/tutorials/airss_workflow.md:318`. The package status page and navigation now exist at `docs/src/status.md:1`, `docs/src/index.md:10`, and `docs/make.jl:16`, but status content should be kept synchronized with supported and experimental workflows.

## Implementation Plan

- [ ] 1. Audit recent implementation and working-tree changes against the documentation tree, recording user-visible options, metadata, output files, and workflows.
- [ ] 2. Classify each audit item as stable, advanced supported, experimental, diagnostic, or internal so later prose does not overstate guarantees.
- [ ] 3. Expand the structure prediction how-to with buildcell surface and constrained-cell behavior, including preserved AIRSS and CASTEP constructs.
- [ ] 4. Explain how ionic constraints, cell constraints, fixed-volume flags, and fixed-cell flags affect variable-cell relaxation in constrained RSS workflows.
- [ ] 5. Document how seed lattice tags FIX, CFIX, and ABFIX map into relaxation constraints, with enough rationale for surface-search users.
- [ ] 6. Add AIRSS workflow tutorial guidance showing where constrained seeds and symmetry generation belong, then link to the deeper reference.
- [ ] 7. Strengthen relaxation API prose for symmetry modes, automatic symmetry activation, trajectory options, cell constraints, and projected convergence stress.
- [ ] 8. Document training-data pruning in iterative training docs, covering absolute energy, composition-relative energy, and minimum pair-distance pruning.
- [ ] 9. Add practical guidance for pruning failure modes so users understand what to change when all structures are removed before featurization.
- [ ] 10. Document RSS trajectory and timing metadata as diagnostic output, including where files are written and which metadata fields appear in structures.
- [ ] 11. Update the package status page to match current maturity levels for training, relaxation, RSS, link, constraints, symmetry, distributed workflows, and experiments.
- [ ] 12. Add a documentation-maintenance checklist for future user-visible options, metadata fields, output files, and workflow modes.
- [ ] 13. Add a lightweight parity check that verifies navigation pages exist and, if practical, scans high-value option structs for missing documentation.
- [ ] 14. Run the documentation build and review rendered pages for broken references, stale terminology, duplicated explanations, and navigation clarity.

## Verification Criteria

- The documentation build command exits successfully with no missing-page failures.
- Every currently user-visible feature identified in the audit has either guide-level documentation, API reference documentation, or an explicit status-page note explaining that it is internal or experimental.
- The structure prediction how-to includes surface and constrained buildcell behavior, symmetry-aware relaxation behavior, pressure precedence, RSS pruning, RSS trajectory output, and RSS timing metadata.
- The iterative training concept guide and AIRSS workflow tutorial include practical guidance for training-data pruning options.
- The package status page accurately distinguishes stable, advanced supported, and experimental workflows.
- A maintainer can inspect one checklist or CI output and see whether new docs pages are missing from navigation and whether documented option coverage has regressed.
- The final documentation update does not require changes to implementation behavior to make the prose true.

## Potential Risks and Mitigations

1. **Risk: Documentation accidentally promises behavior that is only implemented for internal tests or narrow workflows.**
   Mitigation: Ground every new explanation in source references and existing tests, and label experimental or diagnostic behavior explicitly when the implementation does not provide a stable public contract.

2. **Risk: The docs become repetitive because the same options appear in API reference, concepts, tutorials, and how-to guides.**
   Mitigation: Give each page a distinct role: tutorials show where options fit in a workflow, how-to pages explain task-oriented choices, API pages define exact option semantics, and the status page summarizes maturity.

3. **Risk: Automated parity checks create noisy failures for internal helpers or intentionally undocumented fields.**
   Mitigation: Start with a small allowlist-based check for navigation and high-value option structs, then expand coverage only after maintainers agree on what is public.

4. **Risk: Existing working-tree changes continue to move while documentation is being updated.**
   Mitigation: Perform the audit immediately before editing docs, keep documentation changes isolated from source changes, and rerun the audit against the final diff before verification.

5. **Risk: External workflow examples depend on AIRSS, CASTEP, or local GaSb analysis assets that are not portable.**
   Mitigation: Keep general documentation examples independent of local paths and place project-specific surface-analysis material in examples or case-study pages only if it is intended for users.

## Alternative Approaches

1. Minimal patch: Document only the currently known missing features and rely on the existing Documenter build for validation. This is fastest, but it does not prevent the implementation and documentation from drifting again.

2. Full documentation audit: Review every exported option struct, public function, and guide page before editing. This gives the strongest coverage, but it is slower and may delay urgent documentation fixes.

3. CI-first approach: Add navigation and option-coverage checks before writing new prose. This prevents regressions early, but it may require allowlist design before the current user-facing gaps are closed.

4. Case-study approach: Turn the GaSb surface work into a user-facing constrained-surface tutorial. This would make the new features concrete, but it risks mixing project-specific research material with general package documentation unless carefully scoped.
