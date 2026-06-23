# Documentation Page Inventory

## Objective

Define the exact documentation pages to add or update so the documentation matches recent EDDPotentials.jl implementation work. This plan narrows the broader documentation-gap plan into a page-level editing map, with each page assigned a purpose, scope, and integration point.

## Current Evidence

The current documentation tree includes guides, tutorials, concepts, API reference, and package status pages under `docs/src`, with navigation configured in `docs/make.jl:16`. The package status page is now linked from `docs/src/index.md:10` and listed in navigation at `docs/make.jl:19`. The structure prediction guide already covers RSS pruning and pressure behavior at `docs/src/howto/structure_prediction.md:150`, `docs/src/howto/structure_prediction.md:254`, and `docs/src/howto/structure_prediction.md:314`, but it does not yet fully cover constrained buildcell surface workflows.

The AIRSS tutorial introduces seed files and calculation templates at `docs/src/tutorials/airss_workflow.md:51` and `docs/src/tutorials/airss_workflow.md:78`, and it mentions RSS pruning at `docs/src/tutorials/airss_workflow.md:318`. The iterative training concept guide explains link configuration and RSS parameters at `docs/src/concepts/iterative_training.md:170` and `docs/src/concepts/iterative_training.md:249`. The API reference includes relaxation and workflow autodocs at `docs/src/functions.md:114`, `docs/src/functions.md:179`, and `docs/src/functions.md:195`.

The implementation areas needing page coverage include buildcell constraint parsing in `src/search/buildcell.jl:44`, `src/search/buildcell.jl:75`, `src/search/buildcell.jl:103`, `src/search/buildcell.jl:117`, and `src/search/buildcell.jl:157`; relaxation option behavior in `src/simulation/relax.jl:26`, `src/simulation/relax.jl:130`, and `src/simulation/relax.jl:304`; training-data pruning in `src/workflow/link/data.jl:114` and `src/workflow/link/data.jl:148`; and RSS trajectory or timing output in `src/search/rss.jl:43`, `src/search/rss.jl:416`, and `src/search/rss.jl:558`.

## Implementation Plan

- [ ] 1. Update `docs/src/howto/structure_prediction.md` as the primary reference for constrained RSS, surface seeds, symmetry-aware relaxation, trajectory output, and timing metadata.
- [ ] 2. Add a new how-to page at `docs/src/howto/constrained_structure_search.md` if the constrained RSS material becomes too large for the structure prediction guide.
- [ ] 3. Update `docs/make.jl` only if the new constrained-search page is added, placing it beside the structure prediction how-to in the How-To Guides section.
- [ ] 4. Update `docs/src/index.md` to expose the constrained-search material through Quick Links or Documentation Structure when a separate page is added.
- [ ] 5. Update `docs/src/tutorials/airss_workflow.md` with a short tutorial-level section showing where constrained seeds and symmetry settings fit in AIRSS workflows.
- [ ] 6. Update `docs/src/concepts/iterative_training.md` to explain training-data pruning as part of link dataset hygiene and generation quality control.
- [ ] 7. Update `docs/src/functions.md` around the relaxation and workflow sections to add prose that frames the relevant autodocs before the generated API listings.
- [ ] 8. Update `docs/src/status.md` to classify constrained RSS, symmetry-aware relaxation, distributed link, and diagnostic metadata by maturity level.
- [ ] 9. Add a new maintenance page at `docs/src/contributing/documentation.md` if contributor guidance should live in generated docs rather than repository-only instructions.
- [ ] 10. Update `docs/make.jl` to include the new contributor documentation page only if generated docs are chosen as the home for documentation-maintenance rules.
- [ ] 11. Keep `docs/src/getting_started.md` unchanged unless new cross-links are needed, because the missing material is advanced workflow documentation.
- [ ] 12. Keep feature-theory pages such as `docs/src/concepts/features.md` unchanged unless the audit finds terminology that conflicts with pruning or constraint prose.
- [ ] 13. Review rendered navigation after page updates so new advanced pages do not crowd the beginner path from index to getting started to AIRSS tutorial.
- [ ] 14. Run the documentation build after editing and verify that every added page is reachable from navigation or an intentional cross-link.

## Verification Criteria

- The final page inventory clearly distinguishes pages to update from pages to add.
- Every new page has a stated reason to exist and a navigation or cross-link destination.
- `docs/src/howto/structure_prediction.md` remains the main user-facing RSS reference unless the separate constrained-search page is added.
- `docs/src/tutorials/airss_workflow.md` remains tutorial-level and links out for exhaustive constrained-search details.
- `docs/src/functions.md` contains only framing prose and API reference integration, not long workflow explanations.
- `docs/src/status.md` reflects the maturity of the newly documented workflows.
- The documentation build succeeds after any page and navigation changes.

## Potential Risks and Mitigations

1. **Risk: Adding too many pages makes the docs harder to navigate.**
   Mitigation: Prefer updating existing pages unless constrained-search or contributor-maintenance content becomes large enough to deserve its own page.

2. **Risk: A new constrained-search page duplicates the structure prediction guide.**
   Mitigation: Keep the structure prediction guide as the overview and use the new page only for detailed constraint semantics and surface-search recipes.

3. **Risk: Tutorial pages become overloaded with reference details.**
   Mitigation: Add short tutorial context and cross-link to how-to or API pages for exhaustive option behavior.

4. **Risk: Contributor documentation is misplaced.**
   Mitigation: Choose between generated docs and repository instructions based on whether external users or maintainers are the primary audience.

5. **Risk: Navigation changes break the docs build.**
   Mitigation: Add navigation entries only after the target page exists and verify with the Documenter build.

## Alternative Approaches

1. Existing-pages-only approach: Update structure prediction, AIRSS workflow, iterative training, functions, and status pages without adding any new pages. This keeps navigation simple but may make the structure prediction guide long.

2. New constrained-search page approach: Add one dedicated how-to for constrained and surface RSS. This gives users a focused reference but adds another page to maintain.

3. New contributor-maintenance page approach: Add generated contributor docs for documentation parity rules. This improves visibility but may be unnecessary if repository instructions are preferred.

4. Case-study approach: Add a GaSb surface-search case study page. This would show the constrained workflow concretely, but it should wait until the general constrained-search docs are stable.
