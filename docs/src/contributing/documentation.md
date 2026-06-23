# Documentation Maintenance

Documentation should move with user-visible behavior. When a change adds an
option, workflow mode, output file, metadata field, or exported API, update the
docs in the same branch whenever possible.

## What To Update

Use the smallest page set that makes the change discoverable:

- tutorial pages show where a feature fits in a complete workflow
- how-to pages explain task-specific choices and common pitfalls
- concept pages explain why a behavior exists
- `functions.md` frames API reference entries and links to task guides
- `status.md` records workflow maturity and known limits

Avoid copying the same long explanation into multiple pages. Prefer one primary
reference page and short cross-links elsewhere.

## Checklist

Before finishing a user-visible change, check whether it needs:

- a guide update for new workflow behavior
- API prose or docstring updates for new options
- status-page updates for new stable, advanced, or experimental workflows
- navigation updates in `docs/make.jl` for new pages
- index links for important user-facing pages
- examples that avoid local absolute paths and machine-specific assumptions

## Validation

Run the documentation build after page or navigation changes. The build should
fail if `docs/make.jl` references a missing page, so add new pages before adding
them to navigation.

For advanced workflow docs, also review the rendered page manually enough to
confirm that cross-links go to the intended page and that beginner navigation is
not crowded by specialist material.
