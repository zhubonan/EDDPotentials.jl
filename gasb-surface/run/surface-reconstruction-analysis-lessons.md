# GaSb Surface Reconstruction Search: Notes and Lessons

This document summarizes the practical lessons from the GaSb surface reconstruction
search and the mistakes found during the chemical-potential analysis.

## What Was Changed

### RSS Timing Metadata

The RSS relaxation path was updated to write wall-time metadata into generated
`.res` files:

- `rss_wall_time_seconds`
- `rss_started_at_unix`
- `rss_finished_at_unix`

This makes it possible to audit how long each searched structure took, instead of
only relying on relaxation metadata such as iteration count, final force, and
convergence status.

### Adatom Stoichiometry

For the current fixed-slab setup, stoichiometry is defined using only the top
adatom layer. The fixed slab reference is:

```text
Ga24Sb24
```

Therefore, for each total structure:

```text
n_Ga_adatom = N_Ga_total - 24
n_Sb_adatom = N_Sb_total - 24
```

For example:

```text
Total structure: Ga32Sb32
Adatom formula: Ga8Sb8
```

This distinction matters because surface reconstruction ranking should compare
the variable surface/adatom composition, not the fixed lower slab composition.

### MatterSim Chemical Potential Window

MatterSim was used to compute relaxed reference energies for:

- elemental Ga
- elemental Sb
- bulk GaSb

The GaSb stability window is defined by:

```text
Delta mu_Ga + Delta mu_Sb = Delta H_f(GaSb)
Delta mu_Ga <= 0
Delta mu_Sb <= 0
```

with:

```text
Delta H_f(GaSb) = mu_GaSb_bulk - mu_Ga_bulk - mu_Sb_bulk
```

The current MatterSim references give:

```text
Delta H_f(GaSb) = -0.462277889251709 eV/f.u.
```

So:

```text
Ga-rich: Delta mu_Ga = 0, Delta mu_Sb = Delta H_f
Sb-rich: Delta mu_Sb = 0, Delta mu_Ga = Delta H_f
```

### Reconstruction Formation Energy

The final analysis uses a MatterSim-only reconstruction formation energy density:

```text
gamma_recon =
    (E_recon_MS - E_clean_slab_MS - n_Ga_adatom * mu_Ga - n_Sb_adatom * mu_Sb) / A
```

where:

- `E_recon_MS` is the MatterSim energy of the reconstructed slab.
- `E_clean_slab_MS` is the MatterSim energy of the fixed clean Ga24Sb24 slab.
- `n_Ga_adatom` and `n_Sb_adatom` are adatom counts only.
- `mu_Ga` and `mu_Sb` are MatterSim chemical potentials.
- `A` is the surface cell area.

This removes the fixed-slab constant and produces a physically more meaningful
surface reconstruction energy scale.

## Current Analysis Outputs

The main analysis script is:

```text
gasb-surface/run/analyze_adatom_chemical_potential.py
```

Important outputs are written under:

```text
gasb-surface/run/surface-stability-analysis/
```

Key files:

- `mattersim-reference-energies.csv`
- `mattersim-chemical-potential-limits.csv`
- `mattersim-clean-slab-energy.csv`
- `mattersim-surface-energies.csv`
- `adatom-formation-energy-curves.csv`
- `adatom-formation-energy-envelope.csv`
- `adatom-formation-energy-all-vs-dmu-sb.png`
- `adatom-formation-energy-all-vs-dmu-ga.png`
- `adatom-formation-energy-relative-all-vs-dmu-sb.png`
- `adatom-formation-energy-relative-all-vs-dmu-ga.png`

The current MatterSim-only envelope is:

```text
Ga-rich to almost the full window: Ga4Sb2
Very Sb-rich end: Ga4Sb3
```

## Lessons Learned

### Use One Energy Model Consistently

The most important lesson is that the energies in the formation-energy formula
must come from the same model.

Using EDDP for reconstructed slab energies and MatterSim for chemical potentials
is inconsistent. Even if the algebra is correct, the resulting formation energy
does not have a reliable physical meaning.

EDDP is still useful for large-scale structure search and pre-screening. But the
final thermodynamic ranking should be evaluated using one consistent reference
model, such as MatterSim or DFT.

### Grand Potential Is Not Automatically a Surface Formation Energy

The grand-potential expression:

```text
Omega = E - N_Ga * mu_Ga - N_Sb * mu_Sb
```

is useful for comparing structures, but for a slab it includes fixed slab and
bottom-surface contributions. Plotting this directly can give values that look
too large for a surface reconstruction energy.

For a fixed-slab reconstruction comparison, subtracting a clean slab reference is
more appropriate:

```text
E_recon - E_clean_slab - n_Ga_adatom * mu_Ga - n_Sb_adatom * mu_Sb
```

Then divide by surface area to get a physically interpretable energy density.

### Make the Formula Auditable

The curve CSV should include all inputs needed to re-evaluate the formula:

- energy model
- total Ga/Sb counts
- adatom Ga/Sb counts
- MatterSim reconstructed slab energy
- MatterSim clean slab energy
- chemical potentials
- surface area
- grand potential
- reconstruction formation energy

This made it straightforward to catch and fix mistakes.

### Plot All Curves, Not Only the Envelope

Plotting only near-envelope curves can hide most of the search space and make it
look as though only one composition exists. The useful set of plots is:

- all composition curves
- all curves relative to the envelope
- near-envelope zoom

The full comparison plot is needed for coverage; the relative and zoomed plots
are needed for stability interpretation.

### Be Explicit About Adatom vs Total Formula

The same structure can be described by:

```text
Total formula:  Ga28Sb26
Adatom formula: Ga4Sb2
```

For this project, the adatom formula is the relevant surface reconstruction
composition. The total formula is still useful for checking the actual structure
and for auditing atom counts.

## Mistakes Made

### Mixing EDDP and MatterSim Energies

The first formation-energy plots used EDDP `.res` enthalpies for the surface
structures but MatterSim energies for Ga, Sb, and GaSb chemical potentials.

This was the main conceptual error.

### Not Subtracting the Clean Slab

The first MatterSim-consistent correction used:

```text
E_recon_MS - N_Ga_total * mu_Ga - N_Sb_total * mu_Sb
```

This is a valid grand potential, but it still contains the clean slab/bottom
surface constant. The values were therefore too large for a reconstruction
formation energy.

### Calling a Grand-Potential Score a Formation Energy

Some intermediate plots were labeled as formation energy even though they were
closer to grand-potential scores. The final plots now use reconstruction
formation energy density after subtracting the clean slab.

### Only Plotting Near-Envelope Curves Initially

The first plots only showed structures close to the envelope, which made the
visual comparison incomplete. Full all-composition plots were added afterward.

### Treating EDDP-Minimum Structures as MatterSim-Minimum Structures

The current MatterSim-only analysis evaluates the EDDP-selected lowest structure
for each adatom composition using MatterSim. This removes the energy-model
mixing, but it does not prove that those are the MatterSim-lowest structures for
each composition.

## Remaining Limitation

The current workflow is:

```text
EDDP search -> choose lowest structure per adatom composition -> MatterSim single-point evaluation
```

For a stricter MatterSim ranking, the next step should be:

```text
EDDP search -> MatterSim evaluate or relax all candidates -> choose lowest structure per adatom composition by MatterSim
```

An even stricter workflow would use DFT for the final small candidate set.

