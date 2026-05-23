# Crystal Structure Prediction

Advanced techniques for large-scale structure searching with converged EDDP potentials.

## Prerequisites

- Converged EDDP potential from [AIRSS Workflow](../tutorials/airss_workflow.md)
- AIRSS tools installed (`buildcell`, `cabal`, `cryan`)
- Understanding of [Iterative Training](../concepts/iterative_training.md)

## Production RSS Workflow

Once your EDDP potential has converged (typically after 5-10 iterations), use it for large-scale structure prediction.

### Setup

```julia
using EDDPotentials

# Load converged potential
builder = Builder("link.toml")
ensemble = load_ensemble(builder, max(builder.state.iteration - 1, 0))

# Or load directly
artifact = load_model_artifact("workdir/ensemble-gen5.json")
ensemble = artifact.model
cf = artifact.feature_spec
```

### Run Production Search

```julia
# Configure RSS for production
builder.rss.max = 10000                  # Number of written structures
builder.rss.subfolder_name = "production"  # Separate from training
builder.rss.ensemble_id = 5                # Use final generation

# Run search
run_rss(builder)
```

**File organization:**
```
workdir/
├── gen0/, gen1/, ..., gen5/   # Training data
├── production/                 # Production search results
│   ├── CN-*.res               # Relaxed structures
│   └── rss.log                # Search log
└── ensemble-gen5.json          # Final model
```

### Parallel Searches

For maximum throughput, run multiple RSS processes in parallel:

```bash
# Run 64 parallel searches
parallel -j 64 julia -e "using EDDPotentials; run_rss(\"link.toml\")" ::: {1..1000}
```

Each process uses the `[rss]` settings from `link.toml`; `rss.max` is the
number of written structures per process.

## Analyzing Search Results

### Quick Ranking

```bash
cd workdir/production

# Rank all structures
ca -r -t 20  # Top 20 lowest energy

# Filter by composition
ca -f C2N -r -t 10  # Top 10 C2N structures

# Convex hull
ca -m -r  # Maxwell construction (thermodynamic stability)
```

### Systematic Analysis

```julia
using EDDPotentials
using ExtXYZ

# Load all structures
results_dir = "workdir/production"
structures = ExtXYZ.load(joinpath(results_dir, "*.res"))

# Analyze energies
energies = [s.energy / length(s) for s in structures]  # eV/atom
println("Min energy: ", minimum(energies))
println("Max energy: ", maximum(energies))
println("Structures within 0.1 eV/atom: ", count(e -> e - minimum(energies) < 0.1, energies))
```

### Deduplication

```bash
# Remove duplicates (similarity threshold 0.1)
cat production/*.res | cryan -u 0.1 -r > unique_structures.res

# Split back to individual files (optional)
cat unique_structures.res | resplit
```

**Threshold guidelines:**
- **0.01:** Very strict (only removes nearly identical structures)
- **0.05:** Standard (removes clear duplicates)
- **0.1:** Relaxed (removes similar structures)

### In-Process Low-Energy Pruning

For large single-composition searches, RSS can prune structures before writing
them. The search buffers relaxed candidates, computes pool energy statistics,
ranks by predicted enthalpy per atom, removes fingerprint duplicates, and writes
only the lowest-energy unique fraction.

```julia
builder = Builder("link.toml")
builder.rss.max = 10_000                 # Number of written structures
builder.rss.prune.enabled = true
builder.rss.prune.pool_size = 100
builder.rss.prune.keep_fraction = 0.10
builder.rss.prune.dedup_tol = 0.10
run_rss(builder)
```

In `link.toml`, configure standalone production RSS under `[rss.prune]`:

```toml
[rss.prune]
enabled = true
pool_size = 100
keep_fraction = 0.10
dedup_tol = 0.10
```

`rss.max` remains the number of structures written after pruning. Pruning is
disabled automatically for formula-sampling/random-composition searches because
short pools do not give reliable per-composition statistics.

## Composition-Specific Searches

### Binary Systems (C-N)

```
# C-N-SEARCH.cell
#SPECIES=C,N
#NFORM=1-4            # 1-4 formula units
#NATOM=4-16           # 4-16 atoms total
#SLACK=0.25
#MINSEP=1.5 C-C=1.2 N-N=1.2 C-N=1.2

%BLOCK LATTICE_CART
3.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 3.0
#ANGAMP=30
%ENDBLOCK LATTICE_CART
```

### Ternary Systems (Li-Ti-O)

```
# LTO-SEARCH.cell
#SPECIES=Li,Ti,O
#NFORM=1-4
#NATOM=8-32
#SLACK=0.25
#MINSEP=1.8

# Focus on specific compositions
#FOCUS=2,3           # Binary and ternary compounds only

%BLOCK LATTICE_CART
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
#ANGAMP=30
%ENDBLOCK LATTICE_CART
```

**FOCUS parameter:**
- `#FOCUS=1`: Unary (pure elements)
- `#FOCUS=2`: Binary (AB)
- `#FOCUS=3`: Ternary (ABC)
- `#FOCUS=2,3`: Binary and ternary
- `#FOCUS={1,2,2,2,3,3}`: Weighted random (1:3:2 ratio)

### Formula-Space Sampling via `link.toml`

Instead of relying on `#FOCUS`, you can let `Builder` sample explicit reduced formulas by injecting `#FORMULA=...` at runtime:

```toml
[state.rss]
formula_sampling_enabled = true
formula_max_reduced_formula_coeff = 6
formula_require_charge_neutral = true

[state.rss.formula_oxidation_states]
Li = [1]
Ta = [5]
O = [-2]
Cl = [-1]
```

`formula_max_reduced_formula_coeff` is the largest stoichiometric coefficient allowed in reduced formulas.
For example, `Li2TaO3` has max coefficient `3`.

### Fixed Composition

```
# Exactly Li2TiO3
#SPECIES=Li,Ti,O
#NATOM=Li=2,Ti=1,O=3  # Exact stoichiometry
#NFORM=1-3            # 1-3 formula units → 6-18 atoms
```

## Pressure-Dependent Searches

Search at non-zero pressure for high-pressure phases.

### Configure Pressure

```julia
# In link.toml
[state.rss]
pressure_gpa = 10.0  # 10 GPa pressure
```

Or modify buildcell file:
```
#PRESSURE=10.0   # External pressure in GPa
```

**Applications:**
- High-pressure polymorphs (diamond, post-perovskite)
- Preventing low-density structures
- Phase diagrams under pressure

## Advanced buildcell Options

### Symmetry Constraints

```
#SYMMOPS=2-8          # Enforce 2-8 symmetry operations
#SG=194               # Specific space group (P63/mmc)
#SGRANK=1-10          # Sample from 10 highest-symmetry groups
```

### Vacancy and Defects

```
#SPECIES=C,VACANCY
#NATOM=C=15,VACANCY=1  # 15 C atoms + 1 vacancy → 16-atom cell
```

### Molecular Fragments

```
#SPECIES=C,H
#MOLFORM=CH4          # Start with methane molecules
#NMOL=4               # 4 CH4 molecules per cell
```

## Multi-Composition Phase Diagrams

For composition ranges (Li₁₋ₓMₓ), combine results from multiple searches.

### Strategy 1: Separate Searches

```bash
# Generate for different compositions
for x in 0.0 0.25 0.5 0.75 1.0; do
    # Modify NATOM in SEARCH.cell for each x
    run_rss(...)
done

# Combine results
cat */production/*.res | cryan -m -r > phase_diagram.res
```

### Strategy 2: Random Composition

```
#SPECIES=Li,M
#NFORM=1-4            # Random Li:M ratio
#NATOM=8-24
#FOCUS={2}            # Binary compounds only
```

## Selecting Structures for DFT Validation

EDDP errors typically 0.1-0.2 eV/atom. Use statistical selection:

### Selection Criteria

```bash
# 1. Extract convex hull
cat production/*.res | cryan -m -r > hull_structures.res

# 2. Select structures within energy window
# For each composition, keep structures within 0.2 eV/atom of minimum
cat production/*.res | cryan -f C2N | cryan -r | head -n 50 > C2N_candidates.res
cat production/*.res | cryan -f C3N | cryan -r | head -n 30 > C3N_candidates.res

# 3. Deduplicate
cat *_candidates.res | cryan -u 0.1 -r > for_dft.res

# 4. Split for processing
cat for_dft.res | resplit
```

### Batch DFT with crud.pl

```bash
# Move structures to hopper
cp for_dft/*.res hopper/

# Run 8 parallel crud.pl instances (8 cores each)
parallel -j 8 "crud.pl -mpinp 8" ::: {1..8}

# Monitor progress
watch -n 10 "ls hopper/ | wc -l"  # Remaining structures
```

## Post-Processing with EDDP

### Rerank with DFT Energies

```julia
# Load DFT-calculated structures
dft_structures = ExtXYZ.load("good_castep/*.res")

# Extract DFT energies
dft_energies = [s.energy / length(s) for s in dft_structures]

# Compare to EDDP predictions
eddp_energies = [predict_energy(ensemble, cf, s) / length(s) for s in dft_structures]

# Analyze correlation
using Statistics
correlation = cor(eddp_energies, dft_energies)
println("Spearman correlation: ", correlation)

# Identify structures with large errors (outliers)
errors = abs.(eddp_energies .- dft_energies)
outliers = findall(e -> e > 0.2, errors)
println("Structures with >0.2 eV/atom error: ", length(outliers))
```

### Update Potential

If DFT reveals systematic errors, retrain:

```julia
# Add DFT structures to training set
all_structures = vcat(load_structures(builder, 0:5), dft_structures)

# Retrain
fc = FeatureStore(StructureContainer(all_structures), cf)
# ... train new ensemble
```

## Performance Optimization

### RSS Throughput

**Computational cost:**
- EDDP relaxation: ~0.1-1 second per structure (CPU)
- Parallelization: Limited by random number generation overhead

**Optimal configuration:**
```julia
# For 64-core node
builder.state.buildcell_nprocs = 64  # 64 parallel buildcell processes for generation 0
builder.state.build_timeout = 60.0   # Per-build timeout in seconds
builder.rss.max = 50000              # Large production RSS batch
```

**Expected throughput:**
- Single core: ~1000 structures/hour
- 64 cores: ~30,000-50,000 structures/hour

### Storage Management

```bash
# Compress intermediate results
gzip production/*.res

# Pack structures (reduces inodes)
cat production/*.res | respack > production_packed.res
rm production/*.res

# Archive training data
tar -czf training_data.tar.gz gen*/
```

## Troubleshooting

**Problem:** All structures converge to same minimum

**Solution:** Increase diversity
```
#SLACK=0.5           # Increase volume variation
#ANGAMP=45           # Larger angle perturbations
```

**Problem:** Many high-energy failures

**Solution:** Tighten constraints
```
#MINSEP=2.0          # Prevent atomic overlap
#COMPACT             # Favor compact structures
```

**Problem:** Missing compositions in ternary search

**Solution:** Targeted FOCUS
```
#FOCUS={1,1,2,2,2,3,3,3}  # 2:3:3 ratio for 1:2:3 compositions
```

## Next Steps

- [Run Phonon Calculations](run_phonon_calculations.md) - Validate dynamic stability
- [Understanding Features](../concepts/features.md) - Optimize features for specific systems
- [AIRSS Workflow](../tutorials/airss_workflow.md) - Build new potentials
