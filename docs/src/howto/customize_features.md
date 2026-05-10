# Customize Features

Guide for tuning feature parameters (cutoff radii, power sequences, species resolution) to improve potential accuracy for your specific system.

## Quick Reference

```julia
# Default — works for most systems
cf = CellFeature([:C])

# Shorter cutoffs for molecular systems
cf = CellFeature([:C, :H]; rcut2=3.5, rcut3=3.0)

# Longer cutoffs for metallic/ionic systems
cf = CellFeature([:Li, :Ti, :O]; rcut2=5.0, rcut3=4.5)

# Fewer features for faster training
cf = CellFeature([:C]; p2=2:6, p3=2:4, q3=1:2)

# Geometric spacing for better range coverage (recommended)
cf = CellFeature([:C]; p2=2:8, geometry_sequence=true)
```

## Choosing Cutoff Radii

The cutoff radius `rcut` determines how many neighbor atoms contribute to each feature vector. It should be large enough to capture the relevant coordination shells but not so large that computational cost becomes prohibitive.

### Guidelines by Bonding Type

| System type | Recommended `rcut2` | Recommended `rcut3` | Notes |
|---|---|---|---|
| Molecular | 3.0–4.0 Å | 2.5–3.5 Å | Only nearest neighbors matter |
| Covalent (C, Si, Ge) | 3.5–4.5 Å | 3.0–4.0 Å | 2–3 coordination shells |
| Ionic | 4.0–5.5 Å | 3.5–5.0 Å | Long-range Coulomb interactions |
| Metallic | 4.5–6.0 Å | 4.0–5.0 Å | Extended coordination |

### Practical Check

Inspect neighbor statistics on representative structures to verify your cutoff captures the relevant shells:

```julia
using EDDPotentials

cf = CellFeature([:C]; rcut2=4.0, rcut3=3.5)
for (i, structure) in enumerate(structures[1:5])
    nl = NeighbourList(structure, suggest_rcut(cf))
    println("Structure $i: $(length(nl.nneigh)) atoms, max neighbors: $(maximum(nl.nneigh))")
end
```

If structures in your search have very different densities from your training data, increase cutoffs conservatively and re-check validation errors.

### Separate Two-Body and Three-Body Cutoffs

Three-body features scale as O(N_neighbors^2), so they are much more expensive. Use a shorter cutoff for three-body features:

```julia
cf = CellFeature([:C];
    rcut2=4.5,   # Longer for two-body (cheap per-atom sum)
    rcut3=3.5    # Shorter for three-body (triplet enumeration)
)
```

This is the default behavior when `rcut3` is not specified — it defaults to `rcut2`.

## Choosing Power Sequences

The power exponents `p2`, `p3`, `q3` control the flexibility of the feature basis:

- **Low powers (2–3)**: Slowly decaying features that capture long-range interactions
- **High powers (6–8)**: Rapidly decaying features sensitive to nearest neighbors
- **More powers**: More expressive model, but higher risk of overfitting and longer training

### Default Sequences

```julia
# Linear spacing (default)
cf = CellFeature([:C]; p2=2:8)  # [2, 3, 4, 5, 6, 7, 8]

# Geometric spacing (recommended for wide ranges)
cf = CellFeature([:C]; p2=2:8, geometry_sequence=true)
# Produces approximately [2, 2.5, 3.2, 4.0, 5.0, 6.3, 7.9]
```

Geometric spacing gives better coverage of both short and long range interactions with fewer features.

### Minimal vs Standard vs Detailed

```julia
# Minimal — fast training, lower accuracy (for testing)
cf_minimal = CellFeature([:C]; p2=3:5, p3=3:4, q3=1:2)

# Standard — good balance for most systems
cf_standard = CellFeature([:C])  # Uses defaults: p2=2:8, p3=2:8, q3=2:8

# Detailed — higher accuracy, slower training
cf_detailed = CellFeature([:C]; p2=2:10, p3=2:8, q3=1:4, rcut2=4.5)
```

### Diminishing Returns

Adding more powers beyond `p=8` or `q=4` rarely improves accuracy because the features become nearly linearly dependent. If you need more expressiveness, increase the network size instead.

## Multi-Element Systems

For systems with multiple elements, `CellFeature` automatically generates all species combinations:

```julia
# Binary system: generates 4 two-body + 4 three-body types
cf = CellFeature([:C, :N])
# Two-body: C-C, C-N, N-C, N-N
# Three-body: C-C-C, C-C-N, C-N-N, N-C-C, N-C-N, N-N-C, N-N-N, N-N-N
#            (ordered center, unordered neighbors)
```

Feature count grows quadratically with the number of elements:
- 1 element: 1 two-body + 1 three-body type
- 2 elements: 4 two-body + 8 three-body types
- 3 elements: 9 two-body + 27 three-body types

For ternary systems, consider reducing the power ranges to keep training manageable:

```julia
cf = CellFeature([:Li, :Ti, :O]; p2=2:6, p3=2:5, q3=1:2, rcut2=4.5)
```

## Combining Feature Sets

Use `+` to combine feature sets from different element groups or with different parameters:

```julia
# Standard carbon features
cf_c = CellFeature([:C]; rcut2=4.0)

# Hydrogen features with shorter cutoff
cf_h = CellFeature([:C, :H]; rcut2=3.0, rcut3=2.5)

# Combined
cf = cf_c + cf_h
```

## Comparing Feature Sets

Always validate feature choices on held-out data:

```julia
using EDDPotentials, Flux, ExtXYZ
import EDDPotentials: train!, FluxInterface, train_val_split

structures = ExtXYZ.load("data.xyz")
sc = StructureContainer(structures)

feature_sets = [
    ("Minimal", CellFeature([:C]; p2=3:5, p3=3:4, q3=1:2)),
    ("Standard", CellFeature([:C])),
    ("Detailed", CellFeature([:C]; p2=2:8, geometry_sequence=true, rcut2=4.5)),
]

for (name, cf) in feature_sets
    fc = FeatureContainer(cf, sc)
    fc_train, fc_val = train_val_split(fc)

    chain = Chain(Dense(nfeatures(cf), 20, tanh_fast), Dense(20, 1))
    model = FluxInterface(chain)
    train!(model, fc_train, fc_val; maxIter=300, show_progress=false, earlystop=30)

    results = TrainingResults(model, fc_val)
    println("$name ($(nfeatures(cf)) features): RMSE = $(round(rmse_per_atom(results), digits=4)) eV/atom")
end
```

Choose the feature set that gives the best validation RMSE without overfitting (training RMSE much lower than validation RMSE signals overfitting — use fewer features or more training data).

## Next Steps

- [Understanding Features](../concepts/features.md) — mathematical details of feature design
- [AIRSS Workflow](../tutorials/airss_workflow.md) — applying features in automated potential building
- [Batched LM Training](batched_lm_training.md) — training with large feature sets

## Migration Guide: `CellFeature` Refactor

The `CellFeature` type has been refactored to a parametric immutable struct. The `TwoBodyFeature` and `ThreeBodyFeature` types have been removed. The keyword constructor `CellFeature(elements; p2=..., ...)` is unchanged, so most user code works without modification.

### Accessing Feature Parameters

The old `cf.two_body` and `cf.three_body` fields no longer exist. Access parameters directly:

```julia
# Old
cf.two_body[1].p          # first two-body p-vector
cf.two_body[1].rcut       # two-body cutoff
cf.three_body[1].q        # first three-body q-vector
cf.three_body[1].np       # number of three-body p-exponents

# New
cf.p2                     # two-body p-vector (shared across all channels)
cf.rcut2                  # two-body cutoff
cf.q3                     # three-body q-vector (shared across all channels)
length(cf.p3)             # number of three-body p-exponents
```

### Feature Index Lookups

Instead of iterating `cf.two_body` / `cf.three_body` tuples and matching species with `permequal`, use the O(1) lookup methods:

```julia
# Old: iterate all two-body features
for (ife2, f) in enumerate(cf.two_body)
    if permequal(f.species, (si, sj))
        # access f.p[m], f.f, f.g, f.rcut
    end
end

# New: direct O(1) channel lookup
ch = two_body_channel(cf, ci, cj)
rng = two_body_range(cf, ci, cj)
# Use cf.p2[m], cf.f2, cf.g2, cf.rcut2 directly
```

### Species Mapping

```julia
# Old: created separate FeatureSpeciesLookup
lookup = FeatureSpeciesLookup(cf)

# New: use CellFeature methods directly
spec_index(cf, :C)        # Symbol → Int
spec_symbol(cf, 1)        # Int → Symbol
```

### Feature Counts

```julia
# Old
nfeatures(cf.two_body[1])   # features per two-body channel
nfeatures(cf.three_body[1])  # features per three-body channel

# New
nfeatures_per_channel(cf, 2)  # features per two-body channel
nfeatures_per_channel(cf, 3)  # features per three-body channel
nchannels(cf, 2)              # number of two-body channels
nchannels(cf, 3)              # number of three-body channels
```

