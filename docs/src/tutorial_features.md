# Tutorial: Understanding and Selecting Features

Features are the mathematical descriptors that encode atomic environments. Choosing good features is crucial for building accurate EDDP potentials. This tutorial explains what features are, how they work, and how to select them for your system.

## What Are EDDP Features?

EDDP features describe the local environment around each atom in a way that:
- Is invariant to rotation and translation (moving/rotating the system doesn't change features)
- Respects periodic boundary conditions
- Smoothly goes to zero at a cutoff radius
- Captures both distances (two-body) and angles (three-body)

Think of features as a "fingerprint" of the atomic environment. Similar environments produce similar features, which helps the neural network learn patterns.

## Two-Body Features

Two-body features describe pairwise interactions between an atom and its neighbors.

### Mathematical Form

For each neighbor atom j at distance r from atom i, we compute:

```
f(r) = exp(-(r/r0)^p) * cutoff(r)
```

Where:
- `p` is a power parameter (typically 2-6)
- `r0` is a scaling factor
- `cutoff(r)` smoothly goes to zero at the cutoff radius

### Physical Interpretation

- **Small p (p=2)**: Features decay slowly with distance, capture long-range structure
- **Large p (p=6)**: Features decay quickly, sensitive to nearest neighbors only
- Using multiple p values lets the network learn both short and long-range interactions

### Example

```julia
using EDDPotentials

# Create two-body features for C-C interactions
# Powers 2, 3, 4 with 4.0 Å cutoff
feat2 = TwoBodyFeature([2, 3, 4], (:C, :C), 4.0)

println("Number of features: ", nfeatures(feat2))  # 3 (one per power)
println("Cutoff radius: ", feat2.cutoff, " Å")
println("Powers: ", feat2.p)
```

For carbon with cutoff 4.0 Å:
- p=2: Captures up to ~3rd neighbor shell in graphite
- p=4: Mainly nearest neighbors
- p=6: Very localized, ~1st neighbors only

### Why Multiple Powers?

Using multiple powers gives the network flexibility:

```julia
# Minimal features (may miss long-range effects)
feat_minimal = TwoBodyFeature([4], (:C, :C), 4.0)  # Only p=4

# Standard features (good balance)
feat_standard = TwoBodyFeature([2, 3, 4, 5], (:C, :C), 4.0)  # 4 powers

# Detailed features (more flexibility, slower training)
feat_detailed = TwoBodyFeature([2, 3, 4, 5, 6, 7], (:C, :C), 4.5)  # 6 powers
```

More powers = more features = more expressive model, but also:
- Longer training time
- More risk of overfitting with limited data
- Diminishing returns beyond p=2:6

## Three-Body Features

Three-body features capture angular information (bond angles) which is critical for directional bonding.

### Mathematical Form

For each triplet of atoms (i, j, k) where i is the central atom:

```
f(r_ij, r_ik, θ) = exp(-(r_ij/r0)^p) * exp(-(r_ik/r0)^p) * (cos(θ) + 1)^q * cutoff(r_ij) * cutoff(r_ik)
```

Where:
- `p` controls radial decay (like two-body)
- `q` controls angular sensitivity
- `θ` is the angle between bonds i-j and i-k

### Physical Interpretation

- **q=1**: Weakly sensitive to angles, roughly captures coordination
- **q=2**: Moderate angular sensitivity, distinguishes ~30° differences
- **q=3**: Strong angular sensitivity, captures precise bonding angles

### Example

```julia
# Three-body features for C-C-C triplets
# Radial powers [2,3,4], angular powers [1,2], cutoff 3.5 Å
feat3 = ThreeBodyFeature([2, 3, 4], [1, 2], (:C, :C, :C), 3.5)

println("Number of features: ", nfeatures(feat3))  # 6 (3 p × 2 q)
```

### Why Three-Body Features Are Essential

Two-body features alone cannot distinguish:
- Graphite vs diamond (different bond angles but similar distances)
- Square vs tetrahedral coordination
- Bent vs linear molecules

Three-body features resolve these ambiguities:

```julia
using EDDPotentials

# Two-body only (NOT RECOMMENDED for carbon)
cf_2body = CellFeature([:C]; p3=Int[], rcut3=0.0)  # No three-body

# Two-body + three-body (RECOMMENDED)
cf_full = CellFeature([:C])  # Includes both by default

println("Two-body only: ", nfeatures(cf_2body), " features")
println("Full features: ", nfeatures(cf_full), " features")
```

For carbon, always include three-body features - they're essential for capturing sp, sp2, sp3 bonding.

## Cutoff Radius Selection

The cutoff radius determines how many neighbors contribute to the atomic environment.

### Trade-offs

**Short cutoff (2.5-3.0 Å for carbon)**:
- ✓ Fast computation
- ✓ Good for local properties (single bonds, local coordination)
- ✗ Misses long-range order
- ✗ May fail for layered structures (graphite)

**Medium cutoff (3.5-4.5 Å for carbon)**:
- ✓ Captures 1st and 2nd neighbor shells
- ✓ Good general-purpose choice
- ✓ Reasonable computational cost
- ✓ Works for diverse structure types

**Long cutoff (5.0+ Å for carbon)**:
- ✓ Captures long-range structure
- ✓ Necessary for dispersion-bound systems
- ✗ Expensive (many neighbors)
- ✗ May need more training data

### Choosing Cutoff for Your System

```julia
using EDDPotentials
using ExtXYZ

# Load a typical structure
structure = ExtXYZ.load("carbon.xyz")[1]

# Check neighbor statistics at different cutoffs
for rcut in [3.0, 3.5, 4.0, 4.5, 5.0]
    # Count average neighbors per atom
    nl = neighborlist(structure, rcut)
    avg_neighbors = length(nl.i) / length(structure)
    println("Cutoff $rcut Å: $(round(avg_neighbors, digits=1)) neighbors/atom")
end
```

Guidelines:
- **Ionic/metallic**: 4-6 Å (longer range interactions)
- **Covalent (C, Si)**: 3.5-4.5 Å (2-3 coordination shells)
- **Molecular**: 3-4 Å (nearest neighbors only)

Start with the default for your element and increase if:
- Test errors are high despite good training error
- You're modeling layered structures (graphite, MoS₂)
- Long-range order is important (ice, clathrates)

### Separate Cutoffs for Two-Body and Three-Body

Three-body features are more expensive (scale as N²neighbors). Use shorter cutoffs:

```julia
# Different cutoffs for two-body and three-body
cf = CellFeature(
    [:C];
    rcut2=4.5,  # Longer cutoff for two-body (cheaper)
    rcut3=3.5   # Shorter cutoff for three-body (expensive)
)
```

This is the default behavior and works well for most systems.

## The CellFeature Constructor

The `CellFeature` type is a high-level interface that sets up both two-body and three-body features:

```julia
# Default features (recommended starting point)
cf = CellFeature([:C])

# Equivalent explicit form
cf = CellFeature(
    [:C];
    p2=2:6,              # Two-body powers: [2,3,4,5,6]
    p3=2:5,              # Three-body radial powers: [2,3,4,5]
    q3=1:3,              # Three-body angular powers: [1,2,3]
    rcut2=4.0,           # Two-body cutoff: 4.0 Å
    rcut3=3.5,           # Three-body cutoff: 3.5 Å
    geometry_sequence=true  # Use geometric spacing for powers
)
```

### Parameter Meanings

**`p2`**: Two-body powers (default `2:6`)
- Each power creates one feature per neighbor pair
- More powers = more expressive, but diminishing returns beyond 6

**`p3`**: Three-body radial powers (default `2:5`)
- Controls how quickly three-body features decay with distance
- Usually fewer than `p2` because three-body is expensive

**`q3`**: Three-body angular powers (default `1:3`)
- Controls angular resolution
- q=1,2,3 is usually sufficient; higher values rarely help

**`rcut2`, `rcut3`**: Cutoff radii
- System-dependent, see section above
- Default 4.0 Å for two-body, 3.5 Å for three-body

**`geometry_sequence`**: If true, uses geometric spacing for powers
- Provides better coverage of distance scales
- Usually improves accuracy slightly

### Feature Count

Total features = (# two-body) + (# three-body):

```julia
cf = CellFeature([:C]; p2=2:6, p3=2:5, q3=1:3)

n_twobody = length(2:6)  # 5
n_threebody = length(2:5) * length(1:3)  # 4 × 3 = 12
n_total = n_twobody + n_threebody  # 17

println("Total features: ", nfeatures(cf))  # 17
```

More features require more training data to avoid overfitting.

## Feature Selection Strategies

### Start Simple, Add Complexity

```julia
# Strategy 1: Minimal features (fast iteration)
cf_minimal = CellFeature([:C]; p2=3:5, p3=3:4, q3=1:2)
println("Minimal: ", nfeatures(cf_minimal), " features")

# Strategy 2: Standard features (good for most cases)
cf_standard = CellFeature([:C])  # Uses defaults
println("Standard: ", nfeatures(cf_standard), " features")

# Strategy 3: Detailed features (if standard isn't accurate enough)
cf_detailed = CellFeature([:C]; p2=2:7, p3=2:6, q3=1:3, rcut2=4.5)
println("Detailed: ", nfeatures(cf_detailed), " features")
```

Train with minimal features first. If RMSE is too high, add complexity gradually.

### Multiple Elements

For multi-element systems, features are generated for all pair/triplet types:

```julia
# Carbon and hydrogen
cf = CellFeature([:C, :H])

# This creates features for:
# Two-body: C-C, C-H, H-H
# Three-body: C-C-C, C-C-H, C-H-H, H-H-H

println("Total features: ", nfeatures(cf))  # More features than single-element
```

For systems with many elements, you may need:
- More training data (each pair/triplet needs examples)
- Simplified feature sets (fewer powers)

### Optimizing for Speed vs Accuracy

**For fast predictions** (production MD, structure screening):
```julia
cf_fast = CellFeature(
    [:C];
    p2=3:5,      # Fewer two-body powers
    p3=3:4,      # Fewer three-body powers
    q3=1:2,      # Fewer angular powers
    rcut2=3.5,   # Shorter cutoffs
    rcut3=3.0
)
```

**For maximum accuracy** (benchmarking, careful predictions):
```julia
cf_accurate = CellFeature(
    [:C];
    p2=2:7,
    p3=2:6,
    q3=1:3,
    rcut2=5.0,
    rcut3=4.0
)
```

In most cases, standard features provide the best speed/accuracy balance.

## Practical Guidelines

### Do's

✓ **Start with default CellFeature** - it works well for most systems
✓ **Use three-body features** - essential for covalent/molecular systems
✓ **Check cutoff captures relevant neighbors** - use neighborlist statistics
✓ **Match cutoff to your training data** - if trained on bulk, use bulk-appropriate cutoffs
✓ **Test on held-out data** - ensure features generalize

### Don'ts

✗ **Don't use only two-body features** - will fail for angular systems
✗ **Don't use huge cutoffs without justification** - expensive and often unnecessary
✗ **Don't add complexity without testing** - more features ≠ better accuracy
✗ **Don't use different features for train and test** - must be identical
✗ **Don't forget to save your CellFeature** - you'll need it for predictions

## Example: Comparing Feature Sets

Let's compare different feature choices on carbon structures:

```julia
using EDDPotentials
using Flux  # For neural network construction
using ExtXYZ  # For loading .xyz files
import EDDPotentials: train!, FluxInterface, create_ensemble, get_fit_data

# Load data
structures = ExtXYZ.load("carbon_data.xyz")
sc = StructureContainer(structures)

# Test three feature sets
feature_sets = [
    ("Minimal", CellFeature([:C]; p2=3:5, p3=3:4, q3=1:2)),
    ("Standard", CellFeature([:C])),
    ("Detailed", CellFeature([:C]; p2=2:7, p3=2:6, q3=1:3, rcut2=4.5))
]

for (name, cf) in feature_sets
    println("\n=== $name features ($(nfeatures(cf)) total) ===")

    # Create containers and split
    fc = FeatureContainer(cf, sc)
    fc_train, fc_test = train_test_split(fc)

    # Train single model
    chain = Chain(Dense(nfeatures(cf), 20, tanh), Dense(20, 1))
    model = FluxInterface(chain)
    train!(model, fc_train, fc_test; maxIter=300, show_progress=false, earlystop=30)

    # Evaluate
    results = TrainingResults(model, fc_test)
    println("Test RMSE: $(rmse_per_atom(results)) eV/atom")
    println("Test MAE: $(mae_per_atom(results)) eV/atom")
end
```

Typical results for carbon:
- **Minimal**: ~5-10 meV/atom RMSE (fast but may miss subtleties)
- **Standard**: ~2-5 meV/atom RMSE (good balance)
- **Detailed**: ~1-3 meV/atom RMSE (best accuracy, slower)

## Advanced: Custom Feature Specifications

For full control, create `TwoBodyFeature` and `ThreeBodyFeature` directly:

```julia
# Custom two-body features with non-sequential powers
feat2_custom = TwoBodyFeature([2, 4, 6, 8], (:C, :C), 4.5)

# Custom three-body features
feat3_custom = ThreeBodyFeature([2, 4], [1, 3], (:C, :C, :C), 3.5)

# Combine into CellFeature
cf_custom = CellFeature(
    [:C],
    [feat2_custom],  # Vector of two-body features
    [feat3_custom]   # Vector of three-body features
)
```

This is rarely needed - `CellFeature(elements; kwargs...)` covers most use cases.

## Summary

**Key Takeaways**:

1. **Two-body features** capture distances; **three-body features** capture angles
2. **Use both** for covalent/molecular systems (always for carbon)
3. **Powers p**: 2-6 for two-body, 2-5 for three-body (radial)
4. **Angular powers q**: 1-3 is sufficient
5. **Cutoff radius**: 3.5-4.5 Å for carbon; adjust for your system
6. **Start simple**: Default `CellFeature([:C])` works well
7. **Add complexity only if needed**: More features require more data

**Feature Selection Checklist**:
- [ ] Default features tried first
- [ ] Cutoff captures relevant neighbors (check with neighborlist)
- [ ] Three-body features included
- [ ] Test RMSE acceptable for application
- [ ] Features don't change between training and prediction
- [ ] CellFeature saved with model for later use

For most applications, the default `CellFeature` provides excellent results. Focus on getting good training data rather than over-optimizing features.

## Further Reading

- See the API documentation for [`TwoBodyFeature`](@ref), [`ThreeBodyFeature`](@ref), [`CellFeature`](@ref)
- For multi-element systems, see examples in the test suite
- For structure-specific features (surfaces, defects), consider customizing cutoffs
