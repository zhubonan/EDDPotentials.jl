# Tutorial: Understanding Features

Features are mathematical descriptors encoding atomic environments. This tutorial shows how to select and customize features for accurate EDDP potentials.

## What Are EDDP Features?

EDDP features describe local atomic environments with properties that:
- Are rotation/translation invariant
- Respect periodic boundary conditions
- Smoothly decay to zero at cutoff radius
- Capture distances (two-body) and angles (three-body)

Think of features as "fingerprints" of atomic environments - similar environments produce similar features, helping the neural network learn patterns.

## Design Philosophy: From Lennard-Jones to EDDP

EDDP features are inspired by classical interatomic potentials, particularly the Lennard-Jones potential. Understanding this connection reveals why EDDP features have their specific mathematical form.

### Lennard-Jones Potential

The classic Lennard-Jones potential describes pairwise interactions:

```math
E = \sum_{j \neq i} \left(\frac{A}{r_{ij}^{12}} - \frac{B}{r_{ij}^{6}}\right)
```

This can be rewritten as a **linear combination**:

```math
E = w_1 \cdot \sum_j \frac{1}{r^{12}} + w_2 \cdot \sum_j \frac{1}{r^{6}} = \mathbf{w} \cdot \mathbf{F}
```

where `$\mathbf{w}$` is the weight vector and `$\mathbf{F}$` is the feature vector (dot product).

**Key insights:**
- Energy is **linear in weights** (w₁, w₂)
- Features are **sums over neighbors** (Σ_j)
- Different **powers** (12, 6) capture different interaction ranges

### EDDP Generalizes This Approach

EDDP extends the Lennard-Jones idea in three crucial ways:

**1. Controlled Range Cutoff**

Lennard-Jones uses `$f(r) = 1/r$`, which extends to infinity. This creates computational problems for condensed systems. EDDP uses:

```math
f(r) = \sqrt{2(1 - r/r_c)} \quad \text{for } r \leq r_c
```

```math
f(r) = 0 \quad \text{for } r > r_c
```

**Why this form?**
- Smoothly goes to zero at cutoff rc
- Has zero gradient at rc when raised to powers p ≥ 2
- Avoids artificial discontinuities in forces
- Computationally efficient (only neighbors within rc matter)

**2. Multiple Powers for Flexibility**

Instead of just two powers (12, 6), EDDP uses multiple powers `$p = 2, 3, 4, 5, 6$`:

```math
E = w_2 \sum_j f(r)^2 + w_3 \sum_j f(r)^3 + w_4 \sum_j f(r)^4 + \cdots
```

This gives the neural network flexibility to learn both short-range (high p) and long-range (low p) interactions without manually choosing specific functional forms.

**3. Three-Body Extensions**

Classical potentials like Lennard-Jones are purely two-body. EDDP adds three-body terms:

```math
E^{(3)} = \sum_j \sum_{k>j} w_{pq} \cdot f(r_{ij})^p \cdot f(r_{ik})^p \cdot f(r_{jk})^q
```

This **separable form** efficiently captures angular information through the three pairwise distances `$(r_{ij}, r_{ik}, r_{jk})$`, which uniquely determine the triangle geometry and bond angles.

### Linear, Homogeneous, Separable

These three properties make EDDP computationally efficient:

- **Linear in weights**: Fast training with standard optimization
- **Homogeneous in distances**: All terms are powers of f(r)
- **Separable**: Three-body = product of one-body functions

This design balances physical motivation (Lennard-Jones-like) with computational efficiency and machine learning flexibility.

## Two-Body Features

Two-body features describe pairwise interactions. For each neighbor at distance r, the basis function is:

```math
f(r) = \sqrt{2(1 - r/r_c)} \quad \text{for } r \leq r_c
```

```math
f(r) = 0 \quad \text{for } r > r_c
```

This function is smooth and goes to zero at the cutoff radius `$r_c$`. Each feature is `$\sum_j f(r_{ij})^p$` where:
- **p** is a power parameter (typically 2-6)
- **Small p (2-3)**: Features decay slowly, capture long-range structure
- **Large p (5-6)**: Features decay quickly, sensitive to nearest neighbors only

### Example

```julia
using EDDPotentials

# Create two-body features for C-C interactions
feat2 = TwoBodyFeature([2, 3, 4], (:C, :C), 4.0)

println("Features: ", nfeatures(feat2))  # 3 (one per power)
```

### Why Multiple Powers?

Using multiple powers gives the network flexibility to learn both short and long-range interactions:

```julia
using EDDPotentials: TwoBodyFeature
# Minimal (may miss long-range effects)
feat_minimal = TwoBodyFeature([4], (:C, :C), 4.0)

# Standard (good balance)
feat_standard = TwoBodyFeature([2, 3, 4, 5], (:C, :C), 4.0)
```

More powers = more expressive model, but also longer training and higher overfitting risk. Diminishing returns beyond p=2:6.

## Three-Body Features

Three-body features capture angular information essential for directional bonding. They use a separable form that depends on all three pairwise distances in a triplet.

### Mathematical Form

For atom triplet (i, j, k) with i central, using the same `$f(r)$` function as two-body:

```math
F_i = \sum_{j\nei} \sum_{k>j\nei} f(r_{ij})^p \cdot f(r_{ik})^p \cdot f(r_{jk})^q
```

Where:
- **p** controls how quickly the feature decays with distances from the central atom i
- **q** controls sensitivity to the distance between j and k atoms
- The triplet geometry (including bond angle) is implicitly encoded through the three distances

### Example

```julia
using EDDPotentials: ThreeBodyFeature
# Three-body features for C-C-C triplets
feat3 = ThreeBodyFeature([2, 3, 4], [1, 2], (:C, :C, :C), 3.5)

println("Features: ", nfeatures(feat3))  # 6 (3p × 2q)
```

### Why Three-Body Features Are Essential

Two-body features alone cannot distinguish:
- Graphite vs diamond (different angles, similar distances)
- Square vs tetrahedral coordination
- Bent vs linear molecules

```julia
using EDDPotentials: CellFeature
# Two-body only (NOT RECOMMENDED for carbon)
cf_2body = CellFeature([:C]; p3=Int[], rcut3=0.0)

# Full features (RECOMMENDED)
cf_full = CellFeature([:C])  # Includes both by default

println("Two-body only: ", nfeatures(cf_2body))
println("Full features: ", nfeatures(cf_full))
```

For carbon, always include three-body features - essential for sp, sp2, sp3 bonding.

## Cutoff Radius Selection

Cutoff radius determines how many neighbors contribute to features.

### System-Specific Recommendations

- **Ionic/metallic**: 4-6 Å (longer range)
- **Covalent (C, Si)**: 3.5-4.5 Å (2-3 shells)
- **Molecular**: 3-4 Å (nearest neighbors)

### Check Neighbor Statistics

Before finalizing cutoffs, inspect neighbor counts on representative structures.
As a practical target, choose `rcut2`/`rcut3` so the dominant coordination shells
are included without producing excessively dense neighbor lists.

If train and production structures have very different densities/volumes, increase
cutoffs conservatively and re-check validation errors.

### Separate Two-Body and Three-Body Cutoffs

Three-body features scale as N²neighbors, so use shorter cutoffs:

```julia
cf = CellFeature(
    [:C];
    rcut2=4.5,  # Longer for two-body (cheaper)
    rcut3=3.5   # Shorter for three-body (expensive)
)
```

This is the default and works well for most systems.

## CellFeature Constructor

High-level interface for setting up features:

```julia
# Default (recommended starting point)
cf = CellFeature([:C])

# Equivalent explicit form
cf = CellFeature(
    [:C];
    p2=2:6,              # Two-body powers
    p3=2:5,              # Three-body radial powers
    q3=1:3,              # Three-body angular powers
    rcut2=4.0,           # Two-body cutoff
    rcut3=3.5,           # Three-body cutoff
    geometry_sequence=true  # Geometric spacing
)
```

### Parameters

- **`p2`**: Two-body powers (default 2:8) 
- **`p3`**: Three-body radial powers (default 2:8) 
- **`q3`**: Three-body angular powers (default 2:8)
- **`rcut2`, `rcut3`**: Cutoff radii - system-dependent, the default is 4.0.
- **`geometry_sequence`**: Use Geometric spacing for powers.

### Feature Count

Total features = (two-body) + (three-body):

```julia
cf = CellFeature([:C]; p2=2:6, p3=2:5, q3=1:3)

n_twobody = length(2:6)  # 5
n_threebody = length(2:5) * length(1:3)  # 4 × 3 = 12
println("Total: ", nfeatures(cf))  # 17
```

## Practical Guidelines

### Do's

✓ Start with default CellFeature. See [`CellFeatureConfig`](@ref)
✓ Always use three-body features
✓ Check cutoff captures relevant neighbors
✓ Match cutoff to training data
✓ Test on held-out data

### Don'ts

✗ Don't use only two-body features
✗ Don't use huge cutoffs without justification
✗ Don't add complexity without testing
✗ Don't use different features for train and test
✗ Don't forget to save your CellFeature

## Example: Comparing Feature Sets

```julia
using EDDPotentials, Flux, ExtXYZ
import EDDPotentials: train!, FluxInterface, train_val_split

# Load data
structures = ExtXYZ.load("<some_dataset>.xyz")
sc = StructureContainer(structures)

# Test three feature sets
feature_sets = [
    ("Minimal", CellFeature([:C]; p2=3:5, p3=3:4, q3=1:2)),
    ("Standard", CellFeature([:C])),
    ("Detailed", CellFeature([:C]; p2=2:7, p3=2:6, q3=1:3, rcut2=4.5))
]

for (name, cf) in feature_sets
    println("\n=== $name ($(nfeatures(cf)) features) ===")

    fc = FeatureContainer(cf, sc)
    fc_train, fc_val = train_val_split(fc)

    chain = Chain(Dense(nfeatures(cf), 20, tanh), Dense(20, 1))
    model = FluxInterface(chain)
    train!(model, fc_train, fc_val; maxIter=300, show_progress=false, earlystop=30)

    results = TrainingResults(model, fc_val)
    println("Val RMSE: $(rmse_per_atom(results)) eV/atom")
end
```

## Next Steps

- [AIRSS Workflow](../tutorials/airss_workflow.md) - Apply features to structure prediction
