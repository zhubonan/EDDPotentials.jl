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

## Feature Vector Storage Scheme

The feature vector for each atom is stored as a column in the `(nfeatures × natoms)` matrix `fvec`. The layout differs between the standard `CellFeature` and the embedded `RandomEmbeddedCellFeature` / `MultiRandomEmbeddedCellFeature`, but all share the same basic pattern: one-body features → two-body features → three-body features.

### Standard CellFeature Layout

For a `CellFeature` with `N` elements, `M` two-body feature specs, and `K` three-body feature specs:

```
fvec[i, iat] layout (per atom):
  i=1..N            →  one-body (one-hot species encoding, offset=0)
  i=N+1..N+Σnp₂    →  two-body features
  i=...rest...       →  three-body features
```

**One-body** (`cell.jl:260-280`): A one-hot vector of length `one_body_offset = N`. For atom of species `s`, `fvec[ispec, iat] = 1.0` where `ispec` is the index of `s` in `cf.elements`.

**Two-body** (`gradient.jl:379-464`): Features are grouped by `TwoBodyFeature` spec. The specs are listed in `cf.two_body` as a tuple, ordered by all `(i, j)` element pairs. For each spec, `np` powers are stored contiguously. During computation, the species pair of the central atom `iat` and neighbor `jat` is matched via `permequal` against each spec; if it matches, `np` entries are filled; otherwise skipped by advancing the index `i += np`.

Example for `[:C, :H]` with `p2=2:4`:
```
TwoBodyFeature specs tuple order:
  (C,C) (C,H) (H,C) (H,H)   — 4 specs × 3 powers = 12 features

fvec layout after one-body [C, H]:
  [C-C_p2, C-C_p3, C-C_p4, C-H_p2, C-H_p3, C-H_p4,
   H-C_p2, H-C_p3, H-C_p4, H-H_p2, H-H_p3, H-H_p4]
```

**Three-body** (`gradient.jl:505-684`): Similar grouping by `ThreeBodyFeature` spec. Specs are listed in `cf.three_body` as a tuple ordered by center element `i` first, then unordered neighbor pair `(j,k)` with `j ≤ k`. Each spec has `np × nq` features stored in **q-fastest** order: `(m=1, o=1..nq, m=2, o=1..nq, ...)`.

Example for `[:C]` with `p3=2:3, q3=1:2`:
```
ThreeBodyFeature spec: (C,C,C) — 1 spec × 2 powers × 2 q-powers = 4 features

fvec layout after two-body:
  [m1_o1, m1_o2, m2_o1, m2_o2]
```

### Single Embedding Layout (`RandomEmbeddedCellFeature`)

`RandomEmbeddedCellFeature` replaces species one-hot with learnable embedding vectors (`embedded.jl`).

```
fvec[i, iat] layout (per atom):
  i=1..nembed                       →  one-body (embedding vector)
  i=nembed+1..nembed+n₂_channels×np₂  →  two-body features
  i=...rest...                        →  three-body features
```

**One-body** (`embedded.jl:211-218`): Embedding vector of length `nembed`. For atom of species `s`, `fvec[1:nembed, iat] = embedding[spec_index[s], :]`. `one_body_offset = nembed`.

**Two-body** (`embedded.jl:321-344`): Organized by **channel first**, then power. Pair channels enumerate all ordered pairs `(a, b)` where `a, b ∈ 1..nembed` (total `nembed²` channels). For channel `(a, b)` at index `ich`, the value is `raw × embedding[si, a] × embedding[sj, b]`. The `n2proto = nfeatures(feat2) = np₂` powers are stored contiguously per channel:

```
[ch(1,1)_p1, ch(1,1)_p2, ..., ch(1,1)_p{n₂},
 ch(1,2)_p1, ch(1,2)_p2, ..., ch(1,2)_p{n₂},
 ...,
 ch(n,n)_p1, ch(n,n)_p2, ..., ch(n,n)_p{n₂}]
```

**Three-body** (`embedded.jl:379-416`): Organized by **triplet channel first**. Triplet channels enumerate all `(a, (b, c))` where `a ∈ 1..nembed` and `(b, c)` is an unordered pair with `b ≤ c` (total `nembed × nembed×(nembed+1)/2` channels). For channel `(a, b, c)` at index `ich`, the feature value is `raw × wtrip` where `wtrip` is the triplet embedding weight (`embedded.jl:235-250`). Within each channel, `n3proto = np₃ × nq₃` features are stored in **q-fastest** order:

```
[ch1_m1_o1, ch1_m1_o2, ..., ch1_m1_o{nq₃},
 ch1_m2_o1, ch1_m2_o2, ..., ch1_m2_o{nq₃},
 ...,
 ch1_m{n₃}_o{nq₃},
 ch2_...
 ...]
```

### Multi-Embedding Layout (`MultiRandomEmbeddedCellFeature`)

`MultiRandomEmbeddedCellFeature` (`multi_embedded.jl`) concatenates `M` independent `RandomEmbeddedCellFeature` instances. The total feature vector is:

```
[emb1_ob | emb1_2b | emb1_3b | emb2_ob | emb2_2b | emb2_3b | ...]
```

Key difference: `one_body_offset = 0`, since each embedding's one-body features are interleaved with its n-body features (unlike single embedding where all one-body features precede all n-body features). The `reset!(workspace)` function zeros all n-body features (`fvec[1:end, :] .= 0`), then `one_body_vectors!` repopulates one-body sections (`multi_embedded.jl:123-130`).

Each embedding is computed independently via `compute_embedded_fv_gv_one!` with its own `emb_offset = _nbody_offset(cf, k)` (`multi_embedded.jl:108-111`). Hard-core repulsion is only computed for the first embedding to avoid double-counting (`compute_hardcore=(k == 1)`).

### FeatureSpeciesLookup — O(1) Feature Dispatch

The standard `CellFeature` path (non-embedded) uses `FeatureSpeciesLookup` (`gradient.jl:24-47`) to accelerate feature computation from O(N_features × N_neighbors) to O(1) per neighbor pair/triplet. Without it, the code must iterate over all feature specs and test `permequal` for every neighbor interaction. The lookup does the reverse — given the species of the atoms involved, it directly returns only the feature specs and fvec offsets that apply.

**Structure:**
- `spec_to_idx`: `Dict{Symbol → Int}` mapping element symbol to its index in the element list.
- `_2b_data[ci, cj]`: `Vector{Tuple{Int, Int}}` — for species pair `(ci, cj)`, a list of `(start_offset, feature_index)` tuples. `start_offset` is the 1-indexed position in fvec (relative to `offset`), and `feature_index` identifies which `features2[ife2]` to use.
- `_3b_data[ci, cj, ck]`: Same structure for three-body, but with double registration for unordered neighbors.
- `totalfe2`: Total number of two-body feature slots, used as the starting offset for the three-body section.

**Construction** (`gradient.jl:29-46`):
```
offset = 1
for (ife, f) in enumerate(features2)
    ci, cj = species indices of f.sij_idx
    push!(_2b_data[ci, cj], (offset, ife))
    offset += f.np      # advance by number of powers in this spec
end
totalfe2 = offset - 1   # snapshot total two-body feature count

offset = 1
for (ife, f) in enumerate(features3)
    ci, cj, ck = species indices of f.sijk_idx
    push!(_3b_data[ci, cj, ck], (offset, ife))
    cj != ck && push!(_3b_data[ci, ck, cj], (offset, ife))  # double register
    offset += f.np * f.nq
end
```

**Two-body dispatch** (`gradient.jl:379-420`): For a center atom `iat` (species `ci`) and neighbor `jat` (species `cj`):
```
_2b_data[ci, cj] → [(start2_1, ife2_1), (start2_2, ife2_2), ...]
```
Then for each `(start2, ife2)`, features are written to `fvec[offset + start2, iat]` with `np` consecutive slots. This is O(number of matching specs for that species pair) rather than O(total specs).

**Three-body dispatch** (`gradient.jl:506-599`): For a triplet with center `ci`, neighbors `cj` and `ck`:
```
_3b_data[ci, cj, ck] → [(start3_1, ife3_1), (start3_2, ife3_2), ...]
```
Because of the double registration (`cj != ck` case), both `_3b_data[ci, cj, ck]` and `_3b_data[ci, ck, cj]` contain the same entries — so the lookup works regardless of which neighbor order is encountered in the loop. Features are written to `fvec[offset + totalfe2 + start3, iat]` with `np × nq` consecutive slots (q-fastest). A `permequal` guard inside the loop still verifies the species tuple matches (`gradient.jl:516-518`), since the lookup key `(ci, cj, ck)` is coarser than the exact triplet `(center, neighbor1, neighbor2)` — for example, `_2b_data[C, C]` would contain both `(C,C)` and `(C,C)` specs which pass, as well as potentially a `(C,__)` spec that doesn't match the second species.

### Gradient Workspace Array Shapes

- **`fvec`**: `(nfeatures, natoms)` — feature vectors per atom
- **`gvec`**: `(ndims, nfeatures, nn_max, natoms)` — feature gradients `∂F_i / ∂r_j` where `j` ranges over the center atom (index 1) and its unique neighbors (indices 2..nn). `nn_max` is the max number of unique neighbors + 1 across all atoms. `gvec_index[:, iat]` maps local neighbor slot to actual atom index.
- **`stotv`**: `(3, 3, nfeatures, natoms)` — strain derivatives `∂F / ∂ε` for stress computation

### Chain Rule: Forces and Stress

Forces are obtained via `_force_update!` (`gradient.jl:1007-1025`):

```
forces[elm, jat] = -Σᵢ Σₓ gvec[elm, x, j_local, iat] × gv[x, iat]
```

where `gv = dE/d(fvec)` is the upstream gradient from the neural network. Stress is obtained via `_stress_update!` (`gradient.jl:1032-1047`):

```
stress[a, b] = Σᵢₐₜ Σₓ stotv[a, b, x, iat] × gv[x, iat]
```

## Next Steps

- [AIRSS Workflow](../tutorials/airss_workflow.md) - Apply features to structure prediction
