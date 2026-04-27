# Force-Informed Training via Taylor Expansion

EDDPotentials is designed for **energy-only** training — it learns from total
energy labels alone, without requiring expensive DFT force calculations during
training. However, when your DFT data *does* include atomic forces (common in
modern DFT workflows that compute forces at negligible extra cost), those
forces carry valuable gradient information that can improve model accuracy.

This guide describes how to incorporate force information into EDDP training
**without modifying the energy-only loss function**, using the synthetic
displacement method of Cooper et al. (2020).

## Why Force Information Matters

Training with energy-only labels has three failure modes where forces help:

1. **Sloped energy surfaces**: A model may predict the correct energy at a
   data point but the wrong slope (forces), leading to inaccurate relaxations.
2. **Data sparsity**: Energy labels alone may not constrain the model enough
   in sparse regions of configuration space.
3. **Interpolation quality**: Forces provide local curvature information that
   improves generalization between training points.

By augmenting the training set with synthetic structures whose energies encode
force information, your model learns both energy *and* force behaviour while
using the same energy-only training pipeline.

## The Taylor Expansion Approach

Cooper et al. (2020)[^1] proposed an elegant method: instead of adding a
separate force-prediction loss term, inject force information into the
training set by generating **synthetic structures** from existing DFT data.

The procedure for each parent structure with known forces ``\mathbf{F} =
-\nabla E``:

1. **Apply random displacements** ``\Delta\mathbf{r}_i`` to each atom ``i``,
   drawn uniformly on the unit sphere with magnitude uniformly in ``[0,
   \Delta_{\max}]``.
2. **Compute the synthetic energy** via a first-order Taylor expansion:

   ```math
   E_{\text{synth}} = E_{\text{parent}} - \sum_i \mathbf{F}_i \cdot \Delta\mathbf{r}_i
   ```

3. **Treat each displaced structure** as a new energy-only training datum with
   label ``E_{\text{synth}}``.

To first order, a model trained on ``\{\text{displaced structure},
E_{\text{synth}}\}`` implicitly learns the force information because the energy
difference encodes the dot product of forces with the displacement.

The original parent structures are kept alongside the synthetic ones (by
default), so the training set grows from ``N`` to ``N \times (1 +
n_{\text{synthetic}})`` structures carrying force arrays.

!!! note "What this method does *not* do"
    - It does **not** add a joint energy+force loss function.
    - It does **not** require calculating model forces during training.
    - It is **not** a replacement for explicit force training — it is a
      computationally cheaper alternative that works within the existing
      energy-only LM pipeline.

## Prerequisites

Your DFT data must include atomic forces. EDDPotentials reads forces from
`cell.arrays` under the key `:REF_forces`. When loading extxyz files, forces
are automatically normalized — if the file contains a `forces` or `force`
column, it is stored under the `:REF_forces` key.

### Force array format

- Forces must be a ``(3 \times N_{\text{atoms}})`` matrix of `Float64` values
  in Cartesian coordinates.
- Force sign convention: ``\mathbf{F} = -\nabla E`` (negative energy
  gradient), matching the standard DFT convention.
- If a structure lacks the `:REF_forces` key, it is silently skipped (when
  `use_force_data=true`) and only its original entry is included.

## Quick Start

### Via `train_model` (Direct API)

Pass `use_force_data=true` to any `train_model` overload:

```julia
using EDDPotentials

# File paths — auto-loads forces from extxyz files if present
tm = train_model("data/*.res";
    use_force_data=true,
    nsynthetic=5,
    max_disp=0.05,
    nnodes=[20, 20],
)

# With explicit feature spec
tm = train_model("data/*.res", CellFeature([:Al, :N]);
    use_force_data=true,
    nsynthetic=5,
    max_disp=0.05,
)

# Pre-split StructureContainers
tm = train_model(sc_train, sc_val, sc_test, CellFeature([:C]);
    use_force_data=true,
    nsynthetic=5,
)
```

If none of your structures contain force arrays, a clear `ArgumentError` is
raised:

```
ArgumentError: `use_force_data=true` but none of the N structures carry a
`:REF_forces` array. Ensure the input data includes force arrays (e.g. from
DFT-calculated extxyz files with a `REF_forces` column).
```

### Via Builder (`link.toml`)

Add the following to your `link.toml` configuration under `[state]`:

```toml
[state]
use_synthetic_force_data = true
synthetic_force_n = 5
synthetic_force_max_disp = 0.05
```

The Builder workflow automatically reads these settings and applies synthetic
force data augmentation during each training cycle via `split_and_featurize`.

### Direct Call to `synthetic_force_data`

For interactive exploration or custom pipelines, call the core function
directly:

```julia
using EDDPotentials: synthetic_force_data

sc = StructureContainer("data/*.res")

# Generate synthetic data
sc_augmented = synthetic_force_data(sc;
    nsynthetic=3,
    max_disp=0.03,
    rng=MersenneTwister(42),
)

# sc_augmented contains originals + synthetic entries
println("Original structures: $(length(sc))")
println("After augmentation: $(length(sc_augmented))")
```

Synthetic structures carry metadata you can inspect:

```julia
cell = sc_augmented.structures[end]
cell.metadata[:label]                  # e.g., "parent-1-force-shake-3"
cell.metadata[:synthetic]              # true
cell.metadata[:synthetic_parent_label] # "parent-1"
cell.metadata[:synthetic_energy_delta] # energy change from displacement
cell.metadata[:synthetic_max_displacement]  # max single-atom displacement
```

## How Split Integrity Is Preserved

When `use_force_data=true`, the `split_and_featurize` pipeline automatically
enforces `respect_shakes=true`. This means all entries derived from the same
parent structure (original + synthetic shaked copies) are assigned to the
**same** train/validation/test split. This prevents data leakage that would
occur if a synthetic copy of a training structure ended up in the validation
set.

The split proportions are calculated based on the number of unique parent
structures (before augmentation), so adding synthetic structures does not
change the intended split ratios.

## Parameter Reference

| Parameter | Default | Description |
|---|---|---|
| `use_force_data` / `use_synthetic_force_data` | `false` | Enable synthetic force data augmentation |
| `nsynthetic` / `synthetic_force_n` | `1` | Number of synthetic structures per parent with forces |
| `max_disp` / `synthetic_force_max_disp` | `0.05` | Maximum per-atom displacement magnitude (Å) |
| `force_key` | `:REF_forces` | `cell.arrays` key for the force matrix |

### `synthetic_force_data` advanced options

| Parameter | Default | Description |
|---|---|---|
| `rng` | `Random.default_rng()` | Random number generator for reproducible displacements |
| `label_suffix` | `"-force"` | Suffix appended to parent label before shake index |
| `keep_original` | `true` | Whether to include original structures in output |
| `copy_metadata` | `true` | Whether to copy parent metadata onto synthetic structures |
| `skip_missing` | `false` | If true, silently skip structures without force arrays |

## Choosing `nsynthetic` and `max_disp`

### `nsynthetic` (number of synthetic copies per parent)

- **`1–3`**: Minimal augmentation. Suitable when you have many structures or
  limited training budget.
- **`5–10`**: Good default. Provides meaningful gradient information without
  excessive dataset growth.
- **`10–20`**: For small datasets (< 200 structures) where you want maximum
  force information.

More synthetic copies give the model more training data to learn the force
landscape. Returns diminish after ~10 copies per parent because the first-order
Taylor approximation error grows with displacement.

### `max_disp` (maximum displacement magnitude)

- **`0.02–0.05` Å**: Conservative. Stays within the linear regime of most
  interatomic potentials. Best for stiff systems (covalent, ionic).
- **`0.05–0.10` Å**: Moderate. Provides stronger gradient signal. Works well
  for softer systems (metallic, van der Waals).
- **`> 0.10` Å**: Aggressive. The first-order Taylor approximation may break
  down, introducing systematic energy errors. Only use if you have validated
  that the linear regime extends this far.

A good starting point is `max_disp=0.05` with `nsynthetic=5`. Monitor your
validation RMSE — if it increases relative to training without force data,
reduce `max_disp`.

## Verifying Force Data is Loaded

To check whether your extxyz files contain force data:

```julia
using EDDPotentials

sc = StructureContainer("data/*.res")
has_forces = [haskey(cell.arrays, :REF_forces) for cell in sc.structures]
n_with_forces = count(has_forces)
n_total = length(sc)
println("$n_with_forces / $n_total structures have force arrays")

# Inspect the force array of the first structure with forces
idx = findfirst(has_forces)
if idx !== nothing
    forces = sc.structures[idx].arrays[:REF_forces]
    println("Force array shape: $(size(forces))")
    println("Sample forces (atom 1): $(forces[:, 1])")
end
```

## Performance Considerations

- **Dataset growth**: With `nsynthetic=5` and `keep_original=true`, the
  effective training set grows by 5× for structures with force arrays. This
  increases feature computation and training time.
- **Feature computation**: Synthetic structures are full copies, so feature
  generation scales linearly with the expanded dataset size.
- **Memory**: `synthetic_force_data` creates deep copies of each cell. For
  large datasets, consider using `split_and_featurize` (which splits before
  featurizing) to manage memory.

## References

[^1]: Cooper, A. M., Kästner, J., Urban, A., & Artrith, N. (2020). Efficient
      training of ANN potentials by including atomic forces via Taylor
      expansion. *npj Computational Materials*, **6**, 54.
      DOI: [10.1038/s41524-020-00359-z](https://doi.org/10.1038/s41524-020-00359-z)
