# EDDPotentials.jl Changelog

## Unreleased

### `CellFeature` refactored to parametric immutable struct

`CellFeature` is now a parametric immutable struct (`CellFeature{T, F2, G2, F3, G3, NELEM, NP2, NP3, NQ3}`) that centralizes two-body and three-body parameters (`p2`/`p3`/`q3`/`rcut2`/`rcut3`/`f2`/`g2`/`f3`/`g3`). The keyword constructor API is unchanged.

**Removed types:** `TwoBodyFeature`, `ThreeBodyFeature`, `AbstractNBodyFeature`, `permequal`.

**New lookup methods (replacing `FeatureLookup`/`FeatureSpeciesLookup`):**
| Method | Description |
|--------|-------------|
| `spec_index(cf, sym)` | Symbol → integer mapping |
| `spec_symbol(cf, idx)` | Integer → Symbol mapping |
| `two_body_channel(cf, ci, cj)` | O(1) channel index |
| `two_body_range(cf, ci, cj)` / `two_body_range(cf, si, sj)` | Feature range |
| `three_body_channel(cf, ci, cj, ck)` | Channel index, symmetric in cj↔ck |
| `three_body_range(cf, ci, cj, ck)` / `three_body_range(cf, si, sj, sk)` | Feature range |
| `nfeatures_per_channel(cf, bodyorder)` | Features per species channel |
| `nchannels(cf, bodyorder)` | Number of species channels |
| `total_two_body_features(cf)` | Helper: `NELEM² * NP2` |

**`BodyEmbedding`** has a new constructor: `BodyEmbedding(T, nchannels, feat_per_channel, n_embed)`.

**`FeatureProto`** is a lightweight struct replacing `TwoBodyFeature`/`ThreeBodyFeature` for `RandomEmbeddedCellFeature` internal use.

*Backward compatibility:* The `lookup(cf, s)`, `lookup(cf, s1, s2)`, `lookup(cf, s1, s2, s3)` functions and the `CellFeature(elements; p2=..., ...)` keyword constructor remain unchanged. Old JSON artifacts with `"two_body"/"three_body"` sub-objects are auto-detected and converted on deserialization.

**Migration from `cf.two_body` / `cf.three_body` field access:**

| Old pattern | New pattern |
|-------------|-------------|
| `cf.two_body[1].p` | `cf.p2` |
| `cf.two_body[1].rcut` | `cf.rcut2` |
| `cf.three_body[1].p` | `cf.p3` |
| `cf.three_body[1].q` | `cf.q3` |
| `cf.three_body[1].rcut` | `cf.rcut3` |
| `cf.two_body[1].f` | `cf.f2` |
| `length(cf.two_body[1].p)` | `NP2` (via type params) or `length(cf.p2)` |
| `cf.two_body[ife2].p[m]` | `cf.p2[m]` |
| `cf.three_body[ife3].np` | `NP3` (via type params) |
| `cf.three_body[ife3].nq` | `NQ3` (via type params) |
| `nfeatures(cf.two_body[1])` | `nfeatures_per_channel(cf, 2)` |
| `nfeatures(cf.three_body[1])` | `nfeatures_per_channel(cf, 3)` |
| Iterating `cf.two_body` tuple | Use `two_body_range(cf, ci, cj)` / `two_body_channel(cf, ci, cj)` |
| Iterating `cf.three_body` tuple | Use `three_body_range(cf, ci, cj, ck)` / `three_body_channel(cf, ci, cj, ck)` |

### `train_method` now uses Symbols instead of Strings

`train_model` and `train!` now accept `train_method` as a Symbol:

```julia
# Before
train_model("data/*.res", cf; train_method="lm_batched")

# After
train_model("data/*.res", cf; train_method=:lm_batched)
```

Valid values: `:lm`, `:lm_batched`, `:sgd`, `:optim`.

*Backward compatibility:* passing a String to `train!` still works via automatic
`Symbol(train_method)` conversion.

### `maxIter` renamed to `max_epochs`

The `maxIter` parameter is deprecated in `train_lm!` in favour of `max_epochs`,
matching `train_lm_batched!` and `train_sgd!`. The old keyword issues a warning.

### All training parameters explicitly documented

The core `train_model(fc_train, fc_val, fc_test; ...)` method no longer uses
`kwargs...` forwarding. Every training keyword argument is explicitly listed
with type and default, organized by category (common, LM, LM batched, SGD,
checkpointing). See the method's docstring for the full list.

### Shared docstring for convenience wrappers

All 9 convenience `train_model` overloads now share a single docstring via
`const _TRAIN_MODEL_WRAPPER_DOC` + `@doc`, reducing duplication.
