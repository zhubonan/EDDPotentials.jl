# EDDPotentials.jl Changelog

## Unreleased

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
