# Batched LM Training

!!! warning "Experimental Feature"
    Batched LM training is a new experimental feature. The API may change in future releases. It has been tested on datasets with up to ~12,000 structures, but edge cases may exist. Feedback and bug reports are welcome.

The standard Levenberg-Marquardt (LM) trainer computes the full Jacobian over all training data at every iteration. This scales as O(m·n²) per step (m = number of structures, n = number of parameters), which becomes slow for large datasets. Batched LM operates on random mini-batches instead, reducing per-step cost to O(B·n²) where B is the batch size.

## When to Use Batched LM

**Use batched LM when:**
- Training dataset has more than 2,000–3,000 structures
- Training time with standard LM exceeds acceptable limits
- Using larger network architectures (more parameters)

**Stick with standard LM when:**
- Dataset is small (< 1,000 structures) — standard LM is fast enough
- You need the most accurate convergence — standard LM uses exact second-order information

## How It Works

Batched LM breaks training into **epochs**. Each epoch:

1. Shuffles the training data randomly
2. Divides data into mini-batches of size `batch_size`
3. For each batch, computes the normal equation `J_b'J_b + λ·diag(DtD)` from the batch Jacobian `J_b` only
4. Solves for the parameter update and evaluates step quality on the batch
5. After all batches, evaluates full-dataset RMSE for progress tracking and early stopping

This is analogous to stochastic gradient descent, but using the LM trust-region framework instead of gradient descent.

## Quick Start

### Julia API

```julia
using EDDPotentials
using EDDPotentials.EDDPotentialsCore: ManualFluxBackPropInterface

# Load and prepare data
sc = StructureContainer(["data/*.res"])
cf = CellFeature([:C]; p2=2:8, rcut2=5.0)
fc = FeatureContainer(sc, cf)
fc_train, fc_test = split(fc, 450, 50)

# Create model
itf = ManualFluxBackPropInterface(
    cf, 30, 30;
    xt=fc_train.xt, yt=fc_train.yt, apply_xt=false
)

# Train with batched LM
x_train, y_train = get_fit_data(fc_train)
x_test, y_test = get_fit_data(fc_test)

opt_res, params, history, funcs = train_lm_batched!(
    itf, x_train, y_train;
    x_test, y_test,
    batch_size=512,
    maxIter=2000,
    earlystop=50,
    show_progress=true,
)
```

### Via `train!` dispatch

```julia
train!(itf, fc_train, fc_test;
    train_method="lm_batched",
    batch_size=512,
    maxIter=2000,
    earlystop=50,
)
```

### In a `link.toml` workflow

Add `batch_size` to the `[trainer]` section:

```toml
[trainer]
elements = ["C"]
n_nodes = [30, 30]
max_iter = 2000
batch_size = 512    # Set > 0 to activate batched LM
```

Setting `batch_size = 0` (the default) keeps the standard full-dataset LM.

## Choosing the Batch Size

The batch size must be **at least as large as the number of model parameters** for well-conditioned normal equations. If `batch_size < nparams`, the batch Jacobian is rank-deficient and the solver may fail with a `SingularException`.

**Guidelines:**

| Model architecture | Approx. params | Minimum batch size | Recommended batch size |
|---|---|---|---|
| `[8]` | ~250 | 256 | 256–512 |
| `[30]` | ~1,000 | 1,024 | 1,024–2,048 |
| `[30, 30]` | ~1,900 | 2,048 | 2,048–4,096 |
| `[50, 50]` | ~5,300 | 5,632 | 8,192 |

To check the number of parameters:

```julia
using EDDPotentials.EDDPotentialsCore: nparams
println("Parameters: ", nparams(itf))
```

Larger batch sizes give more stable convergence but reduce the speed advantage. A good starting point is `batch_size ≈ 4 × nparams`.

## Comparison with Standard LM

```julia
using EDDPotentials
using EDDPotentials.EDDPotentialsCore: ManualFluxBackPropInterface, reinit

# Standard LM
itf_lm = ManualFluxBackPropInterface(cf, 30, 30;
    xt=fc_train.xt, yt=fc_train.yt, apply_xt=false)
train!(itf_lm, fc_train, fc_test; maxIter=500, show_progress=true)

# Batched LM
itf_blm = reinit(itf_lm)  # fresh copy with same architecture
train!(itf_blm, fc_train, fc_test;
    train_method="lm_batched", batch_size=2048,
    maxIter=500, show_progress=true)

# Compare
tr_lm = TrainingResults(itf_lm, fc_test)
tr_blm = TrainingResults(itf_blm, fc_test)
println("Standard LM RMSE: ", rmse_per_atom(tr_lm), " eV/atom")
println("Batched  LM RMSE: ", rmse_per_atom(tr_blm), " eV/atom")
```

Typical results on datasets with 5,000+ structures show batched LM achieves comparable accuracy (within 10–30% RMSE) with significantly reduced wall-clock time per iteration.

## Parameters Reference

| Parameter | Default | Description |
|---|---|---|
| `batch_size` | 512 | Number of samples per mini-batch. Must be positive and ideally ≥ `nparams(model)`. |
| `maxIter` | 1000 | Maximum total LM iterations (across all batches and epochs). |
| `earlystop` | 50 | Stop if full-dataset test RMSE hasn't improved for this many epochs. |
| `shuffle_seed` | 42 | RNG seed for reproducible data shuffling. |
| `p` | 1.25 | Power for iterative weight reweighting. Same as standard LM. |
| `keep_best` | true | Rewind to best parameters at end. |
| `show_progress` | false | Print RMSE each epoch. |

All other LM parameters (`lambda`, `lambda_increase`, `lambda_decrease`, `min_step_quality`, `good_step_quality`, `x_tol`, `g_tol`) behave identically to the standard LM solver.

## Debugging

Enable Julia's debug logging to see per-batch details:

```bash
JULIA_DEBUG=EDDPotentials julia --project=. train_script.jl
```

This shows:
- Per-batch step quality (`rho`), residual, and lambda updates
- Epoch-level best residual tracking
- Convergence flag status

## Limitations

- **GPU support is not yet available** for batched LM. GPU acceleration will be added in a future release.
- Convergence may be slower than standard LM for small datasets where full-batch is feasible.
- The step quality ratio (`rho`) is computed per-batch and can be noisy, especially with small batch sizes. This may cause lambda oscillations. If you observe this, increase the batch size.
