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

1. Shuffles the training data randomly (or shuffles batch order for disk-backed features)
2. Divides data into mini-batches of size `batch_size`
3. For each batch, computes the normal equation `(J_b'WJ_b + λ·D)·Δx = -J_b'Wf_b` from the batch Jacobian `J_b`
4. Evaluates step quality on the batch; if rejected, retries with increased damping up to `attempts_per_step` times
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
fc_train, fc_val = split(fc, 450, 50)

# Create model
itf = ManualFluxBackPropInterface(
    cf, 30, 30;
    xt=fc_train.xt, yt=fc_train.yt
)

# Train with batched LM
x_train, y_train = get_fit_data(fc_train)
x_val, y_val = get_fit_data(fc_val)

opt_res, params, history, funcs = train_lm_batched!(
    itf, x_train, y_train;
    x_val, y_val,
    batch_size=512,
    max_epochs=200,
    earlystop=50,
    show_progress=true,
)
```

### Via `train!` dispatch

```julia
train!(itf, fc_train, fc_val;
    train_method="lm_batched",
    batch_size=512,
    max_epochs=200,
    earlystop=50,
)
```

### In a `link.toml` workflow

Add `batch_size` to the `[trainer]` section:

```toml
[trainer]
elements = ["C"]
n_nodes = [30, 30]
max_iter = 200       # max epochs for batched LM
batch_size = 512     # Set > 0 to activate batched LM
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
    xt=fc_train.xt, yt=fc_train.yt)
train!(itf_lm, fc_train, fc_val; maxIter=500, show_progress=true)

# Batched LM
itf_blm = reinit(itf_lm)  # fresh copy with same architecture
train!(itf_blm, fc_train, fc_val;
    train_method="lm_batched", batch_size=2048,
    max_epochs=500, show_progress=true)

# Compare
tr_lm = TrainingResults(itf_lm, fc_val)
tr_blm = TrainingResults(itf_blm, fc_val)
println("Standard LM RMSE: ", rmse_per_atom(tr_lm), " eV/atom")
println("Batched  LM RMSE: ", rmse_per_atom(tr_blm), " eV/atom")
```

Typical results on datasets with 5,000+ structures show batched LM achieves comparable accuracy (within 10–30% RMSE) with significantly reduced wall-clock time per iteration.

## Parameters Reference

### Core Parameters

| Parameter | Default | Description |
|---|---|---|
| `batch_size` | 512 | Number of samples per mini-batch. Must be ≥ `nparams(model)`. |
| `max_epochs` | 300 | Maximum number of training epochs. Each epoch processes all batches once. |
| `earlystop` | 50 | Stop if validation RMSE hasn't improved for this many epochs. Set to `0` to disable. |
| `p` | 1.25 | Power for iterative weight reweighting. Same as standard LM. |
| `keep_best` | `true` | Rewind to best parameters at end based on validation RMSE. |
| `show_progress` | `false` | Print RMSE each epoch. |
| `shuffle_seed` | 42 | RNG seed for reproducible data shuffling. |

!!! note
    The old `maxIter` parameter (counting individual batch steps) is deprecated. Use `max_epochs` instead. Passing `maxIter` still works but emits a deprecation warning.

### Damping Mode

Controls how the Levenberg-Marquardt damping term is applied to the Gauss-Newton Hessian `J'J`. Set via `damping_mode` keyword:

| Mode | Formula | Behaviour |
|---|---|---|
| `:fletcher` (default) | `J'J + λ·diag(J'J)` | Scale-aware — each parameter's damping is proportional to its curvature. Good default for most cases. |
| `:standard` | `J'J + λ·I` | Identity damping — simpler but not scale-invariant. Can struggle when parameters have very different magnitudes. |
| `:adaptive` | `J'J + λ·max(diag(J'J))·I` | Scales damping by the largest diagonal element. A compromise between the two. |

```julia
train_lm_batched!(model, x, y;
    damping_mode=:adaptive,
    max_epochs=100,
)
```

### Attempts Per Step

When a batch step is rejected (doesn't reduce the residual), the solver can **retry** the same batch with increased damping instead of immediately moving on. This gives each batch multiple chances to find a good step.

The `J'WJ` matrix is computed once and cached — only the damping diagonal changes per attempt, so retries are cheap.

```julia
train_lm_batched!(model, x, y;
    attempts_per_step=5,   # retry up to 5 times per batch
    max_epochs=100,
)
```

| `attempts_per_step` | Trade-off |
|---|---|
| `1` (default) | Current behaviour — move on after one rejected step |
| `3–5` | Good balance — more robust on noisy data with minimal overhead |
| `10+` | Aggressive — rarely needed, can slow down epochs on easy data |

### Solver Choice

Controls how the damped normal equations `(J'J + λ·D)·Δx = -J'f` are solved. Set via `solver` keyword:

| Solver | Method | Characteristics |
|---|---|---|
| `:direct` (default) | LU decomposition via `\` | Balanced speed and robustness. Works for all matrices. |
| `:cholesky` | Cholesky factorisation | Faster for well-conditioned positive-definite systems. Falls back to `:direct` if the matrix is not positive-definite. |
| `:qr` | QR decomposition | Most robust for ill-conditioned systems. Slower but more numerically stable. |

```julia
train_lm_batched!(model, x, y;
    solver=:cholesky,
    max_epochs=100,
)
```

### Passing Parameters from TOML Config

New parameters can be passed through `training_kwargs` in the `[trainer]` section of `link.toml`:

```toml
[trainer]
elements = ["C"]
n_nodes = [30, 30]
max_iter = 200
batch_size = 1024

[trainer.training_kwargs]
damping_mode = "adaptive"
attempts_per_step = 3
solver = "cholesky"
```

### Full Parameter List

| Parameter | Default | Description |
|---|---|---|
| `batch_size` | 512 | Samples per mini-batch |
| `max_epochs` | 300 | Maximum training epochs |
| `earlystop` | 50 | Epochs without improvement before stopping (`0` = disabled) |
| `damping_mode` | `:fletcher` | Damping strategy (`:standard`, `:fletcher`, `:adaptive`) |
| `attempts_per_step` | 1 | Retries per batch on rejected steps |
| `solver` | `:direct` | Linear solver (`:direct`, `:cholesky`, `:qr`) |
| `p` | 1.25 | Reweighting power |
| `keep_best` | `true` | Keep best validation parameters |
| `lambda` | 10 | Initial damping parameter |
| `tau` | `Inf` | Heuristic for initial lambda from J'J diagonal |
| `lambda_increase` | 2.0 | Factor to increase lambda on bad steps |
| `lambda_decrease` | 0.2 | Factor to decrease lambda on good steps |
| `min_step_quality` | 1e-3 | Step quality threshold for shrinking trust region |
| `good_step_quality` | 0.75 | Step quality threshold for expanding trust region |
| `x_tol` | 1e-8 | Step-size convergence tolerance |
| `g_tol` | 1e-12 | Gradient convergence tolerance |
| `shuffle_seed` | 42 | RNG seed for reproducibility |
| `show_progress` | `false` | Print RMSE each epoch |

## Debugging

Enable Julia's debug logging to see per-batch details:

```bash
JULIA_DEBUG=EDDPotentials julia --project=. train_script.jl
```

This shows:
- Per-batch step quality (`rho`), residual, and lambda updates
- Attempt number within each batch (when `attempts_per_step > 1`)
- Epoch-level best residual tracking
- Convergence flag status

## Limitations

- **GPU support is not yet available** for batched LM. GPU acceleration will be added in a future release.
- Convergence may be slower than standard LM for small datasets where full-batch is feasible.
- The step quality ratio (`rho`) is computed per-batch and can be noisy, especially with small batch sizes. This may cause lambda oscillations. If you observe this, increase the batch size.
